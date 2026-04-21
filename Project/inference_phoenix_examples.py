import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import shutil  # <-- Added for copying directories
from torchvision import transforms
from torchvision.io import read_image
from deep_translator import GoogleTranslator

# Assume these are imported from your project structure
from Models import SLTModel
from Utils.pose_extractor import PoseKeypointExtractor
from train_orig_run_new import greedy_decode, ctc_attention_decode, _ids_to_str
import clip
import json

import sys
custom_target_path = '/home/abdullahm/jaleel/CV_project'
sys.path.insert(0, custom_target_path)
try:
    from Datasets.rwth_phoenix_2014t import Phoenix14TDataset, cslr_collate_fn
    from Datasets.rwth_phoenix_2014 import Phoenix14Dataset
finally:
    sys.path.remove(custom_target_path)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MAX_SENTENCE_LEN = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
MAX_FRAMES = 300  # Added fallback for MAX_FRAMES used in dataset

# ─────────────────────────────────────────────────────────────────
# Model Loading (Reused from previous script)
# ─────────────────────────────────────────────────────────────────
def load_model_and_vocab(checkpoints_dir: str = "checkpoints_orig_run2"):
    vocab_path = os.path.join(checkpoints_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found at {vocab_path}. Run training first.")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_inv = {int(v): k for k, v in vocab.items()}
    vocab_size = len(vocab)
    print(f"Loaded vocab: {vocab_size} tokens.")

    clip_model, _ = clip.load("ViT-B/16", device=DEVICE)

    model = SLTModel(
        clip_model,
        vocab_size=vocab_size,
        pad_idx=vocab.get("<pad>", 0),
        bos_idx=vocab.get("<bos>", 2),
        eos_idx=vocab.get("<eos>", 3),
        adapter_dim=256,
        adapter_layers=range(2, 12),
        clip_frames=4,
        clip_spatial=14,
    )

    for candidate in ["phase3_best_wer.pth", "phase2_best_wer.pth", "phase3_latest.pth", "phase2_latest.pth"]:
        model_path = os.path.join(checkpoints_dir, candidate)
        if os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict)
            break
    else:
        raise FileNotFoundError(f"No trained weights found in {checkpoints_dir}.")

    model.to(DEVICE)
    model.eval()

    ctc_aux_head = None
    for ctc_candidate in ["ctc_aux_head_phase3_best.pth", "ctc_aux_head_best_wer.pth", "ctc_aux_head_phase3_latest.pth"]:
        ctc_path = os.path.join(checkpoints_dir, ctc_candidate)
        if os.path.exists(ctc_path):
            print(f"Loading CTC auxiliary head from: {ctc_path}")
            fused_dim    = model.translator.d_model
            ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(DEVICE)
            ctc_aux_head.load_state_dict(torch.load(ctc_path, map_location=DEVICE))
            ctc_aux_head.eval()
            break

    return model, vocab, vocab_inv, ctc_aux_head

# ─────────────────────────────────────────────────────────────────
# Random Inference Evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate_random_samples(checkpoints_dir: str, root_path: str, output_dir: str, num_samples: int = 5):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Path for the log file
    log_file_path = os.path.join(output_dir, "results.txt")
    video_folder = output_dir + '/Videos'

    # Helper function to print to console AND write to log file
    def log_and_print(message):
        print(message)
        with open(log_file_path, "a") as f:
            f.write(message + "\n")
            
    translator = GoogleTranslator(source='de', target='en')

    # 1. Load model and vocab
    model, vocab, vocab_inv, ctc_aux_head = load_model_and_vocab(checkpoints_dir)
    
    pad_idx   = vocab.get("<pad>",   0)
    blank_idx = vocab.get("<blank>", 1)
    bos_idx   = vocab.get("<bos>",   2)
    eos_idx   = vocab.get("<eos>",   3)
    special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}

    # 2. Initialize test dataset using the loaded vocab
    print(f"\nInitializing Test Dataset from: {root_path}")
    test_dataset = Phoenix14TDataset(
        root_dir=root_path, 
        split="test", 
        vocab=vocab, 
        is_training=False
    )
    
    if len(test_dataset) == 0:
        print("Dataset is empty. Please check your root_path and features/annotations directories.")
        return

    # 3. Select random indices
    selected_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    log_and_print(f"Evaluating {len(selected_indices)} random samples...")
    log_and_print(f"Saving outputs and logs to: {output_dir}")
    log_and_print("-" * 60)

    # 4. Run inference loop
    for i, idx in enumerate(selected_indices, 1):
        video_tensor, kp_stack, target_tensor, actual_len = test_dataset[idx]
        video_id = test_dataset.samples[idx]["video"]
        frame_paths = test_dataset.samples[idx]["paths"]
        video_dir_path = os.path.dirname(frame_paths[0]) if frame_paths else "No frames found"
        
        # ── Copy frames to the output directory ──
        if video_dir_path != "No frames found":
            dest_video_path = os.path.join(video_folder, video_id)
            if not os.path.exists(dest_video_path):
                shutil.copytree(video_dir_path, dest_video_path)
        else:
            dest_video_path = "Failed to locate original frames"
        
        # ── Replicate cslr_collate_fn padding ──
        T = video_tensor.shape[0]
        if T < MAX_FRAMES:
            pad_t = MAX_FRAMES - T
            video_tensor = torch.nn.functional.pad(
                video_tensor, (0, 0, 0, 0, 0, 0, 0, pad_t)
            )                                                          # (MAX_FRAMES, 3, 224, 224)
            kp_stack = torch.cat(
                [kp_stack, torch.zeros(kp_stack.shape[0], pad_t, kp_stack.shape[2])],
                dim=1
            )
        
        # Add batch dimension and move to device
        video_tensor = video_tensor.unsqueeze(0).to(DEVICE)       # (1, T, 3, 224, 224)
        kp_stack     = kp_stack.unsqueeze(0).float().to(DEVICE)   # (1, 2, T, 105)
        
        # Determine Ground Truth string
        gt_ids = target_tensor[1 : actual_len - 1].tolist()
        gt_str = _ids_to_str(gt_ids, vocab_inv, special_ids)
        
        with torch.no_grad():
            # Quick forward to get CTL bounds (optional, but good for max decoding length)
            tgt_seed = torch.tensor([[bos_idx]], dtype=torch.long, device=DEVICE)
            out = model(video_tensor, kp_stack, tgt_tokens=tgt_seed)
            
            max_len = MAX_SENTENCE_LEN
            if "ctl_logits" in out:
                predicted_count = out["ctl_logits"].argmax(dim=-1).item() + 1
                max_len = min(predicted_count + 5, MAX_SENTENCE_LEN)
            
            # Decode using CTC if available, else Greedy
            if ctc_aux_head is not None:
                decoded_ids = ctc_attention_decode(
                    model, video_tensor, kp_stack, vocab,
                    ctc_aux_head, max_decode_len=max_len,
                )
            else:
                decoded_ids = greedy_decode(
                    model, video_tensor, kp_stack, vocab,
                    max_decode_len=max_len,
                )

        # Determine Predicted string
        pred_str = _ids_to_str(decoded_ids[0], vocab_inv, special_ids)
        gt_str_en = translator.translate(gt_str) if gt_str.strip() else ""
        pred_str_en = translator.translate(pred_str) if pred_str.strip() else ""
        
        # Display comparison and log to file
        log_and_print(f"Sample {i}/{len(selected_indices)} (Dataset Index: {idx} | Frames: {actual_len})")
        log_and_print(f"Original Video Path: {video_dir_path}")
        log_and_print(f"Saved Video Path:    {dest_video_path}")
        log_and_print(f"GT (German):   {gt_str}")
        log_and_print(f"GT (English):   {gt_str_en}")
        log_and_print(f"PRED (German): {pred_str}")
        log_and_print(f"PRED (English): {pred_str_en}")
        log_and_print("-" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SLT Dataset Evaluation")
    parser.add_argument("--checkpoints", default="checkpoints_orig_run2", help="Dir containing vocab.json and weights")
    parser.add_argument("--data_root", default="/home/abdullahm/jaleel/CV_project/CLIP-SLA/Data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T", help="Path to PHOENIX dataset root")
    
    # <-- Added output_dir argument
    parser.add_argument("--output_dir", default="./Inference_Test_Samples", help="Path to save copied frames and results.txt") 
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to evaluate")
    args = parser.parse_args()
    
    evaluate_random_samples(
        checkpoints_dir=args.checkpoints, 
        root_path=args.data_root, 
        output_dir=args.output_dir, # <-- Passed new argument
        num_samples=args.num_samples
    )