import cv2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import json
import os
from torchvision import transforms #added

from Models import SLTModel, convert_to_deploy
from Utils.pose_extractor import PoseKeypointExtractor
from Utils.keypoint_utils import sequence_to_gcn_tensor
from train import greedy_decode, ctc_attention_decode, _ids_to_str

# Constants
MAX_SENTENCE_LEN = 60  # Max number of glosses to predict per window (fallback)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16

def load_model_and_vocab(checkpoints_dir="checkpoints"):
    """
    Loads the model, configuration, and vocabulary from the training checkpoints.
    """
    # Load training config
    config_path = os.path.join(checkpoints_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}. Please run training to generate it.")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Load vocab
    vocab_path = os.path.join(checkpoints_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found at {vocab_path}. Please run training to generate it.")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_inv = {int(v): k for k, v in vocab.items()}

    # Instantiate CLIP model based on training config
    clip_model, _ = clip.load(model_config["clip_model_name"], device=DEVICE)

    # Instantiate SLTModel with saved config
    model = SLTModel(
        clip_model=clip_model,
        vocab_size=model_config["vocab_size"],
        pad_idx=model_config["pad_idx"],
        bos_idx=model_config["bos_idx"],
        eos_idx=model_config["eos_idx"],
        adapter_dim=model_config["adapter_dim"],
        adapter_layers=range(*model_config["adapter_layers"]),
        clip_frames=model_config["clip_frames"],
        clip_spatial=model_config["clip_spatial"],
    )

    # Load best model weights from training
    model_path = os.path.join(checkpoints_dir, "phase3_best_wer.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoints_dir, "phase2_best_wer.pth")

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        convert_to_deploy(model)
    else:
        raise FileNotFoundError("No trained model weights (phase3_best_wer.pth or phase2_best_wer.pth) found.")

    model.to(DEVICE)
    model.eval()

    # Load CTC head for hybrid decoding if available
    ctc_aux_head = None
    ctc_head_path = os.path.join(checkpoints_dir, "ctc_aux_head_phase3_best.pth")
    if os.path.exists(ctc_head_path):
        print("Loading CTC auxiliary head for hybrid decoding.")
        fused_dim = model.translator.d_model
        ctc_aux_head = nn.Linear(fused_dim, model_config["vocab_size"]).to(DEVICE)
        ctc_aux_head.load_state_dict(torch.load(ctc_head_path, map_location=DEVICE))
        ctc_aux_head.eval()

    return model, vocab, vocab_inv, ctc_aux_head

def preprocess_frames(frames, preprocess):
    processed = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(img))
        processed.append(img)

    tensor = torch.stack(processed)
    return tensor.unsqueeze(0)

def run_inference():
    model, vocab, vocab_inv, ctc_aux_head = load_model_and_vocab()
    
    # Get special token IDs from vocab
    pad_idx = vocab.get("<pad>", 0)
    blank_idx = vocab.get("<blank>", 1)
    bos_idx = vocab.get("<bos>", 2)
    eos_idx = vocab.get("<eos>", 3)
    special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}

    pose_extractor = PoseKeypointExtractor()
    cap = cv2.VideoCapture(0)
    
    frame_buffer = []
    kp_buffer = []

    # added: Mimic the training spatial transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    #-----------
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Extract and Buffer
        kp = pose_extractor.extract_keypoints(frame)
        frame_buffer.append(frame)
        kp_buffer.append(kp)

        # 2. Process when buffer is full (e.g., 16 or 32 frames)
        if len(frame_buffer) >= WINDOW_SIZE:
            #rgb_tensor = preprocess_frames(frame_buffer, model.rgb_encoder.clip_preprocess).to(DEVICE) 
            rgb_tensor = preprocess_frames(frame_buffer, preprocess).to(DEVICE) #added: In inference.py, you reference model.rgb_encoder.clip_preprocess. However, in clip.py there is no such attribute exposed. You need to recreate the Torchvision transform pipeline locally inside the inference file to avoid an AttributeError.
            # Shape (1, 2, T, 105)
            key_tensor = sequence_to_gcn_tensor(kp_buffer).to(DEVICE)

            with torch.no_grad():
                # We need the full model output dictionary
                out = model(rgb_tensor, key_tensor, tgt_tokens=torch.tensor([[bos_idx]], device=DEVICE))
                
                # Use Boundary Head to see if movement is a valid sign
                is_sign = out['boundary_probs'].item() > 0.5
                
                if is_sign:
                    # Use CTL head to determine max decoding length
                    max_len = MAX_SENTENCE_LEN # fallback
                    if 'ctl_logits' in out:
                        predicted_gloss_count = out['ctl_logits'].argmax(dim=-1).item() + 1
                        max_len = predicted_gloss_count + 5  # Add a buffer

                    # Use the appropriate decoding function
                    if ctc_aux_head is not None:
                        print("Using CTC-Attention hybrid decoding.")
                        decoded_ids = ctc_attention_decode(model, rgb_tensor, key_tensor, vocab, ctc_aux_head, max_decode_len=max_len)
                    else:
                        print("Using greedy decoding.")
                        decoded_ids = greedy_decode(model, rgb_tensor, key_tensor, vocab, max_decode_len=max_len)

                    # `decoded_ids` is a list of lists (one per batch item)
                    translation = _ids_to_str(decoded_ids[0], vocab_inv, special_ids)
                    print("Translated Sign:", translation)

            # Reset buffers
            frame_buffer, kp_buffer = [], []

if __name__ == "__main__":
    run_inference()