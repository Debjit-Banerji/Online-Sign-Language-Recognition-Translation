"""
online_al_eval.py

Simulates online (streaming) inference on a subset of the PHOENIX-2014-T test
set by feeding frames one-by-one at a simulated 10 FPS.  Every WINDOW_SIZE
frames a decode is triggered (same as the webcam script).  Average Lagging is
computed from the frame index at which each predicted token was *first* emitted,
giving a meaningful latency score for an online-style system.

Usage:
    python online_al_eval.py \
        --checkpoints checkpoints_orig_run2 \
        --data_root /path/to/PHOENIX-2014-T \
        --num_samples 50 \
        --window_size 16 \
        --fps 10
"""

import os
import json
import random
import argparse
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import clip
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image

from Models import SLTModel
from train_orig_run_new import greedy_decode, ctc_attention_decode, _ids_to_str

import sys
custom_target_path = '/home/abdullahm/jaleel/CV_project'
sys.path.insert(0, custom_target_path)
try:
    from Datasets.rwth_phoenix_2014t import Phoenix14TDataset
finally:
    sys.path.remove(custom_target_path)


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MAX_SENTENCE_LEN = 60
MAX_FRAMES       = 300
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ─────────────────────────────────────────────────────────────────
# Model loading  (identical to other inference scripts)
# ─────────────────────────────────────────────────────────────────

def load_model_and_vocab(checkpoints_dir: str):
    vocab_path = os.path.join(checkpoints_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found at {vocab_path}.")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_inv  = {int(v): k for k, v in vocab.items()}
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
 
    for candidate in ["phase3_best_wer.pth", "phase2_best_wer.pth",
                       "phase3_latest.pth",   "phase2_latest.pth"]:
        path = os.path.join(checkpoints_dir, candidate)
        if os.path.exists(path):
            print(f"Loading weights: {path}")
            ckpt = torch.load(path, map_location=DEVICE)
            model.load_state_dict(ckpt.get("model", ckpt))
            break
    else:
        raise FileNotFoundError(f"No weights found in {checkpoints_dir}.")
 
    model.to(DEVICE).eval()
 
    ctc_aux_head = None
    for ctc_cand in ["ctc_aux_head_phase3_best.pth",
                     "ctc_aux_head_best_wer.pth",
                     "ctc_aux_head_phase3_latest.pth"]:
        ctc_path = os.path.join(checkpoints_dir, ctc_cand)
        if os.path.exists(ctc_path):
            print(f"Loading CTC head: {ctc_path}")
            fused_dim    = model.translator.d_model
            ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(DEVICE)
            ctc_aux_head.load_state_dict(torch.load(ctc_path, map_location=DEVICE))
            ctc_aux_head.eval()
            break
 
    if ctc_aux_head is None:
        print("No CTC head found — using pure greedy decoding.")
 
    return model, vocab, vocab_inv, ctc_aux_head
 
 
# ─────────────────────────────────────────────────────────────────
# Frame loading  (mirrors Phoenix14TDataset.__getitem__ preprocessing)
# ─────────────────────────────────────────────────────────────────
 
def load_frames(frame_paths: list, max_frames: int = MAX_FRAMES) -> list:
    """
    Load and preprocess PNG frames into a list of (3, 224, 224) tensors.
    Matches the exact pipeline in Phoenix14TDataset.__getitem__:
      read_image → /255 → Resize(224) → Normalize(CLIP_MEAN, CLIP_STD)
    """
    tensors = []
    for path in frame_paths[:max_frames]:
        img = read_image(path).float() / 255.0   # (3, H, W)  float32
        img = _preprocess(img)                    # (3, 224, 224)
        tensors.append(img)
    return tensors                                # list of (3, 224, 224)
 
 
def load_keypoints(kp_path: str, actual_len: int) -> np.ndarray:
    """
    Load precomputed keypoints, truncate/pad to actual_len.
    Returns (actual_len, 105, 2) float32 array.
    """
    if os.path.exists(kp_path):
        kp = np.load(kp_path).astype(np.float32)[:actual_len]  # (T, 105, 2)
        if kp.shape[0] < actual_len:
            pad = np.zeros((actual_len - kp.shape[0], 105, 2), dtype=np.float32)
            kp  = np.concatenate([kp, pad], axis=0)
        if kp.shape[1:] != (105, 2):
            kp = np.zeros((actual_len, 105, 2), dtype=np.float32)
    else:
        kp = np.zeros((actual_len, 105, 2), dtype=np.float32)
    return kp
 
 
# ─────────────────────────────────────────────────────────────────
# Average Lagging  (from STACL / simuleval)
# ─────────────────────────────────────────────────────────────────
 
def compute_AL(delays: list, source_length: int) -> float:
    """
    Correct Average Lagging (Ma et al. 2019)

    delays        : list of frame indices (1-based) when each token is committed
    source_length : total number of frames

    Uses hypothesis length (len(delays)) as target length.
    """
    if not delays or source_length == 0:
        return 0.0

    target_length = len(delays)
    gamma = target_length / source_length

    AL = 0.0
    for t, d in enumerate(delays, start=1):   # t = 1..T
        AL += d - (t - 1) / gamma

    return AL / target_length
 
 
# ─────────────────────────────────────────────────────────────────
# Build window tensor  (pad to MAX_FRAMES like the collate fn)
# ─────────────────────────────────────────────────────────────────
 
def _frames_to_tensor(frame_list: list, kp_list: list):
    """
    Convert a causal prefix of frames + keypoints to a padded
    (1, MAX_FRAMES, 3, 224, 224) / (1, 2, MAX_FRAMES, 105) tensor pair.
 
    Frames seen so far are placed at the START; the remaining positions
    are zero-padded — identical to cslr_collate_fn during training.
    This means the model always receives MAX_FRAMES input but only the
    first T positions contain real signal, which is exactly what it saw
    during training for short videos.
    """
    T = len(frame_list)
 
    # ── RGB ──
    video = torch.stack(frame_list)                           # (T, 3, 224, 224)
    if T < MAX_FRAMES:
        pad   = torch.zeros(MAX_FRAMES - T, 3, 224, 224)
        video = torch.cat([video, pad], dim=0)                # (MAX_FRAMES, 3, 224, 224)
    video = video.unsqueeze(0).to(DEVICE)                     # (1, MAX_FRAMES, 3, 224, 224)
 
    # ── Keypoints ──
    kp_np = np.stack(kp_list, axis=0)                         # (T, 105, 2)
    if kp_np.shape[0] < MAX_FRAMES:
        pad_kp = np.zeros((MAX_FRAMES - T, 105, 2), dtype=np.float32)
        kp_np  = np.concatenate([kp_np, pad_kp], axis=0)     # (MAX_FRAMES, 105, 2)
    kp = (torch.from_numpy(kp_np)
          .permute(2, 0, 1)                                   # (2, MAX_FRAMES, 105)
          .unsqueeze(0).float().to(DEVICE))                   # (1, 2, MAX_FRAMES, 105)
 
    return video, kp
 
 
# ─────────────────────────────────────────────────────────────────
# Per-video online inference  →  returns delay list for AL
# ─────────────────────────────────────────────────────────────────
 
def online_infer_video(
    model:         nn.Module,
    vocab:         dict,
    vocab_inv:     dict,
    ctc_aux_head,
    special_ids:   set,
    frame_tensors: list,       # list[Tensor(3,224,224)], length = actual_len
    kp_array:      np.ndarray, # (actual_len, 105, 2)
    window_size:   int,
    bos_idx:       int,
) -> tuple:
    """
    Simulate causal/online inference using a GROWING window.
 
    Every `window_size` frames a decode is triggered using ALL frames
    accumulated so far (not just the last window_size frames).  The future
    frames are zero-padded, matching the training distribution exactly.
 
    Why growing instead of sliding:
        The model was trained on full videos padded to MAX_FRAMES.  A sliding
        window of only `window_size` real frames leaves ~284 zeros, giving the
        encoder almost no usable signal → single-token outputs.  A growing
        window gives the model an increasing causal prefix, which is the
        correct simulation of online decoding for a non-recurrent model.
 
    AL semantics:
        delays[i] = frame index (1-based) at which token position i was
        *first* committed in any decode output.  Earlier commitment = lower AL.
 
    Returns
    -------
    delays    : list[int]  — one entry per token position in final hypothesis
    final_hyp : list[int]  — token ids of the last full hypothesis
    """
    actual_len = len(frame_tensors)
    committed  = {}   # token_position → frame_index of first appearance
    final_hyp  = []
 
    # Accumulate frames causally
    seen_frames = []
    seen_kps    = []
    
    prev_hyp = []
 
    for frame_idx in range(actual_len):
        seen_frames.append(frame_tensors[frame_idx])
        seen_kps.append(kp_array[frame_idx])               # (105, 2)
 
        # Trigger every window_size frames (and always on the very last frame)
        is_trigger     = (len(seen_frames) % window_size == 0)
        is_last_frame  = (frame_idx == actual_len - 1)
 
        if not (is_trigger or is_last_frame):
            continue
 
        # "Time" = 1-based index of the last frame seen so far
        current_frame = frame_idx + 1
 
        # Build tensor from the full causal prefix seen so far
        video_t, kp_t = _frames_to_tensor(seen_frames, seen_kps)
 
        with torch.no_grad():
            if ctc_aux_head is not None:
                hyp_ids = ctc_attention_decode(
                    model, video_t, kp_t, vocab, ctc_aux_head,
                    max_decode_len=MAX_SENTENCE_LEN,
                )[0]
            else:
                hyp_ids = greedy_decode(
                    model, video_t, kp_t, vocab,
                    max_decode_len=MAX_SENTENCE_LEN,
                )[0]
 
        final_hyp = hyp_ids
        
        # Find longest common prefix with previous hypothesis
        common_prefix_len = 0
        for i in range(min(len(prev_hyp), len(hyp_ids))):
            if prev_hyp[i] == hyp_ids[i]:
                common_prefix_len += 1
            else:
                break

        # Only commit tokens that are stable
        for pos in range(common_prefix_len):
            if pos not in committed:
                committed[pos] = current_frame

        prev_hyp = hyp_ids
 
    # Build delay list ordered by token position
    delays = [committed[pos] for pos in sorted(committed.keys())]
    return delays, final_hyp
 
 
# ─────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────
 
def evaluate_online_al(
    checkpoints_dir: str,
    root_path:       str,
    # num_samples:     int  = 50,
    window_size:     int  = 16,
    fps:             int  = 10,
    seed:            int  = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
 
    model, vocab, vocab_inv, ctc_aux_head = load_model_and_vocab(checkpoints_dir)
 
    pad_idx   = vocab.get("<pad>",   0)
    blank_idx = vocab.get("<blank>", 1)
    bos_idx   = vocab.get("<bos>",   2)
    eos_idx   = vocab.get("<eos>",   3)
    special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}
 
    keypoints_dir = os.path.join(
        '/home/abdullahm/jaleel/CV_project/Project/keypoints', "test"
    )
 
    # ── dataset (vocab-consistent, no augmentation) ────────────────────────
    print(f"\nLoading test dataset from: {root_path}")
    val_dataset = Phoenix14TDataset(
        root_dir=root_path, split="dev", vocab=vocab, is_training=False
    )
    if len(val_dataset) == 0:
        raise RuntimeError("Test dataset is empty — check root_path.")
 
    indices = range(len(val_dataset))
    print(f"Evaluating {len(indices)} samples | "
          f"window={window_size} frames | simulated {fps} FPS\n")
    print("=" * 65)
 
    al_scores_frames  = []   # AL in frames
    al_scores_seconds = []   # AL in seconds  (frames / fps)
 
    for rank, idx in enumerate(tqdm(indices, desc="Online inference"), 1):
        sample      = val_dataset.samples[idx]
        frame_paths = sample["paths"][:MAX_FRAMES]
        actual_len  = len(frame_paths)
        video_name  = sample["video"]
 
        # ── load data ──────────────────────────────────────────────────────
        frame_tensors = load_frames(frame_paths, MAX_FRAMES)
        kp_path  = os.path.join(keypoints_dir, f"{video_name}.npy")
        kp_array = load_keypoints(kp_path, actual_len)     # (actual_len, 105, 2)
 
        # ── reference ─────────────────────────────────────────────────────
        _, _, target_tensor, tgt_len_with_special = val_dataset[idx]
        # tgt_len_with_special = actual_len (video) returned by __getitem__
        # target_tensor = [BOS, g1, ..., gN, EOS]
        ref_ids = target_tensor[1:-1].tolist()             # strip BOS + EOS
        ref_str = _ids_to_str(ref_ids, vocab_inv, special_ids)
 
        if not ref_ids:
            continue   # skip empty references
 
        # ── online streaming inference ─────────────────────────────────────
        delays, final_hyp = online_infer_video(
            model, vocab, vocab_inv, ctc_aux_head, special_ids,
            frame_tensors, kp_array,
            window_size=window_size,
            bos_idx=bos_idx,
        )
 
        pred_str = _ids_to_str(final_hyp, vocab_inv, special_ids)
 
        if not delays:
            # Model produced nothing — treat as maximum latency
            delays = [actual_len]
 
        # ── AL computation ─────────────────────────────────────────────────
        al_f = compute_AL(delays, actual_len)
        al_s = al_f / fps
 
        al_scores_frames.append(al_f)
        al_scores_seconds.append(al_s)
 
        tqdm.write(
            f"[{rank:>3}/{len(indices)}] {video_name}  |  "
            f"frames={actual_len}  AL={al_f:.1f}fr / {al_s:.2f}s\n"
            f"  GT:   {ref_str}\n"
            f"  PRED: {pred_str}"
        )
 
    # ── Aggregate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ONLINE INFERENCE — AVERAGE LAGGING RESULTS")
    print("=" * 65)
    if al_scores_frames:
        print(f"  Samples evaluated   : {len(al_scores_frames)}")
        print(f"  Window size         : {window_size} frames")
        print(f"  Simulated FPS       : {fps}")
        print(f"  Mean AL (frames)    : {mean(al_scores_frames):.2f}")
        print(f"  Mean AL (seconds)   : {mean(al_scores_seconds):.3f} s")
        print(f"  Min  AL (seconds)   : {min(al_scores_seconds):.3f} s")
        print(f"  Max  AL (seconds)   : {max(al_scores_seconds):.3f} s")
    else:
        print("  No valid samples were evaluated.")
    print("=" * 65)
 
 
# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online-style SLT inference with Average Lagging evaluation"
    )
    parser.add_argument("--checkpoints", default="checkpoints_orig_run2",
                        help="Dir with vocab.json and model checkpoints")
    parser.add_argument(
        "--data_root",
        default=("/home/abdullahm/jaleel/CV_project/CLIP-SLA/Data/"
                 "PHOENIX-2014-T-release-v3/PHOENIX-2014-T"),
    )
    # parser.add_argument("--num_samples",  type=int, default=50,
    #                     help="Number of test videos to evaluate")
    parser.add_argument("--window_size",  type=int, default=16,
                        help="Decode trigger interval (frames)")
    parser.add_argument("--fps",          type=int, default=10,
                        help="Simulated frame rate for AL-in-seconds conversion")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
 
    evaluate_online_al(
        checkpoints_dir=args.checkpoints,
        root_path=args.data_root,
        # num_samples=args.num_samples,
        window_size=args.window_size,
        fps=args.fps,
        seed=args.seed,
    )