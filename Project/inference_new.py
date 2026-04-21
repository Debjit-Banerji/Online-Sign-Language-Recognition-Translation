import cv2
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import json
import os
from torchvision import transforms

from Models import SLTModel
from Utils.pose_extractor import PoseKeypointExtractor
from Utils.keypoint_utils import sequence_to_gcn_tensor
from train_phases import greedy_decode, ctc_attention_decode, _ids_to_str

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MAX_SENTENCE_LEN = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16

# CLIP normalisation — must match training dataset constants
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def load_model_and_vocab(checkpoints_dir: str = "checkpoints_orig_run2"):
    """
    Loads model weights and vocabulary from the training checkpoint directory.

    No convert_to_deploy is called here — the reparameterisation bakes weights
    permanently into a different architecture and is incompatible with the saved
    state-dicts when done post-training.

    The model hyper-parameters (adapter_dim, clip_frames, etc.) are the same
    values used in run_demo() in train_phases.py.  If you change them there,
    mirror the change here.
    """
    # ── vocab ──────────────────────────────────────────────────────────────
    vocab_path = os.path.join(checkpoints_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. Run training first."
        )
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_inv = {int(v): k for k, v in vocab.items()}
    vocab_size = len(vocab)
    print(f"Loaded vocab: {vocab_size} tokens.")

    # ── CLIP backbone ──────────────────────────────────────────────────────
    clip_model, _ = clip.load("ViT-B/16", device=DEVICE)

    # ── SLTModel — mirror the exact constructor call from run_demo() ───────
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
        # pre_trained_rgb_encoder_weights not needed at inference
    )

    # ── weights ────────────────────────────────────────────────────────────
    for candidate in ["phase3_best_wer.pth", "phase2_best_wer.pth",
                      "phase3_latest.pth",   "phase2_latest.pth"]:
        model_path = os.path.join(checkpoints_dir, candidate)
        if os.path.exists(model_path):
            print(f"Loading model weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict)
            break
    else:
        raise FileNotFoundError(
            f"No trained weights found in {checkpoints_dir}. "
            "Expected phase3_best_wer.pth / phase2_best_wer.pth etc."
        )

    model.to(DEVICE)
    model.eval()
    print("Model ready (no deploy-mode reparameterisation applied).")

    # ── CTC auxiliary head (optional) ──────────────────────────────────────
    ctc_aux_head = None
    for ctc_candidate in ["ctc_aux_head_phase3_best.pth",
                           "ctc_aux_head_best_wer.pth",
                           "ctc_aux_head_phase3_latest.pth"]:
        ctc_path = os.path.join(checkpoints_dir, ctc_candidate)
        if os.path.exists(ctc_path):
            print(f"Loading CTC auxiliary head from: {ctc_path}")
            fused_dim    = model.translator.d_model
            ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(DEVICE)
            ctc_aux_head.load_state_dict(
                torch.load(ctc_path, map_location=DEVICE)
            )
            ctc_aux_head.eval()
            break

    if ctc_aux_head is None:
        print("No CTC auxiliary head found — will use pure greedy decoding.")

    return model, vocab, vocab_inv, ctc_aux_head


# ─────────────────────────────────────────────────────────────────
# Frame preprocessing  (matches training pipeline exactly)
# ─────────────────────────────────────────────────────────────────

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def preprocess_frames(frames: list) -> torch.Tensor:
    """
    Convert a list of BGR numpy frames (H, W, 3) to a batched tensor
    (1, T, 3, 224, 224) ready for the model.
    """
    processed = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = _preprocess(Image.fromarray(img))
        processed.append(img)
    # (T, 3, 224, 224) -> (1, T, 3, 224, 224)
    return torch.stack(processed).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────

def run_inference(checkpoints_dir: str = "checkpoints_orig_run2"):
    model, vocab, vocab_inv, ctc_aux_head = load_model_and_vocab(checkpoints_dir)

    # Derived special-token ids
    pad_idx   = vocab.get("<pad>",   0)
    blank_idx = vocab.get("<blank>", 1)
    bos_idx   = vocab.get("<bos>",   2)
    eos_idx   = vocab.get("<eos>",   3)
    special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}

    pose_extractor = PoseKeypointExtractor()
    cap = cv2.VideoCapture(0)

    frame_buffer: list = []
    kp_buffer:    list = []

    print("Starting webcam inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Show live feed
        cv2.imshow("Sign Language Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # ── buffer frames & keypoints ──────────────────────────────────────
        try:
            kp = pose_extractor.extract_keypoints(frame)  # (105, 2)
        except Exception:
            kp = np.zeros((105, 2), dtype=np.float32)

        frame_buffer.append(frame)
        kp_buffer.append(kp)

        # ── process when window is full ────────────────────────────────────
        if len(frame_buffer) < WINDOW_SIZE:
            continue

        # -- RGB tensor --
        rgb_tensor = preprocess_frames(frame_buffer).to(DEVICE)  # (1, T, 3, 224, 224)

        # -- Keypoint tensor --
        # kp_buffer: list of (105, 2) arrays → stack → (T, 105, 2)
        # then permute to model's expected (1, 2, T, 105)
        kp_np    = np.stack(kp_buffer, axis=0)                   # (T, 105, 2)
        kp_torch = torch.from_numpy(kp_np).permute(2, 0, 1)      # (2, T, 105)
        kp_torch = kp_torch.unsqueeze(0).float().to(DEVICE)       # (1, 2, T, 105)

        with torch.no_grad():
            # Quick forward to get boundary / CTL signals
            tgt_seed = torch.tensor([[bos_idx]], dtype=torch.long, device=DEVICE)
            out = model(rgb_tensor, kp_torch, tgt_tokens=tgt_seed)

            # Boundary head: skip if model thinks this window has no sign
            is_sign = True
            if "boundary_probs" in out:
                is_sign = out["boundary_probs"].item() > 0.5

            if is_sign:
                # CTL head: refine max decode length
                max_len = MAX_SENTENCE_LEN
                if "ctl_logits" in out:
                    predicted_count = out["ctl_logits"].argmax(dim=-1).item() + 1
                    max_len = predicted_count + 5  # small buffer

                if ctc_aux_head is not None:
                    decoded_ids = ctc_attention_decode(
                        model, rgb_tensor, kp_torch, vocab,
                        ctc_aux_head, max_decode_len=max_len,
                    )
                else:
                    decoded_ids = greedy_decode(
                        model, rgb_tensor, kp_torch, vocab,
                        max_decode_len=max_len,
                    )

                translation = _ids_to_str(decoded_ids[0], vocab_inv, special_ids)
                print(f"[Translated] {translation}")
            else:
                print("[No sign detected in this window]")

        # Reset buffers for next window
        frame_buffer.clear()
        kp_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SLT real-time inference")
    parser.add_argument(
        "--checkpoints", default="checkpoints_orig_run2",
        help="Directory containing vocab.json and model checkpoint files",
    )
    args = parser.parse_args()
    run_inference(checkpoints_dir=args.checkpoints)