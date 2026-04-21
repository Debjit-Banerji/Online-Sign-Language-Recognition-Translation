"""
Model loader — abstracts SLRT model loading and inference.

Supports two modes:
  - "mock"  → MockModel with placeholder outputs (no GPU needed)
  - "slrt"  → Real TwoStreamNetwork S2T pipeline (requires CUDA)

The interface is the same regardless of mode:
    model.predict(frames) -> {"glosses": str, "text": str}
"""

import time
import random
import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

# ────────────────────────────────────────────────────────────────
# Mock model for development / testing without GPU
# ────────────────────────────────────────────────────────────────

# Sample glosses and translations for realistic mock output
MOCK_GLOSSES = [
    "HELLO", "THANK-YOU", "PLEASE", "YES", "NO", "GOOD", "BAD",
    "NAME", "MY", "YOUR", "WHAT", "WHERE", "WHEN", "HOW",
    "MORNING", "EVENING", "FRIEND", "HELP", "SORRY", "WELCOME",
    "EAT", "DRINK", "WANT", "NEED", "LIKE", "UNDERSTAND",
]

MOCK_TRANSLATIONS = [
    "Hello, how are you?",
    "Thank you very much.",
    "Good morning!",
    "What is your name?",
    "I need help please.",
    "Nice to meet you.",
    "Where are you going?",
    "I understand, thank you.",
    "Please help me.",
    "Good evening, welcome.",
    "I want to eat something.",
    "Do you understand?",
]


class MockModel:
    """
    Simulates model inference with realistic-looking outputs.
    Use this for developing/testing the app without loading real models.
    """

    def __init__(self):
        self._call_count = 0

    def predict(self, frames: np.ndarray) -> Dict[str, str]:
        """
        Mock inference on a batch of frames.
        
        Args:
            frames: np.ndarray of shape (1, T, C, H, W) — preprocessed video
        
        Returns:
            dict with "glosses" and "text" keys
        """
        self._call_count += 1
        
        # Simulate inference latency (50-150ms)
        time.sleep(random.uniform(0.05, 0.15))
        
        # Generate mock output
        num_glosses = random.randint(2, 5)
        glosses = " ".join(random.choices(MOCK_GLOSSES, k=num_glosses))
        text = random.choice(MOCK_TRANSLATIONS)
        
        return {
            "glosses": glosses,
            "text": text,
        }


# ────────────────────────────────────────────────────────────────
# Real SLRT model
# ────────────────────────────────────────────────────────────────

# Fixed paths — these are stable in this repo layout
SLRT_ROOT = "/home/jaleel_225/CV_project/SLRT"
TWOSTREAM_DIR = os.path.join(SLRT_ROOT, "TwoStreamNetwork")
DEFAULT_CONFIG = os.path.join(
    TWOSTREAM_DIR,
    "experiments/configs/TwoStream/phoenix-2014t_s2t_video.yaml",
)
DEFAULT_CKPT = os.path.join(
    SLRT_ROOT,
    "slt_checkpoints/video/ckpts/best.ckpt",
)
S3D_CKPT_DIR = os.path.join(
    SLRT_ROOT,
    "pretrained_models/s3ds_actioncls_ckpt",
)


class SLRTModel:
    """
    Real SLRT inference pipeline (TwoStreamNetwork, Phoenix-2014T).

    Architecture (composite model):
        Stage 1 — S3D backbone:
            Raw video (B, C, T, 224, 224) → spatiotemporal features (B, T', 832)
            Loaded separately from the S3D action-classification checkpoint.

        Stage 2 — S2T head (visual head + VLMapper + mBART):
            Features → gloss logits (CTC decode) → German glosses
            Features → VLMapper → mBART → German text

        Stage 3 — Translation:
            German text → English text  (deep_translator / GoogleTranslator)
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        checkpoint_path: str = DEFAULT_CKPT,
        device: str = "cuda",
    ):
        self.device_str = device

        # ── 1. Bootstrap SLRT imports ────────────────────────────
        # The SLRT codebase uses relative paths in YAML configs
        # (e.g. "data/phoenix-2014t/...") that resolve from SLRT_ROOT.
        if TWOSTREAM_DIR not in sys.path:
            sys.path.insert(0, TWOSTREAM_DIR)
        os.chdir(SLRT_ROOT)  # so relative config paths resolve

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings; warnings.filterwarnings("ignore")

        import torch
        from utils.misc import load_config, make_logger, neq_load_customized
        from modelling.model import build_model
        from modelling.S3D import S3D_backbone

        self.torch = torch
        self._device = torch.device(device)

        # Logger (required by SLRT internals)
        log_dir = os.path.join(TWOSTREAM_DIR, "results", "server_log")
        os.makedirs(log_dir, exist_ok=True)
        make_logger(model_dir=log_dir, log_file="slrt_server.log")

        # ── 2. Build & load S2T model (visual head + mBART) ─────
        print("  [SLRT] Loading S2T config...")
        cfg = load_config(config_path)
        cfg["device"] = self._device

        print("  [SLRT] Building S2T model (visual head + VLMapper + mBART)...")
        self.model = build_model(cfg)

        print(f"  [SLRT] Loading S2T checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        neq_load_customized(self.model, state_dict["model_state"], verbose=False)
        self.model.eval()

        # ── 3. Build & load S3D backbone for raw video ──────────
        print(f"  [SLRT] Loading S3D backbone from: {S3D_CKPT_DIR}")
        self.backbone = S3D_backbone(
            in_channel=3,
            use_block=4,
            freeze_block=1,
            pretrained_ckpt=S3D_CKPT_DIR,
            cfg_pyramid=None,  # no pyramid for standalone feature extraction
        ).to(self._device)
        self.backbone.eval()

        # ── 4. Testing config from YAML ─────────────────────────
        self.generate_cfg = cfg.get("testing", {}).get("cfg", {}).get(
            "translation", {"num_beams": 5, "max_length": 100, "length_penalty": 1}
        )

        # ── 5. Translator (German → English) ────────────────────
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source="de", target="en")
            print("  [SLRT] German→English translator ready")
        except Exception as e:
            print(f"  [SLRT] ⚠ Translator init failed ({e}), will return German text")
            self.translator = None

        print("  [SLRT] ✅ Model ready!")

    # ────────────────────────────────────────────────────────────
    def predict(self, frames: np.ndarray) -> Dict[str, str]:
        """
        Run full inference on preprocessed frames.

        Args:
            frames: np.ndarray shape (1, T, C, H, W), float32, normalized [-1,1]
                    exactly as produced by preprocessing.preprocess_numpy_frames()

        Returns:
            {"glosses": "HELLO WORLD", "text": "Hello world"}
        """
        torch = self.torch

        with torch.no_grad():
            # ── A. Prepare video tensor for S3D ──────────────────
            # preprocessing gives (B, T, C, H, W); S3D expects (B, C, T, H, W)
            video = torch.from_numpy(frames).to(self._device)  # (1, T, C, H, W)
            video = video.permute(0, 2, 1, 3, 4).float()       # (1, C, T, H, W)

            B, C, T_in, H, W = video.shape
            sgn_lengths = torch.tensor([T_in], device=self._device)

            # S3D expects BGR channel order and [-1,1] normalization.
            # Our preprocessing already normalizes to [-1,1].
            # The SLRT recognition code does: sgn_videos[:,:,[2,1,0],:,:]
            # (RGB→BGR swap). We replicate that here.
            video = video[:, [2, 1, 0], :, :, :]  # RGB → BGR

            # ── B. S3D backbone → spatiotemporal features ────────
            backbone_out = self.backbone(sgn_videos=video, sgn_lengths=sgn_lengths)
            # backbone_out['sgn'] shape: (B, T', 832) — pooled 3D features
            sgn_features = backbone_out["sgn"]  # (1, T', 832)
            T_out = sgn_features.shape[1]

            # Build mask and lengths for recognition network
            sgn_mask = torch.ones(B, 1, T_out, dtype=torch.bool, device=self._device)
            sgn_len_out = torch.tensor([T_out], device=self._device)

            # Dummy gloss labels (not used in inference, but required by forward sig)
            gloss_labels = torch.zeros(B, 1, dtype=torch.long, device=self._device)
            gls_lengths = torch.tensor([1], device=self._device)

            # ── C. Recognition network → gloss logits ────────────
            # The S2T model was trained with TWO streams (rgb + keypoint).
            # Since we only have RGB, we pass the same features for both
            # streams. This is not ideal but produces reasonable results.
            recog_out = self.model.recognition_network(
                is_train=False,
                sgn_features=sgn_features,         # unused for feature mode
                sgn_mask=sgn_mask,
                sgn_lengths=sgn_len_out,
                gloss_labels=gloss_labels,
                gls_lengths=gls_lengths,
                head_rgb_input=sgn_features,        # RGB stream features
                head_keypoint_input=sgn_features,   # duplicate for keypoint stream
            )

            # ── D. CTC decode → German glosses ───────────────────
            glosses_str = ""
            # Try different logit keys that might be present
            for key in ["ensemble_last_gloss_logits", "gloss_logits", "rgb_gloss_logits"]:
                if key in recog_out and recog_out[key] is not None:
                    decoded = self.model.predict_gloss_from_logits(
                        gloss_logits=recog_out[key],
                        beam_size=1,
                        input_lengths=recog_out["input_lengths"],
                    )
                    glosses_tokens = self.model.gloss_tokenizer.convert_ids_to_tokens(decoded)
                    if glosses_tokens and glosses_tokens[0]:
                        glosses_str = " ".join(glosses_tokens[0]).upper()
                    break

            # ── E. VLMapper + mBART → German text ────────────────
            mapped_feature = self.model.vl_mapper(visual_outputs=recog_out)
            # Prepare transformer inputs manually (like translation_network.forward does)
            transformer_kwargs = self.model.translation_network.prepare_feature_inputs(
                mapped_feature, recog_out["input_lengths"]
            )
            gen_out = self.model.translation_network.generate(
                **transformer_kwargs,
                **self.generate_cfg,
            )
            german_text = gen_out["decoded_sequences"][0] if gen_out["decoded_sequences"] else ""

            # ── F. German → English ──────────────────────────────
            english_text = german_text
            if self.translator and german_text.strip():
                try:
                    english_text = self.translator.translate(german_text)
                except Exception as e:
                    print(f"  [SLRT] Translation failed: {e}")
                    english_text = german_text  # fallback to German

        return {
            "glosses": glosses_str,
            "text": english_text,
            "text_de": german_text,  # bonus: also return original German
        }


# ────────────────────────────────────────────────────────────────
# Custom CLIP+GCN SLT model (user's own trained model)
# ────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Project"
)
DEFAULT_CUSTOM_CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints_orig_run2")


# ── Standalone decode functions (copied from train code to avoid
#    importing the heavy training script with dataset dependencies) ──

_PAD = 0
_BLANK = 1
_BOS = 2
_EOS = 3


def _ids_to_str(ids: list, vocab_inv: dict, special_ids: set) -> str:
    """Convert token-id list to gloss string, skipping special tokens."""
    return " ".join(
        vocab_inv[i] for i in ids
        if i not in special_ids and i in vocab_inv
    )


def _greedy_decode_custom(model, frames, keypoints, vocab, max_decode_len=60):
    """Greedy auto-regressive decode for CustomSLTModel."""
    import torch

    device = frames.device
    bos_idx = vocab.get("<bos>", _BOS)
    eos_idx = vocab.get("<eos>", _EOS)
    pad_idx = vocab.get("<pad>", _PAD)

    out = model(frames, keypoints, tgt_tokens=None)
    memory = out["context_seq"]  # (B, T', D)
    B = memory.size(0)

    generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_decode_len):
        logits = model.translator(memory, generated)  # (B, step, V)
        next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_idx),
            next_token,
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_idx)
        if finished.all():
            break

    decoded = []
    for b in range(B):
        seq = generated[b, 1:].tolist()  # drop BOS
        if eos_idx in seq:
            seq = seq[: seq.index(eos_idx)]
        while seq and seq[-1] == pad_idx:
            seq.pop()
        decoded.append(seq)
    return decoded


def _ctc_attention_decode_custom(
    model, frames, keypoints, vocab, ctc_aux_head,
    ctc_weight=0.3, max_decode_len=60,
):
    """CTC-Attention hybrid decode for CustomSLTModel."""
    import torch

    device = frames.device
    bos_idx = vocab.get("<bos>", _BOS)
    eos_idx = vocab.get("<eos>", _EOS)
    pad_idx = vocab.get("<pad>", _PAD)
    blank_idx = vocab.get("<blank>", _BLANK)

    out = model(frames, keypoints, tgt_tokens=None)
    memory = out["context_seq"]  # (B, T', D)
    B = memory.size(0)

    # CTC greedy on encoder output
    ctc_logits = ctc_aux_head(memory)  # (B, T', V)
    ctc_lprobs = ctc_logits.log_softmax(-1)
    ctc_pred = ctc_lprobs.argmax(-1)  # (B, T')

    ctc_seqs = []
    for b in range(B):
        prev = -1
        seq = []
        for tok in ctc_pred[b].tolist():
            if tok != prev and tok != blank_idx:
                seq.append(tok)
            prev = tok
        ctc_seqs.append(seq)

    # Attention rescore each CTC hypothesis
    decoded = []
    for b in range(B):
        ctc_seq = ctc_seqs[b]
        if not ctc_seq:
            # Fallback to AR greedy
            gen = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
            mem_b = memory[b:b+1]
            for _ in range(max_decode_len):
                logits = model.translator(mem_b, gen)
                nt = logits[:, -1, :].argmax(dim=-1)
                if nt.item() == eos_idx:
                    break
                gen = torch.cat([gen, nt.unsqueeze(1)], dim=1)
            decoded.append(gen[0, 1:].tolist())
            continue

        hyp_ids = torch.tensor(
            [bos_idx] + ctc_seq, dtype=torch.long, device=device
        ).unsqueeze(0)
        mem_b = memory[b:b+1]

        att_logits = model.translator(mem_b, hyp_ids)
        att_lprobs = att_logits.log_softmax(-1)

        target_ids = torch.tensor(
            ctc_seq + [eos_idx], dtype=torch.long, device=device
        )
        L = min(att_lprobs.size(1), len(target_ids))
        att_score = att_lprobs[0, :L, :].gather(
            1, target_ids[:L].unsqueeze(1)
        ).mean().item()

        ctc_score = ctc_lprobs[b].max(-1).values.mean().item()
        combined = (1 - ctc_weight) * att_score + ctc_weight * ctc_score

        if combined < -8.0:
            # Translator strongly disagrees — fallback to AR greedy
            gen = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
            for _ in range(max_decode_len):
                logits = model.translator(mem_b, gen)
                nt = logits[:, -1, :].argmax(dim=-1)
                if nt.item() == eos_idx:
                    break
                gen = torch.cat([gen, nt.unsqueeze(1)], dim=1)
            decoded.append(gen[0, 1:].tolist())
        else:
            decoded.append(ctc_seq)

    return decoded


class CustomSLTModel:
    """
    User's own CLIP+GCN Sign Language Translation model.

    Architecture:
        - CLIP ViT-B/32 (frozen) + SLA adapters → RGB features
        - MediaPipe Holistic → 105 keypoints → GCN PoseEncoder
        - Dynamic gating fusion → TCN → Temporal Transformer
        - Greedy / CTC-Attention hybrid decoder → German glosses
        - deep-translator → English text
    """

    def __init__(self, checkpoints_dir: str, device: str = "cuda"):
        import torch
        import torch.nn as nn
        import json

        self._device = torch.device(device)

        # ── 1. Load config & vocab ──
        config_path = os.path.join(checkpoints_dir, "model_config.json")
        with open(config_path, "r") as f:
            self._config = json.load(f)

        vocab_path = os.path.join(checkpoints_dir, "vocab.json")
        with open(vocab_path, "r") as f:
            self._vocab = json.load(f)
        self._vocab_inv = {int(v): k for k, v in self._vocab.items()}

        pad_idx = self._vocab.get("<pad>", 0)
        bos_idx = self._vocab.get("<bos>", 2)
        eos_idx = self._vocab.get("<eos>", 3)
        blank_idx = self._vocab.get("<blank>", 1)
        self._special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}

        # ── 2. Add Project to path and import model classes ──
        project_dir = os.path.abspath(PROJECT_DIR)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        print(f"  [Custom] Project dir: {project_dir}")

        import clip as clip_module
        from Models import SLTModel as _SLTModel, convert_to_deploy

        # ── 3. Load CLIP backbone ──
        print(f"  [Custom] Loading CLIP {self._config['clip_model_name']}...")
        clip_model, _ = clip_module.load(
            self._config["clip_model_name"], device=self._device
        )

        # ── 4. Build SLTModel ──
        print("  [Custom] Building SLTModel...")
        self._model = _SLTModel(
            clip_model=clip_model,
            vocab_size=self._config["vocab_size"],
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            adapter_dim=self._config["adapter_dim"],
            adapter_layers=range(*self._config["adapter_layers"]),
            clip_frames=self._config["clip_frames"],
            clip_spatial=self._config["clip_spatial"],
            pose_video_len=16,  # must match window_size to prevent CUDA OOB
        )

        # ── 5. Load weights ──
        model_path = os.path.join(checkpoints_dir, "phase2_best_wer.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoints_dir, "phase1_best_loss.pth")
        print(f"  [Custom] Loading weights: {model_path}")
        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint.get("model", checkpoint)
        self._model.load_state_dict(state_dict)
        try:
            convert_to_deploy(self._model)
        except Exception:
            pass  # phase2 weights are incompatible with reparameterisation
        self._model.to(self._device)
        self._model.eval()

        # ── 6. Load CTC auxiliary head (optional) ──
        self._ctc_aux_head = None
        ctc_path = os.path.join(checkpoints_dir, "ctc_aux_head_best_wer.pth")
        if os.path.exists(ctc_path):
            print("  [Custom] Loading CTC auxiliary head...")
            fused_dim = self._model.translator.d_model
            self._ctc_aux_head = nn.Linear(
                fused_dim, self._config["vocab_size"]
            ).to(self._device)
            ctc_state = torch.load(ctc_path, map_location=self._device)
            if isinstance(ctc_state, dict) and "model" in ctc_state:
                ctc_state = ctc_state["model"]
            self._ctc_aux_head.load_state_dict(ctc_state)
            self._ctc_aux_head.eval()

        # ── 7. MediaPipe keypoint extractor ──
        print("  [Custom] Initializing MediaPipe keypoint extractor...")
        from Utils.pose_extractor import PoseKeypointExtractor
        from Utils.keypoint_utils import sequence_to_gcn_tensor
        self._pose_extractor = PoseKeypointExtractor()
        self._sequence_to_gcn_tensor = sequence_to_gcn_tensor

        # ── 8. CLIP-compatible preprocessing ──
        from torchvision import transforms
        from PIL import Image
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        self._Image = Image

        # ── 9. German → English translator ──
        from deep_translator import GoogleTranslator
        self._translator = GoogleTranslator(source="de", target="en")
        print("  [Custom] German→English translator ready")

        print("  [Custom] ✅ Model ready!")

    def predict(self, raw_frames: list) -> Dict[str, str]:
        """
        Run inference on raw BGR numpy frames.

        Args:
            raw_frames: list of numpy arrays (H, W, 3), BGR format

        Returns:
            {"glosses": str, "text": str, "text_de": str}
        """
        import torch
        import cv2

        # ── 1. Preprocess frames for CLIP ──
        processed = []
        for f in raw_frames:
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = self._preprocess(self._Image.fromarray(img))
            processed.append(img)
        rgb_tensor = torch.stack(processed).unsqueeze(0).to(self._device)
        # Shape: (1, T, 3, 224, 224)

        # ── 2. Extract keypoints via MediaPipe ──
        kp_list = []
        for f in raw_frames:
            kp = self._pose_extractor.extract_keypoints(f)
            kp_list.append(kp)
        key_tensor = self._sequence_to_gcn_tensor(kp_list).to(self._device)
        # Shape: (1, 2, T, 105)

        # ── 3. Decode ──
        with torch.no_grad():
            if self._ctc_aux_head is not None:
                decoded_ids = _ctc_attention_decode_custom(
                    self._model, rgb_tensor, key_tensor,
                    self._vocab, self._ctc_aux_head,
                )
            else:
                decoded_ids = _greedy_decode_custom(
                    self._model, rgb_tensor, key_tensor, self._vocab,
                )

        # ── 4. Convert IDs → gloss string ──
        glosses = _ids_to_str(decoded_ids[0], self._vocab_inv, self._special_ids)
        if not glosses.strip():
            glosses = "<no sign detected>"

        # ── 5. Translate to English ──
        try:
            # Lowercase for better machine translation (SUED -> sued -> south)
            translation_input = glosses.lower()
            
            # Manual mapping for critical demo terms that often fail in MT
            manual_map = {
                "sued": "South",
                "nord": "North",
                "ost": "East",
                "west": "West",
                "sonne": "Sun",
                "regen": "Rain",
                "wolke": "Cloud",
                "wind": "Wind",
            }
            
            # Use manual map if it's a single word, otherwise use translator
            if translation_input in manual_map:
                english_text = manual_map[translation_input]
            else:
                english_text = self._translator.translate(translation_input)
                
        except Exception:
            english_text = glosses  # fallback: raw glosses

        return {
            "glosses": glosses,
            "text": english_text,
            "text_de": glosses,  # glosses are already German
        }


# ────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────

def load_model(
    mode: str = "mock",
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load the inference model.
    
    Args:
        mode: "mock" for testing, "slrt" for SLRT model, "custom" for user's model
        config_path: Path to config (optional, has defaults)
        checkpoint_path: Path to checkpoint (optional, has defaults)
        device: "cuda" or "cpu"
    
    Returns:
        Model instance with .predict() method
    """
    if mode == "mock":
        print("📦 Loading mock model (no GPU required)")
        return MockModel()
    elif mode == "slrt":
        cfg = config_path or DEFAULT_CONFIG
        ckpt = checkpoint_path or DEFAULT_CKPT
        print(f"📦 Loading SLRT model...")
        print(f"   Config: {cfg}")
        print(f"   Checkpoint: {ckpt}")
        return SLRTModel(cfg, ckpt, device)
    elif mode == "custom":
        ckpt_dir = checkpoint_path or DEFAULT_CUSTOM_CKPT_DIR
        print(f"📦 Loading Custom SLT model...")
        print(f"   Checkpoints: {ckpt_dir}")
        return CustomSLTModel(ckpt_dir, device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mock', 'slrt', or 'custom'.")

