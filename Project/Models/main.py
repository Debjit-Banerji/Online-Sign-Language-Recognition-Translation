# models.py  — fixed
# Changes vs original:
#  [M1] CTLHead: store max_len as instance attribute so trainer can read it.
#  [M2] SmallTranslator: expose bos_idx / eos_idx as stored attributes for the greedy decoder.
#  [M3] SLTModel.__init__: accept pad_idx and forward it to SmallTranslator.
#  [M4] SLTModel.forward: always return context_seq; logits only when tgt_tokens given.
#       (No behaviour change — but makes it explicit and greedy_decode-compatible.)
#  [M5] SmallTranslator.forward: guard against empty target sequences.

import math
import torch
import torch.nn as nn
from torch.nn import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
)
from .clip import CLIP_SLA_Wrapper, TemporalConvNetwork
from .keypoint_pipeline import PoseEncoder
from .ctl import CTLHead, BoundaryHead


# ─────────────────────────────────────────────
#  SmallTemporalTransformer (unchanged except doc)
# ─────────────────────────────────────────────

class SmallTemporalTransformer(nn.Module):
    def __init__(self, dim, n_layers=2, n_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim if ff_dim is not None else dim * 2
        enc_layer = TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dim = dim

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D) → (T, B, D) → (B, T, D)
        out = self.encoder(x.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        return out.permute(1, 0, 2).contiguous()


# ─────────────────────────────────────────────
#  DynamicFusion (unchanged)
# ─────────────────────────────────────────────

class DynamicFusion(nn.Module):
    def __init__(self, dim_rgb, dim_pose, fused_dim, gate_hidden=128):
        super().__init__()
        self.dim_rgb   = dim_rgb
        self.dim_pose  = dim_pose
        self.fused_dim = fused_dim

        self.proj_rgb  = nn.Linear(dim_rgb,  fused_dim)
        self.proj_pose = nn.Linear(dim_pose, fused_dim)

        self.gate_conv = nn.Sequential(
            nn.Conv1d(fused_dim * 2, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(gate_hidden,  2,            kernel_size=1),
        )

        trans_layer = TransformerEncoderLayer(
            d_model=fused_dim, nhead=4, dim_feedforward=fused_dim * 2, activation="gelu"
        )
        self.fuse_transformer = TransformerEncoder(trans_layer, num_layers=1)

    def forward(self, rgb_seq, pose_seq, prev_fused_seq=None):
        rgb_proj  = self.proj_rgb(rgb_seq)
        pose_proj = self.proj_pose(pose_seq)

        concat = torch.cat([rgb_proj, pose_proj], dim=-1).permute(0, 2, 1)  # (B, 2D, T)
        gate_w = self.gate_conv(concat).permute(0, 2, 1).softmax(dim=-1)    # (B, T, 2)

        fused_seq = gate_w[:, :, 0:1] * rgb_proj + gate_w[:, :, 1:2] * pose_proj

        fused_seq = self.fuse_transformer(
            fused_seq.permute(1, 0, 2)
        ).permute(1, 0, 2).contiguous()
        return fused_seq


# ─────────────────────────────────────────────
#  PositionalEncoding (unchanged)
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# ─────────────────────────────────────────────
#  SmallTranslator
# ─────────────────────────────────────────────

class SmallTranslator(nn.Module):
    def __init__(
        self, d_model, vocab_size,
        pad_idx=0, bos_idx=None, eos_idx=None,
        n_layers=2, n_heads=4, dim_feedforward=None, max_len=100,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pad_idx  = pad_idx
        # [M2] Store BOS/EOS so greedy_decode can read them without touching the vocab dict
        self.bos_idx  = bos_idx  # may be None if not defined in vocab
        self.eos_idx  = eos_idx

        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb     = PositionalEncoding(d_model, max_len=5000)

        dec_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or d_model * 2,
            dropout=0.2,                 # 🔴 ADD THIS
            activation="gelu",
        )
        self.decoder = TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out     = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, memory, tgt_tokens, memory_key_padding_mask=None):
        """
        memory:     (B, T_src, D)
        tgt_tokens: (B, T_tgt)   — token ids, already shifted (no EOS at end)

        [M5] Guard against empty target — can happen if a sample has only 1 gloss
             and tgt[:, :-1] becomes length-0.
        """
        B, T = tgt_tokens.shape
        if T == 0:
            V = self.out.out_features
            return torch.zeros(B, 0, V, device=tgt_tokens.device)

        device = tgt_tokens.device
        tgt_mask            = self.generate_square_subsequent_mask(T, device)
        tgt_key_padding_mask = (tgt_tokens == self.pad_idx)

        tgt    = self.pos_emb(self.tgt_tok_emb(tgt_tokens))   # (B, T, D)
        out    = self.decoder(
            tgt.permute(1, 0, 2),
            memory.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,  # add this
        )
        return self.out(out.permute(1, 0, 2))                  # (B, T, V)


# ─────────────────────────────────────────────
#  Full SLTModel
# ─────────────────────────────────────────────

class SLTModel(nn.Module):
    def __init__(
        self,
        clip_model,
        adapter_dim=256,
        adapter_layers=range(2, 12),
        clip_frames=4,
        clip_spatial=14,
        pose_in_channel=2,
        pose_channels=(256, 256, 512, 512),
        pose_num_joints=105,
        pose_video_len=85,
        fused_dim=512,
        vocab_size=5000,
        pad_idx=0,                                   # [M3] now accepted
        bos_idx=None,
        eos_idx=None,
        pre_trained_rgb_encoder_weights=None,
    ):
        super().__init__()

        self.rgb_encoder = CLIP_SLA_Wrapper(
            clip_model,
            reduction_dim=adapter_dim,
            adapter_layers=adapter_layers,
            frames=clip_frames,
            spatial_side=clip_spatial,
            freeze_clip=True,
        )
        if pre_trained_rgb_encoder_weights is not None:
            self.rgb_encoder.load_state_dict(pre_trained_rgb_encoder_weights)

        self.pose_encoder = PoseEncoder(
            in_channel=pose_in_channel,
            channels=list(pose_channels),
            num_joints=pose_num_joints,
            video_len=pose_video_len,
        )

        self.rgb_proj  = nn.Linear(self._clip_out_dim(), fused_dim)
        self.pose_proj = nn.Linear(self.pose_encoder.out_dim, fused_dim)

        self.fusion   = DynamicFusion(dim_rgb=fused_dim, dim_pose=fused_dim, fused_dim=fused_dim)
        self.tempconv = TemporalConvNetwork(in_channels=fused_dim, out_channels=fused_dim)
        self.temporal = SmallTemporalTransformer(dim=fused_dim, n_layers=2, n_heads=4)
        self.ctl      = CTLHead(hidden_dim=fused_dim, max_len=8)       # [M1] CTLHead stores max_len
        self.boundary = BoundaryHead(hidden_dim=fused_dim)

        self.translator = SmallTranslator(
            d_model=fused_dim,
            vocab_size=vocab_size,
            pad_idx=pad_idx,           # [M3]
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            n_layers=2, n_heads=4,
        )

    def _clip_out_dim(self):
        try:
            return (
                self.rgb_encoder.clip.visual.output_dim
                if hasattr(self.rgb_encoder.clip.visual, "output_dim")
                else self.rgb_encoder.clip.visual.attn.in_proj_weight.shape[1]
            )
        except Exception:
            return 512

    def forward(self, frames, keypoints, tgt_tokens=None, mode="train"):
        """
        frames:     (B, T, 3, H, W)
        keypoints:  (B, 2, T, J)   — processed keypoint tensor
        tgt_tokens: (B, T_tgt) | None
                    If None the translator is skipped (for greedy decoding,
                    the caller will use self.translator directly with the
                    auto-regressively built sequence).

        Returns dict with keys:
            logits        – (B, T_tgt, V) or None
            ctl_logits    – (B, max_len)
            boundary_probs– (B,)
            context_seq   – (B, T', D)   always returned [M4]
        """
        context_seq = self.encode(frames, keypoints)
        ctl_logits     = self.ctl(context_seq)
        # boundary_probs = torch.sigmoid(self.boundary(context_seq.mean(dim=1)))
        # 7. Translator (optional — skipped when no tgt_tokens)
        logits = None
        if tgt_tokens is not None:
            logits = self.translator(context_seq, tgt_tokens)   # (B, T_tgt, V)
        return {
            "logits":         logits,
            "ctl_logits":     ctl_logits,
            # "boundary_probs": boundary_probs,
            "context_seq":    context_seq,
        }
    
    def encode(self, frames: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Run only the encoder stack and return context_seq (B, T', D).
        Call this once per step, then reuse memory for decode() and sequence_loss().
        Avoids the double-encoder-pass that caused OOM in Phase 3.
        """
        # 1. Encode RGB
        rgb_feats = self.rgb_encoder(frames)        # (B, T, Dr)
        rgb_proj  = self.rgb_proj(rgb_feats)        # (B, T, Df)

        # 2. Encode pose
        pose_feats = self.pose_encoder(keypoints)   # (B, T, Dp)
        pose_proj  = self.pose_proj(pose_feats)     # (B, T, Df)

        # 3. Dynamic fusion
        fused_seq = self.fusion(rgb_proj, pose_proj)  # (B, T, Df)

        # 4. TCN → reduces T by 4×
        tcn_feats   = self.tempconv(fused_seq)        # (B, T//4, Df)

        # 5. Temporal transformer
        context_seq = self.temporal(tcn_feats)        # (B, T//4, Df)

        return context_seq                                  # (B, T', D)

    def decode(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> dict:
        """
        Run only the decoder-side heads given pre-computed memory.
        Returns same dict as forward() for compatibility.
        """
        # 6. CTL & boundary heads
        ctl_logits      = self.ctl(memory)
        # boundary_probs  = torch.sigmoid(self.boundary(memory.mean(dim=1)))

        # 7. Translator (optional — skipped when no tgt_tokens)
        logits = None
        if tgt_tokens is not None:
            logits = self.translator(memory, tgt_tokens)   # (B, T_tgt, V)

        return {
            "logits":         logits,
            "ctl_logits":     ctl_logits,
            # "boundary_probs": boundary_probs,
            "context_seq":    memory,   # [M4] always present for greedy decode
        }