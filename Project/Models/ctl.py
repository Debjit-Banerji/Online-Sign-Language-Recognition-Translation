import torch
import torch.nn as nn

# -------------------------
# CTL head and boundary head
# Canonical CTL classifier follows the Sign2Vec design used in CTL/
# - average pooling over time (fallback)
# - classifier: Dropout -> Linear(d,d) -> GELU -> Dropout -> Linear(d, num_classes)
# We'll reuse that pattern here and expose `max_len` as the number of classes.
# -------------------------

class AvgPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_out, padding_mask=None):
        # encoder_out: (B, T, C) expected here
        if padding_mask is None:
            return encoder_out.mean(dim=1)
        else:
            dtype = encoder_out.dtype
            # padding_mask: (B, T) with True for padded positions
            mask = padding_mask.unsqueeze(-1).to(torch.bool)  # (B, T, 1)
            encoder_out = encoder_out.masked_fill(mask, 0.0)
            lengths = (~padding_mask).float().sum(dim=1).unsqueeze(-1)
            out = encoder_out.sum(dim=1) / lengths.clamp(min=1.0)
            return out.to(dtype)


class CTLHead(nn.Module):
    def __init__(self, hidden_dim, max_len=8, dropout=0.0):
        """Sign2Vec-style CTL classifier.

        Args:
            hidden_dim: encoder feature dim
            max_len: number of length-classes to predict
            dropout: classifier dropout probability
        """
        super().__init__()
        self.max_len = max_len
        self.pooler = AvgPooler()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_len),
        )

    def forward(self, fused_seq, padding_mask=None):
        """fused_seq: (B, T, D) -> returns logits (B, max_len)
        padding_mask (optional): (B, T) boolean mask with True for padded positions
        """
        pooled = self.pooler(fused_seq, padding_mask)  # (B, D)
        logits = self.classifier(pooled)  # (B, max_len)
        return logits


class BoundaryHead(nn.Module):
    """
    Per-frame gloss boundary prediction.
    Returns logits (not sigmoid-ed) for each frame.
    The sigmoid is applied in SLTModel.forward().
    """
 
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
 
    def forward(self, context_seq: torch.Tensor) -> torch.Tensor:
        """
        context_seq: (B, T, D)
        Returns:     (B,) — one boundary logit per clip (global mean pooled)
        """
        return self.head(context_seq.mean(dim=1)).squeeze(-1)