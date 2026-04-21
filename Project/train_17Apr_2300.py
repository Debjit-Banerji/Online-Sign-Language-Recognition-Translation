# train_phases.py
# Example phased training script for the SLTModel defined in models.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import json
from tqdm import tqdm
import os
import math
from Datasets.rwth_phoenix_2014t import Phoenix14TDataset, cslr_collate_fn
from Datasets.rwth_phoenix_2014 import Phoenix14Dataset
from Models import SLTModel, convert_to_deploy  # adjust import if in same package

import gc
try:
    import sacrebleu as _sacrebleu
    _HAVE_BLEU = True
except ImportError:
    _HAVE_BLEU = False

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _HAVE_ROUGE = True
except ImportError:
    _HAVE_ROUGE = False

try:
    from bert_score import score as _bert_score
    _HAVE_BERTSCORE = True
except ImportError:
    _HAVE_BERTSCORE = False

from Datasets.rwth_phoenix_2014t import Phoenix14TDataset, cslr_collate_fn
from Models import SLTModel, convert_to_deploy


# ─────────────────────────────────────────────────────────────────
#  Special-token index constants (must match dataset._build_vocab)
# ─────────────────────────────────────────────────────────────────
# These are set once from the vocab dict in run_demo and passed around.
# Never hardcode them here — always read from the vocab.
_PAD   = 0   # default fallback only
_BLANK = 1
_BOS   = 2
_EOS   = 3
_SPECIAL_IDS = {_PAD, _BLANK, _BOS, _EOS}   # updated in run_demo


# ─────────────────────────────────────────────────────────────────
#  Parameter-gating helpers
# ─────────────────────────────────────────────────────────────────

def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_keywords(model: nn.Module, keywords: list) -> None:
    for name, p in model.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad = True


def set_requires_grad(model: nn.Module, keyword: str, flag: bool) -> None:
    for name, p in model.named_parameters():
        if keyword in name:
            p.requires_grad = flag


def _ctc_input_lengths(seq_len: int, batch_size: int, device) -> torch.Tensor:
    return torch.full((batch_size,), seq_len, dtype=torch.long, device=device)


def _real_ctl_labels(
    tgt_lens: torch.Tensor, max_len: int, device
) -> torch.Tensor:
    """
    [B] tgt_lens counts BOS + glosses + EOS.
    Actual gloss count = tgt_lens - 2.
    Clamp to [0, max_len-1] for valid class index.
    """
    gloss_counts = (tgt_lens - 2).clamp(min=1)          # at least 1 gloss
    labels = (gloss_counts - 1).clamp(min=0, max=max_len - 1)
    return labels.to(device)


# ─────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────

def _wer(ref_tokens: list, hyp_tokens: list) -> float:
    """Levenshtein WER. Returns float ≥ 0; lower is better."""
    r, h = ref_tokens, hyp_tokens
    if not r:
        return float(len(h))
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)


def _ids_to_str(ids: list, vocab_inv: dict, special_ids: set) -> str:
    """
    [D] Convert token-id list to a gloss string, skipping ALL special tokens
    (pad, blank, bos, eos).  Previously only index 0 was skipped, so BOS(2)
    and EOS(3) appeared literally in hypothesis/reference strings, corrupting
    WER denominators and BLEU n-gram counts.
    """
    return " ".join(
        vocab_inv[i] for i in ids
        if i not in special_ids and i in vocab_inv
    )


def compute_metrics(
    hypotheses: list,     # list of token-id lists
    references: list,     # list of token-id lists
    vocab_inv: dict,
    special_ids: set = None,
) -> dict:
    """
    Compute WER, BLEU-4, ROUGE-L, BERTScore
    special_ids: set of token indices to strip before scoring.
    """
    if special_ids is None:
        special_ids = _SPECIAL_IDS

    hyp_strs = [_ids_to_str(h, vocab_inv, special_ids) for h in hypotheses]
    ref_strs  = [_ids_to_str(r, vocab_inv, special_ids) for r in references]

    results = {}

    # ── WER ──
    total_wer = 0.0
    for h_str, r_str in zip(hyp_strs, ref_strs):
        total_wer += _wer(r_str.split(), h_str.split())
    results["WER"] = total_wer / max(len(hyp_strs), 1)

    # ── BLEU-4 [E] ──
    if _HAVE_BLEU:
        # corpus_bleu expects: hypotheses list[str], references list[list[str]]
        refs_wrapped = [[r] for r in ref_strs]           # [E] one ref per hyp
        bleu = _sacrebleu.corpus_bleu(hyp_strs, refs_wrapped)
        results["BLEU-4"] = bleu.score
    else:
        results["BLEU-4"] = float("nan")

    # ── ROUGE-L ──
    if _HAVE_ROUGE:
        scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        total_r = sum(
            scorer.score(r, h)["rougeL"].fmeasure
            for h, r in zip(hyp_strs, ref_strs)
        )
        results["ROUGE-L"] = total_r / max(len(hyp_strs), 1)
    else:
        results["ROUGE-L"] = float("nan")
        
        
    # ── BERTScore ──
    if _HAVE_BERTSCORE:
        P, R, F1 = _bert_score(
            hyp_strs,
            ref_strs,
            lang="en",
            verbose=False
        )
        results["BERTScore-P"] = P.mean().item()
        results["BERTScore-R"] = R.mean().item()
        results["BERTScore-F1"] = F1.mean().item()
    else:
        results["BERTScore-P"] = float("nan")
        results["BERTScore-R"] = float("nan")
        results["BERTScore-F1"] = float("nan")

    return results


# ─────────────────────────────────────────────────────────────────
#  Greedy auto-regressive decoder
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    frames: torch.Tensor,
    keypoints: torch.Tensor,
    vocab: dict,
    max_decode_len: int = 60,
) -> list:
    """
    Greedy decode — runs token-by-token, feeds prediction back as next input.
    Returns list of token-id lists with BOS/EOS/PAD stripped.
    """
    device  = frames.device
    bos_idx = vocab.get("<bos>", _BOS)
    eos_idx = vocab.get("<eos>", _EOS)
    pad_idx = vocab.get("<pad>", _PAD)
    blank_idx = vocab.get("<blank>", _BLANK)
    
    # vocab_inv = {v: k for k, v in vocab.items()}   

    memory    = model.encode(frames, keypoints) # (B, T', D)
    B      = memory.size(0)

    generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    finished  = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_decode_len):
        logits     = model.translator(memory, generated)  # (B, step, V)
        # min_val = torch.finfo(logits.dtype).min

        # logits[:, :, pad_idx]   = min_val
        # logits[:, :, blank_idx] = min_val
        # logits[:, :, bos_idx]   = min_val   # BOS should never be generated

        # if generated.size(1) <= 2:
        #     logits[:, :, eos_idx] = min_val  # prevent too-early EOS
            
        # probs = logits[:, -1, :].softmax(dim=-1)            
        # vals, inds = probs.topk(5, dim=-1)
        # print(f"STEP {step}")
        # print(list(zip(inds[0].tolist(), vals[0].tolist())))
        
        next_token = logits[:, -1, :].argmax(dim=-1)      # (B,)
        # print("step", step, "next_token", next_token.item(), "token=", vocab_inv.get(next_token.item(), next_token.item()))
        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_idx),
            next_token,
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished  = finished | (next_token == eos_idx)
        if finished.all():
            break

    decoded = []
    for b in range(B):
        seq = generated[b, 1:].tolist()                   # drop BOS
        if eos_idx in seq:
            seq = seq[: seq.index(eos_idx)]
        while seq and seq[-1] == pad_idx:
            seq.pop()
        decoded.append(seq)
    return decoded

@torch.no_grad()
def _ar_greedy_single(
    model: nn.Module,
    memory: torch.Tensor,           # (1, T', D)
    vocab: dict,
    max_decode_len: int,
) -> list:
    """Autoregressive greedy fallback for a single sequence."""
    device  = memory.device
    bos_idx = vocab.get("<bos>", _BOS)
    eos_idx = vocab.get("<eos>", _EOS)
    pad_idx = vocab.get("<pad>", _PAD)
    blank_idx = vocab.get("<blank>", _BLANK)

    generated = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    for _ in range(max_decode_len):
        logits     = model.translator(memory, generated)
        # min_val = torch.finfo(logits.dtype).min
        # logits[:, :, pad_idx]   = min_val
        # logits[:, :, blank_idx] = min_val
        # logits[:, :, bos_idx]   = min_val

        # if generated.size(1) <= 2:
        #     logits[:, :, eos_idx] = min_val
        
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == eos_idx:
            break

    seq = generated[0, 1:].tolist()
    if eos_idx in seq:
        seq = seq[:seq.index(eos_idx)]
    return seq

@torch.no_grad()
def ctc_attention_decode(
    model: nn.Module,
    frames: torch.Tensor,
    keypoints: torch.Tensor,
    vocab: dict,
    ctc_aux_head: nn.Module,
    ctc_weight: float = 0.3,       # weight of CTC score in final ranking
    max_decode_len: int = 60,
) -> list:
    """
    CTC-Attention hybrid decode.

    Step 1 — CTC greedy: collapse the encoder output to a candidate sequence.
    Step 2 — Attention rescore: score the candidate with the autoregressive
              translator and combine: score = (1-w)*att_score + w*ctc_score.

    This gives CTC's alignment robustness + translator's fluency modelling.
    No beam search needed — we rescore a single CTC hypothesis.
    """
    device  = frames.device
    bos_idx = vocab.get("<bos>", _BOS)
    eos_idx = vocab.get("<eos>", _EOS)
    pad_idx = vocab.get("<pad>", _PAD)
    blank_idx = vocab.get("<blank>", _BLANK)
    
    special_ids = {pad_idx, blank_idx, bos_idx, eos_idx}

    memory    = model.encode(frames, keypoints)   # (B, T', D) — use encode() not forward()                        # (B, T', D)
    B      = memory.size(0)

    # ── Step 1b: CTC greedy on encoder output ──
    ctc_logits = ctc_aux_head(memory)    # (B, T', V)
    
    # Zero out special token logits BEFORE argmax so they can never win
    for idx in special_ids:
        ctc_logits[:, :, idx] = float('-inf')
    # for idx in [pad_idx, bos_idx, eos_idx]:
    #     ctc_logits[:, :, idx] = float('-inf')
    
    ctc_lprobs = ctc_logits.log_softmax(-1)              # (B, T', V)
    ctc_pred   = ctc_lprobs.argmax(-1)                   # (B, T')

    ctc_seqs = []
    for b in range(B):
        prev = -1
        seq  = []
        for tok in ctc_pred[b].tolist():
            if tok != prev and tok not in special_ids:
                seq.append(tok)
            prev = tok
        ctc_seqs.append(seq)

    # ── Step 2: attention rescore each CTC hypothesis ──
    decoded = []
    for b in range(B):
        ctc_seq = ctc_seqs[b]
        if not ctc_seq:
            # CTC produced nothing — fall back to autoregressive greedy
            decoded.append(_ar_greedy_single(model, memory[b:b+1], vocab, max_decode_len))
            continue

        # Build teacher-forced input for the translator: [BOS] + ctc_seq
        hyp_ids = torch.tensor(
            [bos_idx] + ctc_seq, dtype=torch.long, device=device
        ).unsqueeze(0)                                    # (1, L)
        mem_b   = memory[b:b+1]                           # (1, T', D)

        att_logits = model.translator(mem_b, hyp_ids)    # (1, L, V)
        att_lprobs = att_logits.log_softmax(-1)           # (1, L, V)

        # Score: mean token log-prob under the translator
        target_ids = torch.tensor(ctc_seq + [eos_idx], dtype=torch.long, device=device)
        L = min(att_lprobs.size(1), len(target_ids))
        att_score = att_lprobs[0, :L, :].gather(
            1, target_ids[:L].unsqueeze(1)
        ).mean().item()

        # CTC score: mean log-prob of non-blank frames
        ctc_score = ctc_lprobs[b].max(-1).values.mean().item()

        # combined = (1 - ctc_weight) * att_score + ctc_weight * ctc_score

        # For a single hypothesis there's nothing to rank against —
        # use the CTC sequence directly (rescoring is most useful with beams,
        # but even here it acts as a confidence filter: if att_score is very
        # negative, fall back to AR greedy)
        # if combined < -8.0:                            # translator strongly disagrees
        #     decoded.append(
        #         _ar_greedy_single(model, mem_b, vocab, max_decode_len)
        #     )
        # else:
        #     decoded.append(ctc_seq)
        
        # if att_score < ctc_score - 5.0:   # translator strongly contradicts CTC
        #     decoded.append(_ar_greedy_single(model, mem_b, vocab, max_decode_len))
        # else:
        decoded.append(ctc_seq)   # trust the CTC hypothesis

    return decoded

# ─────────────────────────────────────────────────────────────────
#  In-training quick eval (greedy decode on current batch)
# ─────────────────────────────────────────────────────────────────

def quick_eval_batch(
    model: nn.Module,
    frames: torch.Tensor,
    keypts: torch.Tensor,
    tgt: torch.Tensor,
    tgt_lens: torch.Tensor,
    vocab: dict,
    vocab_inv: dict,
    special_ids: set,
) -> dict:
    """
    [I] Greedy decode on the current training batch and compute metrics.
    Gives a cheap signal of free-running quality without a full val-set pass.
    """
    device = frames.device
    model.eval()
    with torch.no_grad():
        decoded = greedy_decode(model, frames, keypts, vocab)
    model.train()

    # [F] strip BOS at position 0 and EOS at position tgt_lens[b]-1
    refs = []
    for b in range(tgt.size(0)):
        # tgt[b] = [BOS, g1, g2, ..., gN, EOS, PAD, PAD, ...]
        ref = tgt[b, 1 : tgt_lens[b] - 1].tolist()       # [F] skip BOS+EOS
        refs.append(ref)

    return compute_metrics(decoded, refs, vocab_inv, special_ids)


# ─────────────────────────────────────────────────────────────────
#  Scheduled sampling helper (Phase 3)
# ─────────────────────────────────────────────────────────────────

# def _scheduled_sample(
#     model: nn.Module,
#     memory: torch.Tensor,
#     tgt: torch.Tensor,
#     ss_prob: float,
# ) -> torch.Tensor:
#     """
#     [H] Scheduled sampling: build the decoder input by mixing teacher-forced
#     tokens with the model's own predictions.

#     At each position t, with probability ss_prob, replace the teacher token
#     tgt[:,t] with argmax of logits at position t-1.

#     This closes the exposure-bias gap: CE is optimised with teacher forcing,
#     but the decoder sees some of its own errors during training, making it
#     more robust at free-running (WER) time.

#     ss_prob should increase linearly from 0.0 (start of Phase 3) toward
#     ~0.3–0.4 (end of Phase 3).  Jumping to 1.0 too fast destabilises training.
#     """
#     if ss_prob <= 0.0:
#         return tgt                                         # pure teacher forcing

#     B, T = tgt.shape
#     device = tgt.device

#     mixed = tgt.clone()
#     # We can only sample from t=1 onward (t=0 is always BOS)
#     for t in range(1, T):
#         if torch.rand(1).item() < ss_prob:
#             with torch.no_grad():
#                 partial_logits = model.translator(memory, mixed[:, :t])   # (B, t, V)
#                 pred = partial_logits[:, -1, :].argmax(dim=-1)            # (B,)
#             mixed[:, t] = pred
#     return mixed

def _scheduled_sample(
    model: nn.Module,
    memory: torch.Tensor,
    tgt: torch.Tensor,
    ss_prob: float,
) -> torch.Tensor:
    """
    Parallel scheduled sampling - samples all positions at once.
    Much faster but slightly less accurate than sequential.
    """
    if ss_prob <= 0.0:
        return tgt
    
    B, T = tgt.shape
    device = tgt.device
    
    # Create mask for positions to sample
    mask = torch.rand(B, T, device=device) < ss_prob
    mask[:, 0] = False  # Keep BOS
    
    if not mask.any():
        return tgt
    
    # Get predictions for entire sequence in one forward pass
    with torch.no_grad():
        logits = model.translator(memory, tgt[:, :-1])  # (B, T-1, V)
        predictions = logits.argmax(dim=-1)  # (B, T-1)
    
    # Create mixed input
    mixed = tgt.clone()
    # Shift predictions to align with target positions (predict token at t using context up to t-1)
    mixed[:, 1:][mask[:, 1:]] = predictions[mask[:, 1:]]
    
    return mixed


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)

def load_checkpoint(path, model):
    checkpoint = torch.load(path, weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model

# ─────────────────────────────────────────────────────────────────
#  Phase 1 — CTC encoder alignment
# ─────────────────────────────────────────────────────────────────

def phase1_train_encoders(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    vocab: dict,
    vocab_inv: dict,
    val_loader = None,
    epochs: int = 3,
    accum_steps: int = 4,
    eval_every: int = 200,
) -> None:
    """
    Train pose encoder + projection layers + TCN + temporal transformer
    with a temporary CTC head.

    [A] CTC blank uses vocab['<blank>'] (index 1), not hardcoded 0.
        pad=0 is a different token; conflating them made CTC treat every
        padding frame as a valid blank, destroying the alignment signal.
    """
    blank_idx  = vocab.get("<blank>", _BLANK)             # [A]
    pad_idx    = vocab.get("<pad>",   _PAD)
    special_ids = {vocab.get(t, i) for t, i in
                   [("<pad>",0),("<blank>",1),("<bos>",2),("<eos>",3)]}

    model.to(device)
    model.train()

    best_wer = float("inf")

    freeze_all(model)
    unfreeze_keywords(model, [
        "pose_encoder", "rgb_proj", "pose_proj",
        "tempconv", "temporal",
    ])
    set_requires_grad(model, "sla_adapter", False)
    set_requires_grad(model, "rgb_encoder", False)

    vocab_size  = model.translator.out.out_features
    fused_dim   = model.translator.d_model
    ctc_head    = nn.Linear(fused_dim, vocab_size).to(device)
    trainable   = [p for p in model.parameters() if p.requires_grad]
    trainable  += list(ctc_head.parameters())
    optimizer   = optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scaler      = torch.cuda.amp.GradScaler()
    ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)  # [A]

    for ep in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (frames, keypts, tgt_padded, tgt_lens) in enumerate(
            tqdm(dataloader, desc=f"Phase 1 ep {ep}")
        ):
            frames   = frames.to(device).float()
            keypts   = keypts.to(device).float()
            tgt_lens = tgt_lens.to(device)

            with torch.cuda.amp.autocast():
                rgb_feats  = model.rgb_encoder(frames)
                rgb_proj   = model.rgb_proj(rgb_feats)
                pose_feats = model.pose_encoder(keypts)
                pose_proj  = model.pose_proj(pose_feats)
                fused_seq  = model.fusion(rgb_proj, pose_proj)         # (B, T, D)

                logits = ctc_head(fused_seq).transpose(0, 1).log_softmax(2)  # (T, B, V)
                T_seq, B, _ = logits.shape
                input_lengths = _ctc_input_lengths(T_seq, B, device)

                # [B] tgt_lens includes BOS+EOS; CTC targets are raw gloss ids.
                # The dataset wraps targets with BOS+EOS, so we slice them out here.
                targets_list = [
                    tgt_padded[b, 1 : tgt_lens[b] - 1].to(device)    # strip BOS+EOS
                    for b in range(B)
                ]
                ctc_target_lens = torch.tensor(
                    [len(t) for t in targets_list], dtype=torch.long, device=device
                )
                targets_1d = torch.cat(targets_list) if targets_list else \
                            torch.tensor([], dtype=torch.long, device=device)

                loss = ctc_loss_fn(logits, targets_1d, input_lengths, ctc_target_lens)

            scaler.scale(loss / accum_steps).backward()

            running_loss += loss.item()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # [I] in-training metric (WER will be high in Phase 1 — expected)
            # if (step + 1) % eval_every == 0:
            #     m = quick_eval_batch(
            #         model, frames, keypts, tgt_padded, tgt_lens,
            #         vocab, vocab_inv, special_ids,
            #     )
            #     print(f"    [P1 step {step+1}] "
            #           f"WER={m['WER']:.3f}  BLEU={m['BLEU-4']:.2f}  "
            #           f"ROUGE-L={m['ROUGE-L']:.4f}  "
            #           f"ctc_loss={loss.item():.4f}")

        n = max(len(dataloader), 1)
        print(f"  Phase 1 epoch {ep} | avg CTC loss: {running_loss/n:.4f}")

        # ── SAVE LATEST ──
        save_checkpoint(model, optimizer, ep, f"checkpoints4/phase1_latest.pth")

        # ── VALIDATION ──
        if val_loader is not None:
            val_m = validate(
                model, val_loader, device,
                f"Phase 1 epoch {ep}",
                vocab, vocab_inv, special_ids,
                max_batches=1000,
            )

            # ── SAVE BEST (based on WER) ──
            if val_m["WER"] < best_wer:
                best_wer = val_m["WER"]
                save_checkpoint(model, optimizer, ep, f"checkpoints4/phase1_best_wer.pth")
                print(f"  ✓ New best Phase1 WER={best_wer:.4f}")


# ─────────────────────────────────────────────────────────────────
#  Phase 2 — fusion + translator CE training
# ─────────────────────────────────────────────────────────────────

def phase2_train_temporal_and_fusion(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    vocab: dict,
    vocab_inv: dict,
    val_loader = None,
    epochs: int = 3,
    accum_steps: int = 4,
    eval_every: int = 100,
    ss_prob_start: float = 0.0,
    ss_prob_end: float = 0.3,
    label_smoothing: float = 0.1,   # [C]
    use_pretrained: bool = False
) -> None:
    """
    Freeze encoder, train fusion + temporal + CTL + translator.

    [C] Label smoothing on CE: distributes 0.1 of probability mass across
        all vocab entries instead of concentrating on the gold token.
        This directly improves BLEU by preventing the model from being
        over-confident on the most frequent glosses.

    [B] CTL label corrected: tgt_lens - 2 gives actual gloss count.
    """
    if use_pretrained:
        model = load_checkpoint("checkpoints4/phase1_best_wer.pth", model)

    blank_idx   = vocab.get("<blank>", _BLANK)
    fused_dim   = model.translator.d_model
    vocab_size  = model.translator.out.out_features
    special_ids = {vocab.get(t, i) for t, i in
                   [("<pad>",0),("<blank>",1),("<bos>",2),("<eos>",3)]}

    # ── CTC auxiliary head (sits on encoder output, keeps encoder gradients alive) ──
    ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(device)
    ctc_loss_fn  = nn.CTCLoss(blank=blank_idx, zero_infinity=True)


    model.to(device)
    model.train()

    best_wer = float("inf")

    freeze_all(model)
    unfreeze_keywords(model, [
        "fusion", "tempconv", "temporal",
        "ctl", "boundary", "translator",
        "rgb_proj", "pose_proj",
    ])

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(ctc_aux_head.parameters()) # Add ctc_aux_head params to optimizer and trainable list

    optimizer = optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    scaler      = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )
    # [C] label_smoothing > 0 prevents probability-mass collapse onto one token
    ce = nn.CrossEntropyLoss(ignore_index=vocab.get("<pad>", _PAD),
                             label_smoothing=label_smoothing)

    max_ctl_len = model.ctl.max_len if hasattr(model.ctl, "max_len") else 8

    for ep in range(epochs):        
        running_ce  = 0.0
        running_ctl = 0.0
        optimizer.zero_grad()

        for step, (frames, keypts, tgt, tgt_lens) in enumerate(
            tqdm(dataloader, desc=f"Phase 2 ep {ep}")
        ):
            frames   = frames.to(device).float()
            keypts   = keypts.to(device).float()
            tgt      = tgt.to(device).long()
            tgt_lens = tgt_lens.to(device)

            with torch.cuda.amp.autocast():
                # tgt shape: (B, L) = [BOS, g1…gN, EOS, PAD…]
                # decoder input:  tgt[:, :-1]  = [BOS, g1…gN, EOS]  (drop last PAD)
                # decoder target: tgt[:, 1:]   = [g1…gN, EOS, PAD]  (shift left)
                # out    = model(frames, keypts, tgt_tokens=tgt[:, :-1])
                # logits = out["logits"]                                 # (B, L-1, V)
                # --- encode once ---
                # memory = model.encode(frames, keypts)

                # # --- scheduled sampling (LIGHT, even in Phase 2) ---
                # dec_input = tgt[:, :-1]

                # if ep >= 5:
                #     ss_prob_eff = min(0.1, 0.02 * (ep - 4))
                #     dec_input = _scheduled_sample(model, memory, dec_input, ss_prob=ss_prob_eff)

                # --- decode ---
                out    = model(frames, keypts, tgt_tokens=tgt[:, :-1])
                logits = out["logits"]
                B, T, V = logits.shape
                ce_loss = ce(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))

                # [B] CTL: actual gloss count = tgt_lens - 2 (BOS + EOS)
                ctl_logits = out["ctl_logits"]
                ctl_label  = _real_ctl_labels(tgt_lens, max_ctl_len, device)
                ctl_loss   = ce(ctl_logits, ctl_label)

                # ── NEW: CTC auxiliary loss on encoder output ──
                context_seq = out["context_seq"]                          # (B, T', D)
                ctc_logits  = ctc_aux_head(context_seq).transpose(0, 1).log_softmax(2)  # (T', B, V)
                T_ctc, B_ctc, _ = ctc_logits.shape
                ctc_input_lens = _ctc_input_lengths(T_ctc, B_ctc, device)
                ctc_targets_list = [
                    tgt[b, 1 : tgt_lens[b] - 1].to(device) for b in range(B_ctc)
                ]
                ctc_target_lens = torch.tensor(
                    [len(t) for t in ctc_targets_list], dtype=torch.long, device=device
                )
                ctc_targets_1d = torch.cat(ctc_targets_list) if ctc_targets_list else torch.tensor([], dtype=torch.long, device=device)
                ctc_aux_loss = ctc_loss_fn(ctc_logits, ctc_targets_1d, ctc_input_lens, ctc_target_lens)

                total_loss = (ce_loss + 0.2 * ctl_loss + 0.3 * ctc_aux_loss) / accum_steps
                # total_loss = (
                #     ce_loss * 0.7 +
                #     0.2 * ctl_loss +
                #     0.5 * ctc_aux_loss
                # ) / accum_steps
                
            scaler.scale(total_loss).backward()

            running_ce  += ce_loss.item()
            running_ctl += ctl_loss.item()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # [I] in-training metrics
            # if (step + 1) % eval_every == 0:
            #     m = quick_eval_batch(
            #         model, frames, keypts, tgt, tgt_lens,
            #         vocab, vocab_inv, special_ids,
            #     )
            #     print(f"    [P2 step {step+1}] "
            #           f"WER={m['WER']:.3f}  BLEU={m['BLEU-4']:.2f}  "
            #           f"ROUGE-L={m['ROUGE-L']:.4f}  "
            #           f"CE={ce_loss.item():.4f}  CTL={ctl_loss.item():.4f}")

        n = max(len(dataloader), 1)
        print(f"  Phase 2 epoch {ep} | avg CE: {running_ce/n:.4f}  avg CTL: {running_ctl/n:.4f}")

        # ── SAVE LATEST ──
        save_checkpoint(model, optimizer, ep, f"checkpoints4/phase2_latest.pth")
        torch.save(ctc_aux_head.state_dict(), "checkpoints4/ctc_aux_head_latest.pth")

        # ── VALIDATION ──
        if val_loader is not None:
            val_m = validate(
                model, val_loader, device,
                f"Phase 2 epoch {ep}",
                vocab, vocab_inv, special_ids,
                max_batches=1000, ctc_aux_head=ctc_aux_head
            )

            # ── SAVE BEST ──
            if val_m["WER"] < best_wer:
                best_wer = val_m["WER"]
                save_checkpoint(model, optimizer, ep, f"checkpoints4/phase2_best_wer.pth")
                torch.save(ctc_aux_head.state_dict(), "checkpoints4/ctc_aux_head_best_wer.pth")
                print(f"  ✓ New best Phase2 WER={best_wer:.4f}")

    return ctc_aux_head

# UPDATED Version after CLAUDE suggestions
def sequence_loss(
    model: nn.Module,
    memory: torch.Tensor,           # (B, T', D) — detached encoder output
    dec_input: torch.Tensor,
    tgt: torch.Tensor,              # (B, L) — full target including BOS/EOS
    vocab: dict,
    num_samples: int = 6, # Changed from 2 to 6 as suggested by Claude
) -> torch.Tensor:
    """
    REINFORCE sequence-level loss using WER as reward signal.

    Fixes vs the suggested version:
      - sampled stays as a tensor throughout (no .tolist() before gather)
      - baseline subtraction (mean reward across samples) reduces gradient
        variance enough for the signal to be useful
      - refs strip BOS/EOS before WER computation
      - rewards are normalised so the loss scale stays consistent with CE
    """
    device = memory.device
    B = tgt.size(0)
    max_len = tgt.size(1) - 1

    bos_idx = vocab.get("<bos>", 2)
    eos_idx = vocab.get("<eos>", 3)
    pad_idx = vocab.get("<pad>", 0)
    special = {bos_idx, eos_idx, pad_idx, vocab.get("<blank>", 1)}

    # Reference glosses (strip BOS at 0, EOS at end)
    refs_cpu = []
    for b in range(B):
        ref = tgt[b, 1:].tolist()
        if eos_idx in ref:
            ref = ref[: ref.index(eos_idx)]
        refs_cpu.append(ref)

    # dec_input = tgt[:, :-1]                           # (B, L-1) — teacher input
    all_log_probs = []
    all_rewards   = []

    for _ in range(num_samples):
        # ---- autoregressive sampling ----
        generated = dec_input[:, :1]
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        step_log_probs = []
        T = dec_input.size(1)
        for t in range(T):
            logits = model.translator(memory, generated)          # (B, t+1, V)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B, V)

            probs = log_probs.exp()
            next_tok = torch.multinomial(probs, num_samples=1)     # (B, 1)

            # gather log-prob of sampled token (WITH grad)
            lp = log_probs.gather(1, next_tok).squeeze(1)          # (B,)
            mask = (~finished).float()
            lp = lp * mask
            step_log_probs.append(lp)

            # mask finished sequences
            next_tok = next_tok.squeeze(1)
            next_tok = torch.where(
                finished,
                torch.full_like(next_tok, pad_idx),
                next_tok
            )

            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_idx)

            if finished.all():
                break

        log_probs_seq = torch.stack(step_log_probs, dim=1)

        # WER reward per sequence
        rewards = []
        for b in range(B):
            hyp = generated[b, 1:].tolist()
            if eos_idx in hyp:
                hyp = hyp[: hyp.index(eos_idx)]

            hyp = [t for t in hyp if t not in special]
            wer = _wer(refs_cpu[b], hyp) # Made a change such that instead of -WER we are doing e^(-WER). As recommended by CLAUDE.
            reward = math.exp(-wer)
            rewards.append(reward)
            # rewards.append(-_wer(refs_cpu[b], hyp))       # negative WER ∈ (-∞, 0]

        all_log_probs.append(log_probs_seq)
        all_rewards.append(rewards)

    # Baseline = mean reward across samples (reduces variance)
    # Shape: (num_samples, B)
    # ---- reward tensor ----
    reward_tensor = torch.tensor(
        all_rewards, dtype=torch.float32, device=device
    )  # (S, B)

    # ---- normalize (important) ----
    mean = reward_tensor.mean(dim=0, keepdim=True)   # (1, B)
    std  = reward_tensor.std(dim=0, keepdim=True)    # (1, B)
    advantage = (reward_tensor - mean) / (std + 1e-6) # As suggested by CLAUDE, to stabilize RL Training.
    advantage.detach()
    # baseline = reward_tensor.mean(dim=0, keepdim=True)    # (1, B)
    # advantage = reward_tensor - baseline                   # (S, B)

    loss = torch.tensor(0.0, device=device)
    for i in range(num_samples):
        logp = all_log_probs[i]
        adv = advantage[i].unsqueeze(-1).detach()                   # (B, 1)
        # advantage * mean log-prob across timesteps
        # loss = loss - (adv * all_log_probs[i].mean(dim=-1)).mean()
        lengths = (logp != 0).sum(dim=1).clamp(min=1)
        loss = loss - (adv * (logp.sum(dim=1) / lengths)).mean()

    return loss / num_samples


# ─────────────────────────────────────────────────────────────────
#  Phase 3 — joint fine-tuning
# ─────────────────────────────────────────────────────────────────

def phase3_joint_finetune(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    vocab: dict,
    vocab_inv: dict,
    ctc_aux_head: nn.Linear = None,
    val_loader: DataLoader = None,
    epochs: int = 2,
    accum_steps: int = 4,
    eval_every: int = 100,
    val_max_batches: int = 1000,
    label_smoothing: float = 0.1,   # [C]
    ss_prob_start: float = 0.0,     # [H] scheduled sampling start probability
    ss_prob_end:   float = 0.3,     # [H] ramps linearly to this by final epoch
    use_pretrained: bool = False
) -> None:
    """
    Joint fine-tune: SLA adapters + fusion + temporal + translator.

    [G] Two param groups with different LRs:
        - SLA adapters:  5e-6  (already pre-trained; large updates destroy
                                CLIP's visual representations → BLEU collapse)
        - Everything else: 1e-5

    [H] Scheduled sampling ramps from ss_prob_start to ss_prob_end over epochs.
        At each decoder step, with probability ss_prob the model's own previous
        prediction is used instead of the gold token. This trains the decoder to
        recover from its own errors, directly reducing free-running WER.

    [C] Label smoothing carried over from Phase 2.
    [B] CTL label corrected for BOS+EOS offset.
    """
    if use_pretrained:
        model = load_checkpoint("checkpoints4/phase2_best_wer.pth", model)

        fused_dim  = model.translator.d_model
        vocab_size = model.translator.out.out_features
        ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(device)
        ctc_aux_head = load_checkpoint("checkpoints4/ctc_aux_head_best_wer.pth", ctc_aux_head)

    special_ids = {vocab.get(t, i) for t, i in
                   [("<pad>",0),("<blank>",1),("<bos>",2),("<eos>",3)]}
    blank_idx = vocab.get("<blank>", 1)

    model.to(device)
    model.train()

    freeze_all(model)
    unfreeze_keywords(model, [
        "sla_adapter", "fusion", "tempconv", "temporal",
        "ctl", "boundary", "translator",
        "rgb_proj", "pose_proj",
    ])

    # [G] Split adapter params from everything else for different LRs
    adapter_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "sla_adapter" in n
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "sla_adapter" not in n
    ]
    optimizer = optim.AdamW([
        {"params": adapter_params, "lr": 5e-6},   # [G] gentle for pre-trained
        {"params": other_params,   "lr": 1e-5},
        {"params": list(ctc_aux_head.parameters()), "lr": 1e-5}
    ], weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler()  # for mixed precision

    ce = nn.CrossEntropyLoss(
        ignore_index=vocab.get("<pad>", _PAD),
        label_smoothing=label_smoothing,           # [C]
    )
    ctc_loss_fn  = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    max_ctl_len = model.ctl.max_len if hasattr(model.ctl, "max_len") else 8

    best_wer = float("inf")

    for ep in range(epochs):
        # [H] linearly ramp scheduled sampling probability
        if epochs > 1:
            ss_prob = ss_prob_start + (ss_prob_end - ss_prob_start) * ep / (epochs - 1)
        else:
            ss_prob = ss_prob_start
        print(f"  Phase 3 epoch {ep} | scheduled sampling prob: {ss_prob:.3f}")

        running_ce  = 0.0
        running_ctl = 0.0
        # optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)

        for step, (frames, keypts, tgt, tgt_lens) in enumerate(
            tqdm(dataloader, desc=f"Phase 3 ep {ep}")
        ):
            frames   = frames.to(device).float()
            keypts   = keypts.to(device).float()
            tgt      = tgt.to(device).long()
            tgt_lens = tgt_lens.to(device)

            with torch.cuda.amp.autocast():

                memory = model.encode(frames, keypts)

                # [H] Build mixed decoder input
                dec_input = tgt[:, :-1]                           # (B, L-1)
                if ss_prob > 0.0:
                    dec_input = _scheduled_sample(model, memory, dec_input, ss_prob)

                # Full forward with (possibly mixed) decoder input
                out    = model.decode(memory, tgt_tokens=dec_input)
                logits = out["logits"]
                B, T, V = logits.shape
                ce_loss = ce(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))

                # [B] CTL with corrected label
                ctl_logits = out["ctl_logits"]
                ctl_label  = _real_ctl_labels(tgt_lens, max_ctl_len, device)
                ctl_loss   = ce(ctl_logits, ctl_label)

                # ── NEW: CTC auxiliary loss on encoder output ──
                if ctc_aux_head is not None:
                    context_seq = out["context_seq"]                          # (B, T', D)
                    ctc_logits  = ctc_aux_head(context_seq).transpose(0, 1).log_softmax(2)  # (T', B, V)
                    T_ctc, B_ctc, _ = ctc_logits.shape
                    ctc_input_lens = _ctc_input_lengths(T_ctc, B_ctc, device)
                    ctc_targets_list = [
                        tgt[b, 1 : tgt_lens[b] - 1].to(device) for b in range(B_ctc)
                    ]
                    ctc_target_lens = torch.tensor(
                        [len(t) for t in ctc_targets_list], dtype=torch.long, device=device
                    )
                    ctc_targets_1d = torch.cat(ctc_targets_list)
                    ctc_aux_loss = ctc_loss_fn(ctc_logits, ctc_targets_1d, ctc_input_lens, ctc_target_lens)
                else:
                    ctc_aux_loss = 0

                # Warm up with CE only, then introduce REINFORCE
                use_reinforce = (ep >= 1)  # or based on val WER crossing a threshold
                if use_reinforce:
                    seq_loss   = sequence_loss(model, memory.detach(), dec_input, tgt, vocab)  # REINFORCE with WER reward
                    total_loss = (ce_loss + 0.1 * ctl_loss + 0.05 * seq_loss + 0.3 * ctc_aux_loss) / accum_steps # 0.05 acc. to GPT to reduce REINFORCE's weight
                else:
                    total_loss = (ce_loss + 0.1 * ctl_loss + 0.3 * ctc_aux_loss) / accum_steps

            scaler.scale(total_loss).backward()

            running_ce  += ce_loss.item()
            running_ctl += ctl_loss.item()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    adapter_params + other_params + list(ctc_aux_head.parameters()), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)

            # [I] in-training metrics
            # if (step + 1) % eval_every == 0:
            #     m = quick_eval_batch(
            #         model, frames, keypts, tgt, tgt_lens,
            #         vocab, vocab_inv, special_ids,
            #     )
            #     print(f"    [P3 step {step+1}] "
            #           f"WER={m['WER']:.3f}  BLEU={m['BLEU-4']:.2f}  "
            #           f"ROUGE-L={m['ROUGE-L']:.4f}  "
            #           f"CE={ce_loss.item():.4f}  ss={ss_prob:.2f}")

        n = max(len(dataloader), 1)
        print(f"  Phase 3 epoch {ep} | "
              f"avg CE: {running_ce/n:.4f}  avg CTL: {running_ctl/n:.4f}")

        # ── SAVE LATEST ──
        save_checkpoint(model, optimizer, ep, f"checkpoints4/phase3_latest.pth")

        torch.cuda.empty_cache() 
        gc.collect()

        # Full val-set decode at the end of each epoch
        if val_loader is not None:
            val_m = validate(
                model, val_loader, device,
                f"Phase 3 epoch {ep}", vocab, vocab_inv, special_ids,
                max_batches=val_max_batches, ctc_aux_head=ctc_aux_head
            )

            # ── SAVE BEST ──
            # if val_m["WER"] < best_wer: #issue: note saving ctc
            #     best_wer = val_m["WER"]
            #     save_checkpoint(model, optimizer, ep, f"checkpoints/phase3_best_wer.pth")
            #     print(f"  ✓ New best Phase3 WER={best_wer:.4f}")

            # ── SAVE BEST ──
            if val_m["WER"] < best_wer:
                best_wer = val_m["WER"]
                save_checkpoint(model, optimizer, ep, f"checkpoints4/phase3_best_wer.pth")
                # Add this so the hybrid decoder works at test time!
                if ctc_aux_head is not None:
                    torch.save(ctc_aux_head.state_dict(), "checkpoints4/ctc_aux_head_phase3_best.pth")
                print(f"  ✓ New best Phase3 WER={best_wer:.4f}")


# ─────────────────────────────────────────────────────────────────
#  Validation — full val-set greedy decode
# ─────────────────────────────────────────────────────────────────

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    phase_name: str,
    vocab: dict,
    vocab_inv: dict,
    special_ids: set = None,
    max_batches: int = 100,
    ctc_aux_head: nn.Module = None
) -> dict:
    """
    [F] Reference slice: tgt[b, 1 : tgt_lens[b]-1] strips BOS at index 0
        and EOS at index tgt_lens[b]-1, so the reference matches the
        hypothesis format (raw gloss ids only).

    [D] ids_to_str filters all special tokens via special_ids set.
    """
    if special_ids is None:
        special_ids = {vocab.get(t, i) for t, i in
                       [("<pad>",0),("<blank>",1),("<bos>",2),("<eos>",3)]}

    model.eval()
    all_hyps, all_refs = [], []
    print(f"\n--- Validation: {phase_name} ---")

    with torch.no_grad():
        for i, (frames, keypts, tgt, tgt_lens) in enumerate(
            tqdm(dataloader, desc="val", total=min(max_batches, len(dataloader)))
        ):
                
            if i >= max_batches:
                break
            frames = frames.to(device).float()
            keypts = keypts.to(device).float()
            
            # if i == 0:
            #     memory = model.encode(frames, keypts)
            #     # print("memory debug:", memory)
            #     bos = torch.full((1, 1), vocab["<bos>"], device=device)
            #     logits1 = model.translator(memory, bos)
            #     probs = logits1.softmax(dim=-1)
            #     topk = 10
            #     vals, inds = probs.topk(topk, dim=-1)
            #     # squeeze batch/seq dims if present, move to CPU for printing
            #     vals_s = vals.squeeze().cpu().tolist()
            #     inds_s = inds.squeeze().cpu().tolist()
            #     # ensure lists when single-dim squeeze returns scalar
            #     if isinstance(vals_s, float):
            #         vals_s = [vals_s]
            #     if isinstance(inds_s, int):
            #         inds_s = [inds_s]
            #     print(list(zip(inds_s, vals_s)))
            #     logits2 = model.translator(torch.zeros_like(memory), bos)
            #     print(logits1.abs().mean().item())
            #     print(logits2.abs().mean().item())
            #     print((logits1 - logits2).abs().mean().item())
            
            # with torch.amp.autocast('cuda'): #added
            if ctc_aux_head is not None:
                decoded = ctc_attention_decode(
                    model, frames, keypts, vocab, ctc_aux_head
                )
            else:
                decoded = greedy_decode(model, frames, keypts, vocab)
                    
            print(decoded)

            for b in range(tgt.size(0)):
                # [F] strip BOS (pos 0) and EOS (pos tgt_lens[b]-1)
                ref = tgt[b, 1 : tgt_lens[b] - 1].tolist()
                all_hyps.append(decoded[b])
                all_refs.append(ref)

    metrics = compute_metrics(all_hyps, all_refs, vocab_inv, special_ids)
    no_bleu  = " (install sacrebleu)"  if not _HAVE_BLEU  else ""
    no_rouge = " (install rouge-score)" if not _HAVE_ROUGE else ""
    no_bert  = " (install bert-score)" if not _HAVE_BERTSCORE  else ""
    print(f"  WER:     {metrics['WER']:.4f}  ↓ lower is better")
    print(f"  BLEU-4:  {metrics['BLEU-4']:.2f}{no_bleu}")
    print(f"  ROUGE-L: {metrics['ROUGE-L']:.4f}{no_rouge}")
    print(f"  BERT-P:  {metrics['BERTScore-P']:.4f}{no_bert}")
    print(f"  BERT-R:  {metrics['BERTScore-R']:.4f}{no_bert}")
    print(f"  BERT-F1: {metrics['BERTScore-F1']:.4f}{no_bert}")

    model.train()
    return metrics

# ─────────────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────────────

def run_demo():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = ("/home/abdullahm/jaleel/CV_project/CLIP-SLA/Data/"
                 "PHOENIX-2014-T-release-v3/PHOENIX-2014-T")

    import clip as openai_clip
    clip_model, _ = openai_clip.load("ViT-B/16", device=device)

    pre_trained = torch.load("Pre-Trained Models/best_model.pth", map_location=device)
    rgb_weights  = {k[15:]: v for k, v in pre_trained.items()
                    if k.startswith("visual_encoder.")}

    # train_dataset = Phoenix14TDataset(root_dir=root_path, split="train", is_training=True)
    # val_dataset   = Phoenix14TDataset(root_dir=root_path, split="dev",
    #                                   vocab=train_dataset.vocab, is_training=False)
    # test_dataset  = Phoenix14TDataset(root_dir=root_path, split="test",
    #                                   vocab=train_dataset.vocab, is_training=False)

    # os.makedirs("checkpoints", exist_ok=True)
    # with open("checkpoints/vocab.json", "w") as f:
    #     json.dump(train_dataset.vocab, f)

    # vocab     = train_dataset.vocab
    # vocab_inv = {v: k for k, v in vocab.items()}


    #**
    # 1. Initialize train dataset
    train_dataset = Phoenix14TDataset(root_dir=root_path, split="train", is_training=True)
    os.makedirs("checkpoints4", exist_ok=True)

    # 2. 🚨 CRITICAL: Load saved vocab if it exists so IDs don't scramble 🚨
    if os.path.exists("checkpoints4/vocab.json"):
        print("Loading saved vocab from disk to prevent ID scrambling...")
        with open("checkpoints4/vocab.json", "r") as f:
            vocab = json.load(f)
        # Override the randomly generated vocab with the saved one
        train_dataset.vocab = vocab 
    else:
        print("No saved vocab found. Creating and saving a new one...")
        vocab = train_dataset.vocab
        with open("checkpoints4/vocab.json", "w") as f:
            json.dump(vocab, f)

    vocab_inv = {v: k for k, v in vocab.items()}

    # 3. Pass the consistent vocab to Val and Test
    val_dataset   = Phoenix14TDataset(root_dir=root_path, split="dev",
                                      vocab=vocab, is_training=False)
    test_dataset  = Phoenix14TDataset(root_dir=root_path, split="test",
                                      vocab=vocab, is_training=False)
    #**
    
    
    # Derive special_ids from the actual vocab so all helpers are consistent
    special_ids = {vocab.get(t, i) for t, i in
                   [("<pad>",0),("<blank>",1),("<bos>",2),("<eos>",3)]}

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        collate_fn=cslr_collate_fn, pin_memory=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=cslr_collate_fn, pin_memory=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=cslr_collate_fn, pin_memory=False, num_workers=0,
    )

    vocab_size = len(vocab)

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
        pre_trained_rgb_encoder_weights=rgb_weights,
    )
    model.to(device)
    print("Model loaded.\n")

    is_run_phase1 = True
    is_run_phase2 = True
    is_run_phase3 = True

    # ── Phase 1 ──
    print("=== Phase 1: Encoder Alignment (CTC) ===")
    if is_run_phase1:
        phase1_train_encoders(
            model, train_loader, device,
            vocab=vocab, vocab_inv=vocab_inv, val_loader=val_loader,
            epochs=10, accum_steps=4, eval_every=400,
        )
        torch.cuda.empty_cache(); gc.collect()
    else:
        print("Skipping Phase 1. Utilizing Models Trained Earlier!")

    # ── Phase 2 ──
    print("\n=== Phase 2: Temporal & Translator ===")
    if is_run_phase2:
        ctc_aux_head = phase2_train_temporal_and_fusion(
            model, train_loader, device,
            vocab=vocab, vocab_inv=vocab_inv, val_loader=val_loader,
            epochs=20, accum_steps=4, eval_every=200,
            label_smoothing=0.1, use_pretrained=not is_run_phase1
        )
        torch.cuda.empty_cache(); gc.collect()
    else:
        ctc_aux_head = None
        print("Skipping Phase 2. Utilizing Models Trained Earlier!")

    # ── Phase 3 ──
    print("\n=== Phase 3: Joint Fine-tuning ===")
    if is_run_phase3:
        phase3_joint_finetune(
            model, train_loader, device,
            vocab=vocab, vocab_inv=vocab_inv, ctc_aux_head=ctc_aux_head, val_loader = val_loader,
            epochs=5, accum_steps=4, eval_every=200,
            val_max_batches=1000,
            label_smoothing=0.1,
            ss_prob_start=0.0,
            ss_prob_end=0.3,
            use_pretrained=not is_run_phase2
        )
        torch.cuda.empty_cache(); gc.collect()
    else:
        print("Skipping Phase 3. Utilizing Models Trained Earlier!")

    # ── Deploy ──
    # print("\nConverting to deploy mode …")
    # model.eval()
    # convert_to_deploy(model)
    # torch.save(model.state_dict(), "checkpoints/slt_model_deployed.pth")
    # print("Done.")
    # if is_run_phase1 or is_run_phase2 or is_run_phase3: #added
    #     print("\nConverting to deploy mode …")
    #     model.eval()
    #     convert_to_deploy(model)
    #     torch.save(model.state_dict(), "checkpoints/slt_model_deployed.pth")
    #     print("Done.")
    # else: #added
    #     print("\nLoading previously deployed model for testing...")
    #     model.eval()
    #     convert_to_deploy(model)
    #     if os.path.exists("checkpoints/slt_model_deployed.pth"):
    #         model.load_state_dict(torch.load("checkpoints/slt_model_deployed.pth", map_location=device))
    #     else:
    #         print("Warning: slt_model_deployed.pth not found!")

    # ── Test Evaluation & Deploy ──
    # print("\n=== Preparing Model for Testing ===")
    # if os.path.exists("checkpoints/phase3_best_wer.pth"):
    #     print("Loading Phase 3 best weights...")
    #     model = load_checkpoint("checkpoints/phase3_best_wer.pth", model)
    # elif os.path.exists("checkpoints/phase2_best_wer.pth"):
    #     print("Loading Phase 2 best weights...")
    #     model = load_checkpoint("checkpoints/phase2_best_wer.pth", model)
    # else:
    #     print("Warning: No best weights found! Testing with current initialized weights.")

    # # Load best CTC head for hybrid decoding
    # fused_dim = model.translator.d_model
    # vocab_size = model.translator.out.out_features
    # ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(device)
    # if os.path.exists("checkpoints/ctc_aux_head_best_wer.pth"):
    #     print("Loading CTC auxiliary head...")
    #     ctc_aux_head.load_state_dict(torch.load("checkpoints/ctc_aux_head_best_wer.pth", map_location=device))
    #     ctc_aux_head.eval()
    # else:
    #     ctc_aux_head = None
    #     print("Warning: No CTC head found. Falling back to greedy decoding.")

    # model.eval()
    # print("Converting architecture to deploy mode...")
    # convert_to_deploy(model)
    # model.to(device) # Ensure new reparameterized layers are properly on the GPU
    
    # print("Saving slt_model_deployed.pth...")
    # torch.save(model.state_dict(), "checkpoints/slt_model_deployed.pth")
    
    # print("\n=== Testing on Test Set ===")
    # validate(
    #     model, test_loader, device,
    #     "Test Evaluation", vocab, vocab_inv, special_ids,
    #     max_batches=len(test_loader), ctc_aux_head=ctc_aux_head
    # )


# # new test evaluation
#     # ── Test Evaluation ──
#     print("\n=== Preparing Model for Testing ===")
#     if os.path.exists("checkpoints/phase3_best_wer.pth"):
#         print("Loading Phase 3 best weights...")
#         model = load_checkpoint("checkpoints/phase3_best_wer.pth", model)
#     elif os.path.exists("checkpoints/phase2_best_wer.pth"):
#         print("Loading Phase 2 best weights...")
#         model = load_checkpoint("checkpoints/phase2_best_wer.pth", model)
#     else:
#         print("Warning: No best weights found!")

#     # 1. Ensure model is in eval mode so BatchNorm/Dropout act correctly
#     model.eval()
#     model.to(device)
    
#     # 2. 🚨 CRITICAL: Do NOT call convert_to_deploy 🚨
#     # print("Converting architecture to deploy mode...")
#     # convert_to_deploy(model) 
    
#     # 3. Force pure greedy decoding to isolate the main model
#     ctc_aux_head = None 
    
#     print("\n=== Testing on Test Set (Without Deploy Mode) ===")
#     validate(
#         model, test_loader, device,
#         "Test Evaluation", vocab, vocab_inv, special_ids,
#         max_batches=len(test_loader), ctc_aux_head=ctc_aux_head
#     )

    # ── Final Test Evaluation ──
    print("\n=== Preparing Model for Final Testing ===")
    
    # 1. Load best translator weights
    if os.path.exists("checkpoints4/phase3_best_wer.pth"):
        print("Loading Phase 3 best weights...")
        model = load_checkpoint("checkpoints4/phase3_best_wer.pth", model)
    elif os.path.exists("checkpoints4/phase2_best_wer.pth"):
        print("Loading Phase 2 best weights...")
        model = load_checkpoint("checkpoints4/phase2_best_wer.pth", model)
    else:
        print("Warning: No trained weights found!")

    # 2. Load best CTC head for hybrid decoding
    fused_dim = model.translator.d_model
    ctc_aux_head = nn.Linear(fused_dim, vocab_size).to(device)
    
    if os.path.exists("checkpoints4/ctc_aux_head_phase3_best.pth"):
        print("Loading Phase 3 CTC auxiliary head...")
        ctc_aux_head.load_state_dict(torch.load("checkpoints4/ctc_aux_head_phase3_best.pth", map_location=device))
        ctc_aux_head.eval()
    elif os.path.exists("checkpoints4/ctc_aux_head_best_wer.pth"):
        print("Loading Phase 2 CTC auxiliary head...")
        ctc_aux_head.load_state_dict(torch.load("checkpoints4/ctc_aux_head_best_wer.pth", map_location=device))
        ctc_aux_head.eval()
    else:
        print("Warning: No CTC head found. Falling back to pure greedy decoding.")
        ctc_aux_head = None

    # 3. Setup for inference
    model.eval()
    model.to(device)
    
    # Keeping deploy mode off for the first verification run
    # convert_to_deploy(model) 
    
    print("\n=== Testing on Test Set ===")
    validate(
        model, test_loader, device,
        "Final Test Evaluation", vocab, vocab_inv, special_ids,
        max_batches=len(test_loader), ctc_aux_head=ctc_aux_head
    )
if __name__ == "__main__":
    run_demo()
