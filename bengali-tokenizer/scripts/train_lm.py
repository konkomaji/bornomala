r"""
The BMBT-vs-v1 downstream comparison: a small decoder-only Transformer LM,
trained from scratch (no pretrained weights), on ONE tokenizer's token
stream (scripts/prepare_lm_tokens.py's output). Run this script twice - once
per tokenizer directory - with IDENTICAL hyperparameters; the only thing
that differs between the two runs is which tokenizer produced the input
token stream. That is the whole experiment.

Reports held-out BITS-PER-BYTE, not raw per-token perplexity: v1 and BMBT
have different vocabularies and encode the same text into different token
counts (even though their fertility nearly ties), so per-token
cross-entropy alone is not a fair comparison. Bits-per-byte fixes the
denominator to the held-out text's actual UTF-8 byte length (recorded in
prepare_lm_tokens.py's meta.json), the standard way to compare language
models across tokenizers.

Meant for Colab, same checkpoint/resume discipline as
scripts/train_banglish_translit.py: point --ckpt-dir at Drive, re-run to
resume.

Usage:
  python scripts/train_lm.py --tokens-dir artifacts/lm-tokens-v1 \
    --ckpt-dir /content/drive/MyDrive/lm-ckpt-v1 --device cuda
  python scripts/train_lm.py --tokens-dir artifacts/lm-tokens-bmbt \
    --ckpt-dir /content/drive/MyDrive/lm-ckpt-bmbt --device cuda
  # (identical --max-steps/--batch-size/--block-size/--d-model/etc for both runs)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GPT(nn.Module):
    """A small, standard decoder-only Transformer (GPT-style), from scratch.
    Nothing tokenizer-specific here - it only sees integer ids, which is the
    point: the SAME architecture, fed two different tokenizers' ids, is what
    makes this a controlled comparison of the tokenizer, not the model.
    """

    def __init__(self, vocab_size: int, block_size: int, d_model: int = 384,
                 nhead: int = 6, num_layers: int = 6, dim_ff: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(block_size, block_size) * float("-inf"), diagonal=1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_embed(idx) + self.pos_embed(pos))
        x = self.blocks(x, mask=self.causal_mask[:t, :t])
        x = self.ln_f(x)
        return self.head(x)


class BinaryTokenDataset:
    """Memory-mapped random block sampling over a flat uint32 token array -
    the standard nanoGPT-style setup, so training does not load the whole
    token stream into RAM.
    """

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: torch.device):
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f[len("step-"):-len(".pt")]))
    return os.path.join(ckpt_dir, ckpts[-1])


@torch.no_grad()
def eval_bits_per_byte(model, held_out_path: str, held_out_bytes: int, block_size: int,
                        device: torch.device, max_blocks: int = 200) -> float:
    """Held-out cross-entropy, summed in nats, converted to bits, divided by
    the held-out text's fixed original UTF-8 byte count - NOT by the token
    count, which is what makes this comparable across tokenizers with
    different vocabularies and different token counts for the same text.
    """
    model.eval()
    data = np.memmap(held_out_path, dtype=np.uint32, mode="r")
    n_blocks = min(max_blocks, max(1, (len(data) - 1) // block_size))
    total_nats = 0.0
    total_tokens = 0
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, len(data) - 1)
        if end <= start:
            break
        x = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start + 1:end + 1].astype(np.int64)).unsqueeze(0).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nats += loss.item()
        total_tokens += y.numel()
    model.train()
    total_bits = total_nats / math.log(2)
    # scale to the FULL held-out set's byte count proportionally to how much
    # of it these max_blocks blocks actually covered, so bits/byte stays a
    # correctly normalized rate regardless of max_blocks
    fraction_covered = (n_blocks * block_size) / max(1, len(data))
    bytes_covered = held_out_bytes * min(1.0, fraction_covered)
    return total_bits / bytes_covered if bytes_covered > 0 else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", required=True, help="output of scripts/prepare_lm_tokens.py")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=1536)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                    help="micro-batches accumulated per optimizer step. Effective batch = "
                         "--batch-size * --grad-accum-steps. The 64k-vocab head's logits tensor "
                         "is batch_size * block_size * vocab_size floats (e.g. 64*256*64000*4 "
                         "bytes = ~4.2GB at the defaults) - a T4 can OOM on that alone before "
                         "the rest of the model even factors in. Lower --batch-size and raise "
                         "this to keep the SAME effective batch size (keeps the two runs' "
                         "hyperparameters comparable) while shrinking the tensor that actually "
                         "OOMs.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    with open(os.path.join(args.tokens_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    print(f"tokens-dir meta: {meta}", file=sys.stderr)

    train_ds = BinaryTokenDataset(os.path.join(args.tokens_dir, "train.bin"), args.block_size)
    held_out_path = os.path.join(args.tokens_dir, "held_out.bin")

    model_config = {
        "vocab_size": meta["vocab_size"], "block_size": args.block_size, "d_model": args.d_model,
        "nhead": args.nhead, "num_layers": args.num_layers, "dim_ff": args.dim_ff,
    }
    model = GPT(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = find_latest_checkpoint(args.ckpt_dir)
    if latest is not None:
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {latest} at step {step}", file=sys.stderr)

    if args.grad_accum_steps > 1:
        print(f"effective batch size: {args.batch_size * args.grad_accum_steps} "
              f"({args.batch_size} x {args.grad_accum_steps} grad-accum steps)", file=sys.stderr)

    model.train()
    t0 = time.time()
    while step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = train_ds.get_batch(args.batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss / args.grad_accum_steps).backward()
            accum_loss += loss.item() / args.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{args.max_steps}  loss {accum_loss:.4f}  "
                  f"({step / elapsed:.1f} steps/s)", file=sys.stderr)
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"step-{step}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": model_config}, ckpt_path)
            print(f"  saved {ckpt_path}", file=sys.stderr)
        if step % args.eval_every == 0:
            bpb = eval_bits_per_byte(model, held_out_path, meta["held_out_bytes"], args.block_size, device)
            print(f"  held-out bits-per-byte: {bpb:.4f}", file=sys.stderr)

    final_path = os.path.join(args.ckpt_dir, f"step-{step}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": model_config}, final_path)
    final_bpb = eval_bits_per_byte(model, held_out_path, meta["held_out_bytes"], args.block_size,
                                    device, max_blocks=100000)
    print(f"training done, final checkpoint: {final_path}", file=sys.stderr)
    print(f"FINAL held-out bits-per-byte (full held-out set): {final_bpb:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
