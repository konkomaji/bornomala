r"""
Tier-3 Banglish transliteration model: from scratch, no pretrained
checkpoint, no fine-tuning. A small character-level Transformer
encoder-decoder (Latin characters in, Bengali codepoints out), trained on
scripts/assemble_banglish_translit_dataset.py's output.

Meant to run on Colab's free tier: small enough to train fast on a T4, and
checkpoints every `--save-every` steps to `--ckpt-dir` (point this at a
mounted Drive path in Colab) so a run survives the ~90min-idle / ~12hr
session limits - re-running this script with the same --ckpt-dir resumes
from the latest checkpoint automatically, no separate flag needed.

Usage (local smoke test, tiny data, CPU, just to prove the mechanics work):
  python scripts/train_banglish_translit.py --data-dir artifacts/banglish-translit-data \
    --ckpt-dir /tmp/banglish-ckpt --max-steps 50 --batch-size 32 --log-every 10

Usage (Colab, real run):
  python scripts/train_banglish_translit.py --data-dir /content/drive/MyDrive/banglish-data \
    --ckpt-dir /content/drive/MyDrive/banglish-ckpt --max-steps 20000 --batch-size 256 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


class TranslitDataset(Dataset):
    def __init__(self, tsv_path: str, vocab: dict):
        self.pairs = []
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                self.pairs.append((parts[0], parts[1]))
        self.src_stoi = vocab["src_stoi"]
        self.tgt_stoi = vocab["tgt_stoi"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        latin, native = self.pairs[idx]
        src = [self.src_stoi.get(c, self.src_stoi[UNK]) for c in latin]
        tgt = [self.tgt_stoi[BOS]] + [self.tgt_stoi.get(c, self.tgt_stoi[UNK]) for c in native] + [self.tgt_stoi[EOS]]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def make_collate(src_pad: int, tgt_pad: int):
    def collate(batch):
        srcs, tgts = zip(*batch)
        src_len = max(len(s) for s in srcs)
        tgt_len = max(len(t) for t in tgts)
        src_batch = torch.full((len(srcs), src_len), src_pad, dtype=torch.long)
        tgt_batch = torch.full((len(tgts), tgt_len), tgt_pad, dtype=torch.long)
        for i, s in enumerate(srcs):
            src_batch[i, :len(s)] = s
        for i, t in enumerate(tgts):
            tgt_batch[i, :len(t)] = t
        return src_batch, tgt_batch
    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TranslitTransformer(nn.Module):
    """A small from-scratch Transformer encoder-decoder. Character-level
    both sides: no subword vocabulary, so no OOV problem on noisy Latin
    spelling variance or digit-substitution slang.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, dim_ff: int = 1024, dropout: float = 0.1,
                 src_pad: int = 0, tgt_pad: int = 0):
        super().__init__()
        self.d_model = d_model
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding = (src == self.src_pad)
        tgt_key_padding = (tgt_in == self.tgt_pad)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(src.device)
        s = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        t = self.pos_enc(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer(
            s, t, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding, tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
        )
        return self.out_proj(out)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 32) -> torch.Tensor:
        self.eval()
        device = src.device
        batch = src.size(0)
        src_key_padding = (src == self.src_pad)
        s = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(s, src_key_padding_mask=src_key_padding)
        ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            t = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
            out = self.transformer.decoder(t, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding)
            logits = self.out_proj(out[:, -1, :])
            next_tok = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            done = done | (next_tok.squeeze(1) == eos_id)
            if done.all():
                break
        return ys


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: int(f[len("step-"):-len(".pt")]))
    return os.path.join(ckpt_dir, ckpts[-1])


def evaluate(model, dev_loader, vocab, device, max_batches: int = 20) -> float:
    model.eval()
    bos_id, eos_id, pad_id = vocab["tgt_stoi"][BOS], vocab["tgt_stoi"][EOS], vocab["tgt_stoi"][PAD]
    correct = total = 0
    for i, (src, tgt) in enumerate(dev_loader):
        if i >= max_batches:
            break
        src = src.to(device)
        pred = model.greedy_decode(src, bos_id, eos_id, max_len=tgt.size(1) + 4)
        for b in range(src.size(0)):
            gold = [c for c in tgt[b].tolist() if c not in (pad_id, bos_id, eos_id)]
            hyp = [c for c in pred[b].tolist() if c not in (pad_id, bos_id, eos_id)]
            if hyp and hyp[-1] == eos_id:
                hyp = hyp[:-1]
            total += 1
            if hyp == gold:
                correct += 1
    model.train()
    return correct / total if total else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    with open(os.path.join(args.data_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    src_pad, tgt_pad = vocab["src_stoi"][PAD], vocab["tgt_stoi"][PAD]

    train_ds = TranslitDataset(os.path.join(args.data_dir, "train.tsv"), vocab)
    dev_ds = TranslitDataset(os.path.join(args.data_dir, "dev.tsv"), vocab)
    collate = make_collate(src_pad, tgt_pad)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"train pairs: {len(train_ds)}, dev pairs: {len(dev_ds)}, "
          f"src vocab: {len(vocab['src_vocab'])}, tgt vocab: {len(vocab['tgt_vocab'])}", file=sys.stderr)

    model_config = {
        "src_vocab_size": len(vocab["src_vocab"]), "tgt_vocab_size": len(vocab["tgt_vocab"]),
        "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
        "dim_ff": args.dim_ff, "src_pad": src_pad, "tgt_pad": tgt_pad,
    }
    model = TranslitTransformer(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}", file=sys.stderr)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = find_latest_checkpoint(args.ckpt_dir)
    if latest is not None:
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {latest} at step {step}", file=sys.stderr)

    model.train()
    t0 = time.time()
    while step < args.max_steps:
        for src, tgt in train_loader:
            if step >= args.max_steps:
                break
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                ignore_index=tgt_pad,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"step {step}/{args.max_steps}  loss {loss.item():.4f}  "
                      f"({step / elapsed:.1f} steps/s)", file=sys.stderr)
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"step-{step}.pt")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": model_config}, ckpt_path)
                print(f"  saved {ckpt_path}", file=sys.stderr)
            if step % args.eval_every == 0:
                acc = evaluate(model, dev_loader, vocab, device)
                print(f"  dev exact-match accuracy: {acc:.1%}", file=sys.stderr)

    final_path = os.path.join(args.ckpt_dir, f"step-{step}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": model_config}, final_path)
    print(f"training done, final checkpoint: {final_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
