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
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok.banglish_tier3 import BOS, EOS, PAD, UNK, _build_model_classes

# PositionalEncoding/TranslitTransformer live in bntok.banglish_tier3 now -
# bntok (the library) needs the model class too, for load_tier3_fn(), so
# there is exactly one definition instead of this script's copy and the
# library's copy silently drifting apart.
PositionalEncoding, TranslitTransformer = _build_model_classes()


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
