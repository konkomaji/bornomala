r"""
The one honest blind evaluation for the tier-3 transliteration model:
Dakshina's lexicon TEST split (bn.translit.sampled.test.tsv), reserved
untouched since the tier-1 lookup table was first built and never seen by
train.tsv, dev.tsv, or the synthetic generator's calibration.

Reports two numbers, not one, because fertility alone (or any single
aggregate) can hide a model that produces plausible-looking but wrong
Bengali:
  - word-level exact-match accuracy (strict: does the output match a real
    attested spelling for this word exactly)
  - character error rate (Levenshtein distance / target length, averaged) -
    partial credit for "close but not exact", which exact-match alone
    cannot show

Usage:
  python scripts/eval_banglish_translit.py \
    --dakshina-dir <path to dakshina_dataset_v1.0/bn> \
    --data-dir artifacts/banglish-translit-data \
    --ckpt <path to step-N.pt> --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_banglish_translit import BOS, EOS, PAD, UNK, TranslitTransformer


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def load_test_pairs(path: str) -> dict[str, set[str]]:
    """latin_spelling -> set of ALL real attested Bengali spellings for it
    (Dakshina's test lexicon can list more than one real spelling per
    Latin form - a prediction is correct if it matches ANY of them, not
    just the single most-attested one).
    """
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            native, roman, _cnt = parts
            out.setdefault(roman.lower(), set()).add(native)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dakshina-dir", required=True)
    p.add_argument("--data-dir", required=True, help="directory with vocab.json")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    with open(os.path.join(args.data_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    src_stoi, tgt_stoi, tgt_vocab = vocab["src_stoi"], vocab["tgt_stoi"], vocab["tgt_vocab"]
    src_pad, tgt_pad = src_stoi[PAD], tgt_stoi[PAD]
    bos_id, eos_id = tgt_stoi[BOS], tgt_stoi[EOS]

    ckpt = torch.load(args.ckpt, map_location=device)
    if "config" not in ckpt:
        raise ValueError(
            f"{args.ckpt} has no saved model config (an older checkpoint format) - "
            "cannot reconstruct the exact architecture it was trained with; retrain."
        )
    model = TranslitTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt.get('step', '?')})", file=sys.stderr)

    test_path = os.path.join(args.dakshina_dir, "lexicons", "bn.translit.sampled.test.tsv")
    test_pairs = load_test_pairs(test_path)
    latin_words = list(test_pairs.keys())
    print(f"{len(latin_words)} distinct Latin spellings in the reserved test split", file=sys.stderr)

    exact = 0
    total_cer_num = total_cer_den = 0
    for i in range(0, len(latin_words), args.batch_size):
        batch_words = latin_words[i:i + args.batch_size]
        max_len = max(len(w) for w in batch_words)
        src = torch.full((len(batch_words), max_len), src_pad, dtype=torch.long)
        for j, w in enumerate(batch_words):
            for k, c in enumerate(w):
                src[j, k] = src_stoi.get(c, src_stoi[UNK])
        src = src.to(device)
        pred = model.greedy_decode(src, bos_id, eos_id, max_len=max_len + 8)
        for j, w in enumerate(batch_words):
            hyp_ids = [c for c in pred[j].tolist() if c not in (tgt_pad, bos_id, eos_id)]
            hyp = "".join(tgt_vocab[c] for c in hyp_ids)
            golds = test_pairs[w]
            if hyp in golds:
                exact += 1
            best_dist = min(levenshtein(hyp, g) for g in golds)
            best_len = min(len(g) for g in golds) or 1
            total_cer_num += best_dist
            total_cer_den += best_len

    print(file=sys.stderr)
    print(f"word-level exact-match accuracy: {exact}/{len(latin_words)} = {exact/len(latin_words):.1%}", file=sys.stderr)
    print(f"character error rate (best-match, lower is better): {total_cer_num/total_cer_den:.1%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
