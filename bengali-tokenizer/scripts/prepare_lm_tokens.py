r"""
Tokenize scripts/assemble_lm_corpus.py's train.txt/held_out.txt with ONE
tokenizer (v1 bn-bpe-64k or BMBT bmbt-64k) into a flat token-id array, ready
for scripts/train_lm.py. Run this twice, once per tokenizer, into separate
output directories - that pair of token streams is the controlled
comparison: same text, same everything else, only the tokenizer differs.

Concatenates all lines into one long token stream (each line's ids
followed by the tokenizer's own end-of-word/line boundary handling; no
extra separator injected beyond what each tokenizer already encodes),
saved as a flat uint32 numpy array via memmap - scripts/train_lm.py samples
random contiguous blocks from this for training, the standard next-token-
prediction LM setup.

Also records the held-out text's total UTF-8 byte length, the fixed
denominator scripts/train_lm.py needs to convert held-out cross-entropy
into bits-per-byte - the metric that makes the v1-vs-BMBT comparison fair
despite the two tokenizers' vocabularies (and therefore raw per-token
perplexity) not being directly comparable.

Usage:
  python scripts/prepare_lm_tokens.py --corpus-dir artifacts/lm-corpus \
    --tokenizer artifacts/bn-bpe-64k --out-dir artifacts/lm-tokens-v1
  python scripts/prepare_lm_tokens.py --corpus-dir artifacts/lm-corpus \
    --tokenizer artifacts/bmbt-64k --bmbt --out-dir artifacts/lm-tokens-bmbt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize_file(path: str, tok, log) -> np.ndarray:
    ids: list[int] = []
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.extend(tok.encode(line))
            n_lines += 1
            if n_lines % 50000 == 0:
                log(f"  {n_lines} lines, {len(ids)} tokens so far ...")
    log(f"  done: {n_lines} lines, {len(ids)} tokens")
    return np.array(ids, dtype=np.uint32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", required=True, help="dir with train.txt and held_out.txt")
    p.add_argument("--tokenizer", required=True, help="tokenizer artifact directory")
    p.add_argument("--bmbt", action="store_true", help="tokenizer is a BMBT artifact, not v1 BengaliTokenizer")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)

    if args.bmbt:
        from bntok.bmbt import BMBT
        tok = BMBT.load(args.tokenizer)
    else:
        from bntok import BengaliTokenizer
        tok = BengaliTokenizer.load(args.tokenizer)
    vocab_size = tok.config["actual_vocab_size"]
    log(f"loaded {args.tokenizer} (vocab_size={vocab_size})")

    os.makedirs(args.out_dir, exist_ok=True)

    log("tokenizing train.txt ...")
    train_ids = tokenize_file(os.path.join(args.corpus_dir, "train.txt"), tok, log)
    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))

    log("tokenizing held_out.txt ...")
    held_out_path = os.path.join(args.corpus_dir, "held_out.txt")
    held_out_ids = tokenize_file(held_out_path, tok, log)
    held_out_ids.tofile(os.path.join(args.out_dir, "held_out.bin"))
    held_out_bytes = os.path.getsize(held_out_path)  # includes the newlines this script wrote; consistent across both tokenizer runs since it's the SAME held_out.txt file both times

    meta = {
        "vocab_size": int(vocab_size),
        "train_tokens": int(len(train_ids)),
        "held_out_tokens": int(len(held_out_ids)),
        "held_out_bytes": int(held_out_bytes),
        "tokenizer_path": args.tokenizer,
        "bmbt": args.bmbt,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {args.out_dir}/{{train.bin, held_out.bin, meta.json}}")
    log(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
