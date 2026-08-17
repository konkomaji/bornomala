r"""
Assemble the training and held-out text for the BMBT-vs-v1 downstream LM
comparison (docs/known-issues.md's "downstream quality" gap - the one
unmeasured piece of BMBT's actual bet: does the structure make a language
model better per token, not just tie on token count).

Deliberately real text, not synthetic: reuses configs/bpe-64k.json, the
SAME literary-weighted corpus mix already used to train the tokenizers
themselves, at a smaller total_lines (this is a controlled relative
comparison between two tokenizers, not an attempt to pretrain a strong
model - a smaller real corpus is enough to see a real quality difference
if one exists, and keeps the two Colab training runs bounded).

Held-out: Wikipedia articles after the tokenizer training range
(WIKIPEDIA_TRAIN_ARTICLES=15000, the same disjointness scripts/compare.py
already relies on), so this reuses an established boundary rather than
inventing a new one.

Output: train.txt and held_out.txt, one line per line of real Bengali text
(NFC-normalized). Tokenization into each tokenizer's own ids happens in
scripts/prepare_lm_tokens.py, not here - this script is tokenizer-agnostic.

Usage:
  python scripts/assemble_lm_corpus.py --out-dir artifacts/lm-corpus --train-lines 200000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "bpe-64k.json"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-lines", type=int, default=200000)
    p.add_argument("--held-out-articles", type=int, default=3000)
    args = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)

    from bntok.corpus import build_configured_corpus, WIKIPEDIA_TRAIN_ARTICLES
    from bntok.normalize import normalize

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config = dict(config)
    config["total_lines"] = args.train_lines
    log(f"building train corpus: {args.train_lines} lines, same source mix as configs/bpe-64k.json ...")
    train_lines = build_configured_corpus(config, log=log)
    log(f"train: {len(train_lines)} lines")

    log(f"building held-out: Wikipedia articles after the training range "
        f"(skip {WIKIPEDIA_TRAIN_ARTICLES} ARTICLES, not lines, {args.held_out_articles} articles) ...")
    # bntok.corpus.stream_wikipedia's `limit` is an article (row) count but it
    # returns a FLATTENED list of lines (many per article) - slicing that
    # list by [WIKIPEDIA_TRAIN_ARTICLES:] would skip by LINE index, landing
    # somewhere around article ~500 rather than after article 15000, badly
    # overlapping this held-out set with build_configured_corpus's own
    # bengali_wikipedia source above (which draws from stream_wikipedia's
    # first WIKIPEDIA_TRAIN_ARTICLES articles). Caught by inspecting a first
    # run's output line count (439,303 lines from a 20-article held-out
    # request - obviously wrong) before this corpus was ever used to train
    # anything. Fixed by skipping ROWS directly against the dataset, the
    # same pattern scripts/compare.py's own held_out() already uses
    # correctly for this exact purpose.
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
    held_out_lines: list[str] = []
    articles_seen = 0
    for i, row in enumerate(ds):
        if i < WIKIPEDIA_TRAIN_ARTICLES:
            continue
        if articles_seen >= args.held_out_articles:
            break
        articles_seen += 1
        for line in row.get("text", "").split("\n"):
            if line.strip():
                held_out_lines.append(line.strip())
    log(f"held-out: {articles_seen} articles, {len(held_out_lines)} lines")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(normalize(line) + "\n")
    with open(os.path.join(args.out_dir, "held_out.txt"), "w", encoding="utf-8") as f:
        for line in held_out_lines:
            f.write(normalize(line) + "\n")
    log(f"wrote {args.out_dir}/{{train.txt, held_out.txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
