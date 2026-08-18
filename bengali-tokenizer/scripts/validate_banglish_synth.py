r"""Measure bntok.banglish_synth's word-level hit rate against real Dakshina
spellings, reproducing the methodology docs/known-issues.md already cites
(3000-word sample, N-seed coverage per word: does ANY of the sampled random
renders match a real recorded spelling for that word).

Uses artifacts/banglish-translit-data/dev.tsv as the real-spelling reference
(Dakshina lexicon dev split, real-only, already grouped bengali -> latin
variants seen in the wild) rather than re-downloading the raw Dakshina
release - dev.tsv already IS that data, unmodified, one (latin, bengali) pair
per line.

Usage:
    python scripts/validate_banglish_synth.py --dev artifacts/banglish-translit-data/dev.tsv \
        [--sample 3000] [--seeds 15] [--seed-base 0]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from bntok.banglish_synth import render_word_latin


def load_real_spellings(path: str) -> dict[str, set[str]]:
    """bengali word -> set of real latin spellings seen in dev.tsv."""
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            latin, bengali = line.split("\t", 1)
            out[bengali].add(latin)
    return out


def hit_rate(words: list[str], real: dict[str, set[str]], seeds: int, seed_base: int) -> tuple[float, list[str]]:
    hits = 0
    misses: list[str] = []
    for w in words:
        real_spellings = real[w]
        found = False
        for s in range(seeds):
            rendered = render_word_latin(w, random.Random(seed_base + s))
            if rendered in real_spellings:
                found = True
                break
        if found:
            hits += 1
        else:
            misses.append(w)
    return hits / len(words) if words else 0.0, misses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", required=True)
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--show-misses", type=int, default=15)
    args = p.parse_args()

    real = load_real_spellings(args.dev)
    all_words = sorted(real.keys())
    rng = random.Random(42)
    sample = rng.sample(all_words, min(args.sample, len(all_words)))

    rate, misses = hit_rate(sample, real, args.seeds, args.seed_base)
    print(f"sampled {len(sample)} words, {args.seeds} seeds each")
    print(f"word-level hit rate: {rate * 100:.1f}%")
    if args.show_misses:
        print(f"\nfirst {args.show_misses} misses (bengali -> real spellings):")
        for w in misses[: args.show_misses]:
            print(f"  {w} -> {sorted(real[w])}")


if __name__ == "__main__":
    main()
