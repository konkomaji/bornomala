r"""
Build the tier-1 Banglish lookup table (real romanized-Bengali surface form
-> canonical Bengali word), the cheap O(1) tier of the tiered-cascade design
(docs/known-issues.md, Banglish section): most real traffic resolves here,
never touching a model at all.

Sources, real trust ordering (real data always wins over synthetic):
  1. Dakshina v1.0 (Google Research, MIT-licensed) word lexicon, train+dev
     splits only - the test split is reserved untouched, so this table can
     later be evaluated against it honestly, the same disjoint-held-out
     discipline bntok.corpus already uses for tokenizer training.
  2. Dakshina's natural romanized-sentence word alignment (a separate
     collection method in the same release: real full sentences, not
     elicited single words), all of it - no held-out concern here, because
     the pipeline's real generalization test is against scripts/compare.py's
     `banglish` register (CC-100 bn_rom), a wholly independent source.
  3. Synthetic pairs (bntok.banglish_synth) generated from our own training
     corpus, added ONLY for Bengali words absent from (1)/(2) - gap-filling,
     never overriding a real observation, and tagged "synthetic" in the
     output so a consumer can tell the difference.

Usage:
  python scripts/build_banglish_lookup.py --dakshina-dir <path to dakshina_dataset_v1.0/bn> --out artifacts/banglish-lookup.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_lexicon_counts(path: str) -> Counter:
    """(bengali_word, latin_spelling) -> real attested count, from one
    Dakshina lexicon TSV split (native TAB roman TAB count)."""
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            native, roman, cnt = parts
            try:
                c = int(cnt)
            except ValueError:
                continue
            counts[(native, roman.lower())] += c
    return counts


def load_aligned_counts(path: str) -> Counter:
    """(bengali_word, latin_spelling) -> occurrence count, from Dakshina's
    word-aligned natural-sentence resource (native TAB roman, one word pair
    per line, no count column - each line is one real occurrence)."""
    counts: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            native, roman = parts
            counts[(native, roman.lower())] += 1
    return counts


def build_table(dakshina_dir: str, log=lambda m: None) -> dict[str, tuple[str, int, str, str]]:
    """Merge sources into latin_spelling -> (bengali_word, count, source).

    On a collision (same Latin spelling attested for more than one Bengali
    word), the higher-count candidate is the table's primary entry, but the
    runner-up is kept too (`table[latin]`'s 4th element), not silently
    dropped. Checked directly rather than assumed how much this matters:
    only 6.4% of distinct Latin spellings (7,995 of 125,733) collide at
    all, and inspecting a sample of the close calls (runner-up within 2x of
    the winner's count) shows most are the SAME word under Bengali's own
    real orthographic inconsistency (ন/ণ, ং/ঙ spelling variance - e.g.
    "ankito" -> অঙ্কিত/অংকিত, not two different words), not genuine
    different-word ambiguity. A context-aware disambiguator (a bigram
    language model over resolved words, say) would be real engineering for
    a problem this small and mostly harmless; kept out of scope for that
    reason, not because it was not considered. The runner-up is exposed so
    a future pass has what it needs without rebuilding this table.
    """
    lex_train = load_lexicon_counts(os.path.join(dakshina_dir, "lexicons", "bn.translit.sampled.train.tsv"))
    lex_dev = load_lexicon_counts(os.path.join(dakshina_dir, "lexicons", "bn.translit.sampled.dev.tsv"))
    aligned = load_aligned_counts(os.path.join(dakshina_dir, "romanized", "bn.romanized.rejoined.aligned.tsv"))
    log(f"lexicon train+dev: {len(lex_train) + len(lex_dev)} real pairs")
    log(f"aligned natural sentences: {len(aligned)} real pairs")

    merged: Counter = Counter()
    for c in (lex_train, lex_dev, aligned):
        for k, v in c.items():
            merged[k] += v

    by_latin: dict[str, dict[str, int]] = defaultdict(dict)
    for (native, roman), cnt in merged.items():
        if not roman:
            continue
        by_latin[roman][native] = by_latin[roman].get(native, 0) + cnt

    table: dict[str, tuple[str, int, str, str]] = {}
    for latin, candidates in by_latin.items():
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1])
        native, cnt = ranked[0]
        runner_up = f"{ranked[1][0]}:{ranked[1][1]}" if len(ranked) > 1 else ""
        table[latin] = (native, cnt, "real", runner_up)
    log(f"merged real table: {len(table)} distinct Latin surface forms")
    return table


def augment_synthetic(table: dict[str, tuple[str, int, str, str]], corpus_lines: list[str],
                       variants_per_line: int = 2, seed: int = 0,
                       log=lambda m: None) -> None:
    """Fill gaps only: add synthetic (latin -> bengali_word) pairs for words
    with no real-data entry at all, mutating `table` in place. Never
    overwrites a real entry - real observations always win.
    """
    from bntok.banglish_synth import generate_pairs
    pairs = generate_pairs(corpus_lines, variants_per_line=variants_per_line, seed=seed)
    added = 0
    for latin, bengali_line in pairs:
        # generate_pairs operates on whole lines; only single-word lines
        # produce a usable word-level lookup entry here (multi-word lines
        # are for the training corpus, not this table).
        if " " in bengali_line.strip() or " " in latin.strip():
            continue
        if not latin or latin in table:
            continue
        table[latin] = (bengali_line, 0, "synthetic", "")
        added += 1
    log(f"synthetic gap-fill: +{added} entries (real entries never overwritten)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dakshina-dir", required=True, help="path to dakshina_dataset_v1.0/bn")
    p.add_argument("--out", required=True)
    p.add_argument("--synthetic-words", type=int, default=20000,
                    help="how many single-word lines to draw from our own corpus for gap-filling")
    args = p.parse_args(argv)

    log = lambda m: print(m, file=sys.stderr)
    table = build_table(args.dakshina_dir, log=log)

    # Gap-fill from our own corpus: split real Bengali lines into words,
    # dedup, so each synthetic-augmentation candidate is a single word (this
    # table is word-level), not a full sentence.
    from bntok.corpus import is_clean_bengali_line, stream_wikipedia
    from bntok.normalize import normalize
    log("streaming our own corpus for gap-fill candidate words ...")
    raw_lines = stream_wikipedia(lang="bn", limit=3000)
    words: set[str] = set()
    for line in raw_lines:
        if not is_clean_bengali_line(line):
            continue
        for w in normalize(line).split():
            if len(w) >= 2:
                words.add(w)
        if len(words) >= args.synthetic_words:
            break
    log(f"{len(words)} candidate words for gap-filling")
    augment_synthetic(table, list(words), log=log)

    real_n = sum(1 for v in table.values() if v[2] == "real")
    syn_n = len(table) - real_n
    ambiguous_n = sum(1 for v in table.values() if v[3])
    log(f"final table: {len(table)} entries ({real_n} real, {syn_n} synthetic, "
        f"{ambiguous_n} with a stored runner-up)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(f"{latin}\t{bengali}\t{cnt}\t{source}\t{runner_up}\n" for latin, (bengali, cnt, source, runner_up) in sorted(table.items()))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
