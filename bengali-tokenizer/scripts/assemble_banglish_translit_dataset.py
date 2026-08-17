r"""
Assemble the tier-3 transliteration model's training data: train/dev/test
splits plus a character vocabulary, ready to hand to Colab.

Splits, kept disjoint the same way the rest of this project keeps held-out
data disjoint from training:
  - train.tsv: Dakshina lexicon TRAIN split (real) + the aligned natural-
    sentence real pairs + large-scale synthetic augmentation from our own
    corpus (bntok.banglish_synth). This is the bulk of the data.
  - dev.tsv:   Dakshina lexicon DEV split (real) ONLY - no synthetic, so
    checkpoint selection during training is judged against real human
    spellings, not our own generator's approximations of them.
  - test.tsv:  NOT written by this script. Dakshina's lexicon TEST split
    (bn.translit.sampled.test.tsv) is used directly, once, after training -
    reserved untouched since the tier-1 lookup table was first built
    (scripts/build_banglish_lookup.py), for exactly this purpose.

Usage:
  python scripts/assemble_banglish_translit_dataset.py \
    --dakshina-dir <path to dakshina_dataset_v1.0/bn> \
    --out-dir artifacts/banglish-translit-data \
    --synthetic-words 300000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_valid_pair(latin: str, native: str) -> bool:
    """Word-level pair validity: source must be pure Latin letters (plus
    apostrophe, for contractions/loanwords), target must be pure Bengali-
    block codepoints (U+0980-U+09FF), nothing else.

    Caught by inspecting a first assembly run's vocab.json before ever
    starting a Colab training run, not assumed clean: Dakshina's real-
    Wikipedia-sourced data and this project's own corpus both legitimately
    contain foreign-script text (a real Wikipedia sentence can quote a
    place name in Devanagari, a word in Greek, an emoji), which
    `is_clean_bengali_line`'s line-level 75% ratio filter correctly lets
    through for LINES but a single contaminated WORD from that 25% budget
    still slipped into "native word" training pairs. An unfiltered first
    run measured a target vocabulary of 1,077 symbols spanning a dozen
    foreign scripts, emoji, and control characters - real noise, not
    signal, for a model whose only job is Latin word in, Bengali word out.
    """
    if not latin or not all(c.isascii() and (c.isalpha() or c == "'") for c in latin):
        return False
    return bool(native) and all(0x0980 <= ord(c) <= 0x09FF for c in native)


def load_lexicon_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            native, roman, _cnt = parts
            pairs.append((roman.lower(), native))
    return pairs


def load_aligned_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            native, roman = parts
            pairs.append((roman.lower(), native))
    return pairs


def build_synthetic_pairs(n_words: int, log=lambda m: None) -> list[tuple[str, str]]:
    """Draw real Bengali words from our own large corpus (not just the small
    sample used for the tier-1 lookup table's gap-fill) and generate noisy
    Latin renderings via bntok.banglish_synth, reusing the same validated
    (against Dakshina, see docs/known-issues.md) reverse phonetic table.
    """
    from bntok.banglish_synth import generate_pairs
    from bntok.corpus import is_clean_bengali_line, stream_sangraha, stream_wikipedia
    from bntok.normalize import normalize

    words: set[str] = set()
    log(f"streaming real corpus for up to {n_words} candidate words ...")
    for line in stream_wikipedia(lang="bn", limit=15000):
        if not is_clean_bengali_line(line):
            continue
        for w in normalize(line).split():
            if len(w) >= 2:
                words.add(w)
        if len(words) >= n_words:
            break
    if len(words) < n_words:
        try:
            for line in stream_sangraha("ben", doc_type="web", limit_docs=5000, max_files=2):
                if not is_clean_bengali_line(line):
                    continue
                for w in normalize(line).split():
                    if len(w) >= 2:
                        words.add(w)
                if len(words) >= n_words:
                    break
        except Exception as e:  # noqa: BLE001 - optional supplementary source
            log(f"  (sangraha supplement skipped: {e})")
    log(f"{len(words)} candidate words")
    pairs = generate_pairs(list(words), variants_per_line=2, seed=7)
    # generate_pairs works over whole lines; keep only single-word results.
    pairs = [(latin, bn) for latin, bn in pairs if " " not in latin and " " not in bn]
    log(f"{len(pairs)} synthetic pairs")
    return pairs


def build_char_vocab(pairs: list[tuple[str, str]]) -> dict:
    src_chars = set()
    tgt_chars = set()
    for latin, native in pairs:
        src_chars.update(latin)
        tgt_chars.update(native)
    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    src_vocab = specials + sorted(src_chars)
    tgt_vocab = specials + sorted(tgt_chars)
    return {
        "src_stoi": {c: i for i, c in enumerate(src_vocab)},
        "tgt_stoi": {c: i for i, c in enumerate(tgt_vocab)},
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }


def write_pairs(path: str, pairs: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{latin}\t{native}\n" for latin, native in pairs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dakshina-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--synthetic-words", type=int, default=300000)
    args = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)

    lex_train = load_lexicon_pairs(os.path.join(args.dakshina_dir, "lexicons", "bn.translit.sampled.train.tsv"))
    lex_dev = load_lexicon_pairs(os.path.join(args.dakshina_dir, "lexicons", "bn.translit.sampled.dev.tsv"))
    aligned = load_aligned_pairs(os.path.join(args.dakshina_dir, "romanized", "bn.romanized.rejoined.aligned.tsv"))
    log(f"lexicon train: {len(lex_train)}, lexicon dev: {len(lex_dev)}, aligned: {len(aligned)}")

    synthetic = build_synthetic_pairs(args.synthetic_words, log=log)

    train_pairs = lex_train + aligned + synthetic
    dev_pairs = lex_dev  # real only, deliberately no synthetic - see module docstring

    # dedup, keep first occurrence (real sources listed before synthetic in
    # train_pairs, so a synthetic duplicate of a real pair never overrides it)
    def dedup_and_filter(pairs):
        seen = set()
        out = []
        dropped = 0
        for latin, native in pairs:
            if not _is_valid_pair(latin, native) or (latin, native) in seen:
                if latin and native:
                    dropped += 1
                continue
            seen.add((latin, native))
            out.append((latin, native))
        return out, dropped

    n_train_raw, n_dev_raw = len(train_pairs), len(dev_pairs)
    train_pairs, train_dropped = dedup_and_filter(train_pairs)
    dev_pairs, dev_dropped = dedup_and_filter(dev_pairs)
    log(f"train: {n_train_raw} raw -> {len(train_pairs)} kept ({train_dropped} dropped: "
        f"non-Latin source, non-Bengali-block target, or duplicate)")
    log(f"dev:   {n_dev_raw} raw -> {len(dev_pairs)} kept ({dev_dropped} dropped)")
    log("(test: use Dakshina's own bn.translit.sampled.test.tsv directly, not copied here)")

    os.makedirs(args.out_dir, exist_ok=True)
    write_pairs(os.path.join(args.out_dir, "train.tsv"), train_pairs)
    write_pairs(os.path.join(args.out_dir, "dev.tsv"), dev_pairs)
    vocab = build_char_vocab(train_pairs + dev_pairs)
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    log(f"src vocab size: {len(vocab['src_vocab'])}, tgt vocab size: {len(vocab['tgt_vocab'])}")
    log(f"wrote {args.out_dir}/{{train.tsv, dev.tsv, vocab.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
