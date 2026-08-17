r"""
MorphScore evaluation: does our grammar-first structure segment Bengali words
along real morpheme boundaries better than the incumbents, not just tie on
raw fertility (BMBT vs bn-bpe-64k, see docs/known-issues.md)?

Uses catherinearnett/morphscore (Arnett, Hudspeth & O'Connor, ICML 2025
Tokenization Workshop, https://arxiv.org/abs/2507.06378), a public framework
scoring 70+ languages against Universal Dependencies morpheme boundaries.

Two things this script does NOT take on faith, verified before writing a
single line of eval code:

  1. Bengali is NOT in MorphScore's headline 70-language table (its own cutoff
     is >=100 items after filtering). The raw HF dataset has 102 ben_beng rows
     total, before MorphScore's own unique+stem==lemma filtering - likely
     under 100 survive. Report the actual N every time; treat this as a
     small-sample result, not a headline number.
  2. MorphScore ships no precomputed scores for SUTRA, Sarvam, or anyone else.
     Every row here, ours and external, is computed fresh in this run - there
     was no free comparison waiting.

Two real bugs caught before trusting any number, neither hypothetical:

1. MorphScore reconstructs predicted morpheme-boundary character offsets by
   summing decoded-token lengths and comparing them against the gold
   wordform's own character length. Our tokenizers recompose decomposed
   consonant+nukta sequences (ড়/ঢ়/য়) to their NFC-excluded singleton on
   normalize() - so an unnormalized UD wordform and its normalized token
   reconstruction differ in length even when they render identically,
   silently misaligning every boundary in words containing those letters.
   Fixed for OUR tokenizers by normalizing wordform/stem/lemma/preceding_part/
   following_part through the same bntok.normalize.normalize() our tokenizers
   use internally, before scoring.

   That fix is WRONG for external tokenizers, though, and applying it
   universally was a second bug, caught by checking the raw UD source data
   directly rather than trusting the first fix: WordPiece models (BanglaBERT,
   mBERT) were trained on the DECOMPOSED spelling this project's own UD
   source data ships in (verified: every raw wordform containing this letter
   uses U+09AF U+09BC, never the singleton U+09DF) - our singleton is not
   standard NFC (a documented Unicode composition exclusion), so no external
   tokenizer's vocabulary was built expecting it. Force-normalizing external
   tokenizers' input to our singleton form made every such word a fully
   out-of-vocabulary [UNK] for them - an artifact of this script's own
   preprocessing, not a real property of those tokenizers.

   Fixed properly: two separate CSVs, ours-normalized for our two tokenizers
   (matches what their encode() does internally regardless of input), the
   untouched raw UD text for every external tokenizer (matches what their own
   training data looked like).

2. MorphScore's own special-token filtering matches decoded tokens against
   `tokenizer.special_tokens_map` by STRING equality, which also strips a
   genuine `[UNK]` whenever a whole word is out-of-vocabulary - a fully-OOV
   word then decodes to zero non-special tokens, and MorphScore's own
   macro-average divides by that zero (a real upstream ZeroDivisionError, not
   a config mistake here). Handled per-tokenizer by detecting and excluding
   fully-OOV wordforms before scoring, logged and disclosed, not silently
   caught and mislabeled "unavailable".

MorphScore is not vendored (no license file in its repo, so no redistribution
right assumed) - this script clones it fresh into a cache directory, same
pattern as the Colab notebooks' own `git clone`.

Usage:
  python scripts/morphscore_eval.py --out benchmarks/morphscore.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok.normalize import normalize

MORPHSCORE_REPO = "https://github.com/catherinearnett/morphscore"
MORPHSCORE_REF = "v2"
_CACHE_DIR = os.environ.get(
    "BNTOK_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
)

OUR_MODELS = [
    ("bn-bpe-64k (Bornomala v1)", "artifacts/bn-bpe-64k", False),
    ("bmbt-64k (Bornomala BMBT)", "artifacts/bmbt-64k", True),
]

# Same external registry compare.py uses, so this table never drifts from the
# fertility one. Not every entry here will load (gated models, offline
# tokenizer classes) - reported honestly as unavailable, matching compare.py.
#
# Fourth field is MorphScore's `subword_prefix`: WordPiece tokenizers
# (BanglaBERT/ELECTRA, mBERT/BERT) mark CONTINUATION subwords with a leading
# '##', which the default '' (no stripping) never removes - every multi-token
# word's reconstructed length then includes literal '#' characters, never
# matches the gold wordform, and every sample gets excluded (both first came
# back as ZeroDivisionError before this was diagnosed and fixed - not
# genuinely "unavailable"). SentencePiece tokenizers (everyone else here)
# don't need one: their '▁' word-boundary marker is already stripped by
# AutoTokenizer's own single-id decode, verified empirically against the
# other 7 rows below, which all produced plausible non-zero N without it.
EXTERNAL_MODELS = [
    ("Sarvam-1 (Sarvam AI)", "sarvamai/sarvam-1", ""),
    ("SUTRA (TWO AI)", "TWO/sutra-mlt256-v2", ""),
    ("BrahmicTokenizer-131K (TSAI)", "theschoolofai/BrahmicTokenizer-131K", ""),
    ("Krutrim (Krutrim AI)", "krutrim-ai-labs/Krutrim-2-instruct", ""),
    ("IndicBERTv2 (AI4Bharat)", "ai4bharat/IndicBERTv2-MLM-only", ""),
    ("BanglaBERT (csebuetnlp)", "csebuetnlp/banglabert", "##"),
    ("BanglaT5 (csebuetnlp)", "csebuetnlp/banglat5", ""),
    ("mBERT (Google)", "google-bert/bert-base-multilingual-cased", "##"),
    ("XLM-RoBERTa (Meta)", "FacebookAI/xlm-roberta-base", ""),
]


def ensure_morphscore(log=lambda m: None) -> str:
    """Clone (or reuse a cached clone of) the morphscore repo, return its PARENT
    dir. The repo itself is a flat `morphscore.py` (no package structure, no
    __init__.py) - cloning it into a directory literally named `morphscore`
    and putting that directory's PARENT on sys.path makes it resolve as a
    Python 3 implicit namespace package, matching the README's own
    `from morphscore.morphscore import MorphScore` usage.
    """
    dest = os.path.join(_CACHE_DIR, "morphscore")
    if os.path.isfile(os.path.join(dest, "morphscore.py")):
        log(f"reusing cached morphscore clone at {dest}")
        return _CACHE_DIR
    os.makedirs(_CACHE_DIR, exist_ok=True)
    log(f"cloning {MORPHSCORE_REPO}@{MORPHSCORE_REF} into {dest} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", MORPHSCORE_REF, MORPHSCORE_REPO, dest],
        check=True,
    )
    return _CACHE_DIR


def build_ben_beng_csvs(log=lambda m: None) -> tuple[str, str, int]:
    """Pull ben_beng rows from the HF dataset once, write TWO versions of
    MorphScore's expected ben_beng_data.csv: raw (untouched UD text, for
    every external tokenizer) and normalized (through the same
    bntok.normalize.normalize() our own tokenizers apply internally, for
    ours only - see module docstring, bug 1). Returns
    (raw_data_dir, normalized_data_dir, raw_row_count).
    """
    from datasets import load_dataset

    ds = load_dataset("catherinearnett/morphscore", split="train")
    ds = ds.filter(lambda r: r["language"] == "ben_beng")
    raw_n = len(ds)
    log(f"ben_beng: {raw_n} raw rows from the HF dataset (pre MorphScore filtering)")

    raw_dir = os.path.join(_CACHE_DIR, "morphscore-data-raw")
    os.makedirs(raw_dir, exist_ok=True)
    ds.to_pandas().to_csv(os.path.join(raw_dir, "ben_beng_data.csv"), index=False)

    def norm_field(v):
        if not isinstance(v, str):
            return v
        try:
            return normalize(v)
        except Exception:  # noqa: BLE001 - leave unparseable fields as-is, MorphScore will drop bad rows
            return v

    text_cols = {"wordform", "stem", "lemma", "preceding_part", "following_part"}
    norm_ds = ds.map(lambda r: {c: norm_field(r[c]) for c in text_cols if c in r})
    norm_dir = os.path.join(_CACHE_DIR, "morphscore-data-normalized")
    os.makedirs(norm_dir, exist_ok=True)
    norm_ds.to_pandas().to_csv(os.path.join(norm_dir, "ben_beng_data.csv"), index=False)

    return raw_dir, norm_dir, raw_n


class _OurTokenizerAdapter:
    """Minimal HF-AutoTokenizer-shaped wrapper so MorphScore.get_morphscore can
    call our tokenizers unmodified: it needs `tokenizer(word)['input_ids']`,
    `tokenizer.decode(single_id)`, and `.special_tokens_map`.

    Reuses the real per-token `decode([id])` for each id (not the human-
    readable `encode_tokens()`), since decode() already strips the Metaspace
    word-boundary marker per call - verified empirically to reconstruct the
    exact wordform when ids are concatenated (see module docstring).
    """

    def __init__(self, real_tokenizer):
        self._t = real_tokenizer
        self.special_tokens_map: dict = {}

    def __call__(self, word: str) -> dict:
        return {"input_ids": self._t.encode(word)}

    def decode(self, token_id) -> str:
        return self._t.decode([token_id])


def load_our_tokenizer(path: str, is_bmbt: bool):
    if is_bmbt:
        from bntok.bmbt import BMBT
        return BMBT.load(path)
    from bntok.tokenizer import BengaliTokenizer
    return BengaliTokenizer.load(path)


def _fully_oov_wordforms(tokenizer, csv_path: str) -> set:
    """Real upstream MorphScore bug (verified by direct debugging, not
    guessed): it filters decoded tokens against `tokenizer.special_tokens_map`
    by STRING equality, which also strips a genuine `[UNK]` token whenever a
    whole word is out-of-vocabulary - a fully-OOV word then decodes to zero
    non-special tokens, and MorphScore's own macro-average divides by that
    zero (ZeroDivisionError), rather than scoring it as a real (very bad)
    single-token result. Pre-identify and exclude such words per tokenizer,
    documented, rather than silently mislabel the tokenizer "unavailable".
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    special = set(tokenizer.special_tokens_map.values()) if hasattr(tokenizer, "special_tokens_map") else set()
    oov = set()
    for w in df["wordform"].dropna().unique():
        ids = tokenizer(str(w))["input_ids"]
        toks = [tokenizer.decode(i) for i in ids]
        if all(t in special for t in toks):
            oov.add(w)
    return oov


def run_morphscore(morphscore_dir: str, data_dir: str, tokenizer, label: str, log=lambda m: None,
                    subword_prefix: str = "") -> dict:
    sys.path.insert(0, morphscore_dir)
    from morphscore.morphscore import MorphScore

    def _make(scorer_data_dir):
        return MorphScore(
            data_dir=scorer_data_dir,
            language_subset=["ben_beng"],
            unique_only=True,
            stem_eq_lemma=True,
            exclude_numbers=True,
            freq_scale=False,
            exclude_single_tok=True,
            exclude_single_morpheme=True,
            subword_prefix=subword_prefix,
        )

    try:
        results = _make(data_dir).eval(tokenizer)
    except ZeroDivisionError:
        import pandas as pd
        csv_path = os.path.join(data_dir, "ben_beng_data.csv")
        oov = _fully_oov_wordforms(tokenizer, csv_path)
        log(f"{label}: hit MorphScore's fully-OOV divide-by-zero bug, "
            f"excluding {len(oov)} fully-OOV wordform(s): {sorted(oov)}")
        df = pd.read_csv(csv_path)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        filtered_dir = os.path.join(_CACHE_DIR, f"morphscore-data-filtered-{safe_label}")
        os.makedirs(filtered_dir, exist_ok=True)
        df[~df["wordform"].isin(oov)].to_csv(os.path.join(filtered_dir, "ben_beng_data.csv"), index=False)
        results = _make(filtered_dir).eval(tokenizer)

    row = results.get("ben_beng", {})
    log(f"{label}: recall={row.get('morphscore_recall')} precision={row.get('morphscore_precision')} "
        f"n={row.get('num_samples')}")
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmarks/morphscore.json")
    p.add_argument("--skip-external", action="store_true", help="only score our own two tokenizers")
    args = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)

    morphscore_dir = ensure_morphscore(log=log)
    raw_dir, norm_dir, raw_n = build_ben_beng_csvs(log=log)

    rows = {}
    for label, path, is_bmbt in OUR_MODELS:
        try:
            tok = load_our_tokenizer(path, is_bmbt)
            adapter = _OurTokenizerAdapter(tok)
            rows[label] = run_morphscore(morphscore_dir, norm_dir, adapter, label, log=log)
        except Exception as e:  # noqa: BLE001 - report ANY failure honestly, not silently
            log(f"{label}: FAILED - {type(e).__name__}: {e}")
            rows[label] = {"error": str(e)}

    if not args.skip_external:
        from transformers import AutoTokenizer
        for label, repo, subword_prefix in EXTERNAL_MODELS:
            try:
                tk = AutoTokenizer.from_pretrained(repo, use_fast=True)
                rows[label] = run_morphscore(morphscore_dir, raw_dir, tk, label, log=log,
                                              subword_prefix=subword_prefix)
            except Exception as e:  # noqa: BLE001 - unavailable, not estimated
                log(f"{label}: unavailable - {type(e).__name__}: {e}")
                rows[label] = {"error": f"unavailable: {e}"}

    out = {
        "raw_ben_beng_rows_before_morphscore_filtering": raw_n,
        "note": "MorphScore's own README cutoff for its headline 70-language table is "
                ">=100 items after filtering; Bengali is below that bar and is not in "
                "that table. Treat num_samples below as the real, small N.",
        "results": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
