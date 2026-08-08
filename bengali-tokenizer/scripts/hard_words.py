"""Encode a fixed set of hard conjunct words and Bengali place names on
every tokenizer this project already tracks (scripts/compare.py's own
HF_MODELS/TIKTOKEN_MODELS, imported directly so this can never drift out
of sync with the register benchmark's own model list), plus ours (v1,
BMBT). Writes benchmarks/hard-words.md.

Not a held-out corpus measurement like compare.py - a small, fixed,
qualitative word list chosen for cultural resonance (deity names, a
national poet, well-known places) and conjunct density, the kind of
words a general benchmark's aggregate fertility number can hide. Every
number here is a real encode() call against the real public tokenizer,
same honesty standard as the rest of this benchmarks/ directory: nothing
estimated, unavailable models reported as unavailable, not faked.

Usage: python scripts/hard_words.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok import BengaliTokenizer
from bntok.bmbt import BMBT
from scripts.compare import HF_MODELS, TIKTOKEN_MODELS

WORDS = [
    ("স্ত্রী", "wife/woman - classic hard conjunct"),
    ("আকাঙ্ক্ষা", "aspiration - triple conjunct ঙ্ক্ষ"),
    ("ঋত্বিক", "name - ত্ব conjunct"),
    ("বিজ্ঞান", "science - জ্ঞ conjunct"),
    ("স্বাধীনতা", "independence - স্ব conjunct"),
    ("কৃষ্ণ", "Krishna - ষ্ণ conjunct"),
    ("রবীন্দ্রনাথ", "Rabindranath (Tagore) - ন্দ্র conjunct"),
    ("পশ্চিমবঙ্গ", "West Bengal - শ্চ conjunct"),
    ("বর্ধমান", "Bardhaman - র্ধ reph conjunct"),
    ("মুর্শিদাবাদ", "Murshidabad - র্শ reph conjunct"),
    ("বিষ্ণুপুর", "Bishnupur - ষ্ণ conjunct"),
    ("শান্তিনিকেতন", "Santiniketan - ন্ত conjunct"),
    ("দার্জিলিং", "Darjeeling - র্জ reph conjunct"),
]


def main() -> None:
    print("loading ours...", file=sys.stderr)
    tok_v1 = BengaliTokenizer.load("artifacts/bn-bpe-64k")
    tok_bmbt = BMBT.load("artifacts/bmbt-64k")

    results: dict[str, dict[str, int | None]] = {w: {} for w, _ in WORDS}
    for w, _ in WORDS:
        results[w]["Bornomala v1"] = len(tok_v1.encode(w))
        results[w]["Bornomala BMBT"] = len(tok_bmbt.encode(w))

    for name, enc_name in TIKTOKEN_MODELS:
        print(f"loading {name}...", file=sys.stderr)
        try:
            import tiktoken
            enc = tiktoken.get_encoding(enc_name)
            for w, _ in WORDS:
                results[w][name] = len(enc.encode(w))
        except Exception as e:  # noqa: BLE001 - report ANY load failure as "unavailable", not estimated
            print(f"  FAILED: {e}", file=sys.stderr)
            for w, _ in WORDS:
                results[w][name] = None

    for name, hf_id in HF_MODELS:
        print(f"loading {name} ({hf_id})...", file=sys.stderr)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            for w, _ in WORDS:
                results[w][name] = len(tok.encode(w, add_special_tokens=False))
        except Exception as e:  # noqa: BLE001 - report ANY load failure as "unavailable", not estimated
            print(f"  UNAVAILABLE: {type(e).__name__}: {e}", file=sys.stderr)
            for w, _ in WORDS:
                results[w][name] = None

    all_models = (
        ["Bornomala v1", "Bornomala BMBT"]
        + [n for n, _ in TIKTOKEN_MODELS]
        + [n for n, _ in HF_MODELS]
    )

    lines = [
        "# Hard words: conjuncts and Bengali place names",
        "",
        (
            "Not a held-out corpus average - a fixed, small, qualitative word list"
            " (deity names, a national poet, well-known West Bengal places), chosen"
            " for cultural resonance and conjunct density. Regenerate with"
            " `python scripts/hard_words.py` from `bengali-tokenizer/`."
        ),
        "",
        "| Word | Note | " + " | ".join(all_models) + " |",
        "|---|---" + "|--:" * len(all_models) + "|",
    ]
    for w, note in WORDS:
        cells = [str(results[w].get(m, "?")) if results[w].get(m) is not None else "unavailable" for m in all_models]
        lines.append(f"| {w} | {note} | " + " | ".join(cells) + " |")

    lines += ["", "## Averages (available models only)", "", "| Tokenizer | Avg tokens/word |", "|---|--:|"]
    for m in all_models:
        vals = [results[w][m] for w, _ in WORDS if results[w].get(m) is not None]
        if vals:
            lines.append(f"| {m} | {sum(vals) / len(vals):.2f} |")
        else:
            lines.append(f"| {m} | unavailable |")

    out = "\n".join(lines) + "\n"
    with open("benchmarks/hard-words.md", "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote benchmarks/hard-words.md", file=sys.stderr)


if __name__ == "__main__":
    main()
