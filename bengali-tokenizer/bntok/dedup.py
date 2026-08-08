r"""
Corpus dedup and quality filtering (Track A2, spec section 16.2 Gate G3).

Gate G3 asks one question with a real number as the answer: after dedup and
quality filtering, what fraction of raw Bengali text survives, and does the
surviving corpus clear 5B clean tokens? This module builds the pipeline that
question needs. It operates on lines (this project's existing corpus unit
throughout `corpus.py`), in three stages, each independently testable and
each reporting how much it removed:

  1. **Exact dedup** - a set-based pass, catches byte-identical repeats
     (a common symptom of overlapping scrape sources or repeated
     boilerplate like navigation text).
  2. **Near dedup** - MinHash LSH over word 5-gram shingles (`datasketch`),
     catches near-identical lines exact dedup misses (templated
     boilerplate with a changing timestamp or id, minor re-punctuation).
  3. **Quality filter** - rule-based rejection of lines that are not
     usable Bengali prose: too short, too little actual Bengali/ASCII
     content (reuses `corpus.is_clean_bengali_line`'s heuristic), or
     dominated by digits/repeated characters (spam and junk headers).

No LM-perplexity filtering (the classic CCNet third stage): the `kenlm`
PyPI wheel ships query-only, with no `lmplz` trainer, so there is no way
to train a Bengali ARPA model from this environment without building
kenlm from source. Documented, not silently worked around - see
`docs/known-issues.md`. Rule-based quality filtering alone is a real,
precedented pipeline design (Sangraha and CCNet both use non-LM rule
filters as one of their stages, not only LM perplexity), not a
placeholder.
"""

from __future__ import annotations

import re
from collections import Counter

from .corpus import is_clean_bengali_line
from .errors import ConfigError


def exact_dedup(lines: list[str]) -> tuple[list[str], int]:
    """Remove byte-identical repeats, keeping first occurrence order.

    Returns (surviving lines, number removed).
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out, len(lines) - len(out)


def _shingles(line: str, n: int = 5) -> set[str]:
    """Word n-gram shingles for MinHash. Falls back to the whole line for
    text shorter than `n` words, so short lines still get a (weak, single-
    shingle) signature instead of being silently skipped by near-dedup."""
    words = line.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def near_dedup(
    lines: list[str], threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 5,
) -> tuple[list[str], int]:
    """Remove near-duplicate lines via MinHash LSH over word shingles.

    `threshold` is the Jaccard similarity above which two lines are treated
    as duplicates (datasketch's own LSH parameter). Keeps first occurrence
    order, same contract as `exact_dedup`. Meant to run AFTER `exact_dedup`
    (cheaper exact pass first; MinHash is the expensive stage). Returns
    (surviving lines, number removed).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as e:
        raise ConfigError("near_dedup needs datasketch: pip install datasketch") from e
    if not 0.0 < threshold <= 1.0:
        raise ConfigError(f"threshold must be in (0, 1], got {threshold}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    out: list[str] = []
    removed = 0
    for i, line in enumerate(lines):
        shingles = _shingles(line, shingle_size)
        if not shingles:
            out.append(line)
            continue
        mh = MinHash(num_perm=num_perm)
        for s in shingles:
            mh.update(s.encode("utf-8"))
        if lsh.query(mh):
            removed += 1
            continue
        lsh.insert(str(i), mh)
        out.append(line)
    return out, removed


_DIGIT = re.compile(r"[0-9০-৯]")


def _is_low_quality(line: str, max_digit_ratio: float = 0.5, max_char_repeat_ratio: float = 0.4) -> bool:
    """Rule-based junk detection beyond `is_clean_bengali_line`'s
    Bengali/ASCII ratio check: rejects digit-dominated lines (page numbers,
    ID dumps, tables rendered as text) and lines dominated by one repeated
    character (a common scrape/OCR artefact: rules, separators, redacted
    runs)."""
    if not line:
        return True
    digit_ratio = len(_DIGIT.findall(line)) / len(line)
    if digit_ratio > max_digit_ratio:
        return True
    counts = Counter(line)
    most_common_ratio = counts.most_common(1)[0][1] / len(line)
    return most_common_ratio > max_char_repeat_ratio


def quality_filter(lines: list[str]) -> tuple[list[str], int]:
    """Rule-based quality filter: `is_clean_bengali_line` plus digit- and
    repeated-character-dominance rejection. Returns (surviving lines,
    number removed)."""
    out = [ln for ln in lines if is_clean_bengali_line(ln) and not _is_low_quality(ln)]
    return out, len(lines) - len(out)


def survival_report(raw_lines: list[str], near_dedup_threshold: float = 0.8) -> dict:
    """Run the full pipeline (exact dedup -> near dedup -> quality filter)
    and report a survival ratio at each stage, in both line and
    whitespace-word (a fertility-independent token proxy) counts.

    This is the number Gate G3 asks for. Whitespace words are used as the
    token proxy rather than a trained tokenizer's own subword count,
    deliberately: the point of this measurement is corpus quality before
    tokenizer training, not after, so it must not depend on a specific
    tokenizer's vocabulary.
    """
    if not raw_lines:
        raise ConfigError("raw_lines is empty")

    raw_words = sum(len(ln.split()) for ln in raw_lines)

    after_exact, removed_exact = exact_dedup(raw_lines)
    after_near, removed_near = near_dedup(after_exact, threshold=near_dedup_threshold)
    after_quality, removed_quality = quality_filter(after_near)

    final_words = sum(len(ln.split()) for ln in after_quality)

    return {
        "raw_lines": len(raw_lines),
        "raw_words": raw_words,
        "removed_exact_dup": removed_exact,
        "removed_near_dup": removed_near,
        "removed_low_quality": removed_quality,
        "surviving_lines": len(after_quality),
        "surviving_words": final_words,
        "survival_ratio_lines": len(after_quality) / len(raw_lines),
        "survival_ratio_words": final_words / raw_words if raw_words else 0.0,
    }
