r"""
The akshara finite-state parser (v2 design roadmap step 3,
docs/design/reading-bengali-on-its-own-terms.md section 6-7,
docs/design/FORMAL_SPEC.md sections 2-4).

Where the shipped v1 tokenizer (tokenizer.py) discovers Bengali's structure
statistically (BPE merges over grapheme-cluster atoms), this module parses it
directly from Bengali's own generative grammar: the akshara (orthographic
syllable) is a consonant, optionally extended by further virama-joined
consonants (an unbounded conjunct), followed by an optional vowel sign and
zero or more modifiers - or, in the other branch, an independent vowel
followed by modifiers. That grammar is regular (only Kleene-star, no
recursion), so it is recognisable by a single left-to-right scan with no
backtracking: O(n) in the length of the input (FORMAL_SPEC section 4).

The grammar implemented here is refined from the design doc's simplified
version after checking it against `regex`'s own UAX #29 `\X` behaviour on the
actual edge cases (ZWJ continues a conjunct, ZWNJ terminates it explicitly;
Nukta can repeat; a trailing virama needs its own slot):

    Akshara := Consonant Nukta*
                 (Virama ZWJ? Consonant Nukta*)*
                 (Virama ZWNJ?)?
                 Matra?
                 Modifier*
             |  Vowel Modifier*

Anything that does not start a Consonant or Vowel branch (foreign scripts,
ASCII, digits, punctuation, an orphan/leading virama, an isolated matra, an
unrecognised combining mark, emoji, whitespace) falls back to exactly one
UAX #29 grapheme cluster (via the same `\X` machinery graphemes.py uses),
never one raw codepoint and never a run of several clusters. This keeps one
structural invariant true uniformly for every returned chunk, "akshara" or
"other": it is always a whole number of grapheme clusters, never a partial
one.

Guarantees (FORMAL_SPEC sections 2-3), all achieved by construction, not by
special-casing:
  * SEG / LOSSLESS - `aksharas` only ever cuts `text`, never rewrites it, so
    concatenating every chunk's `.text` in order reproduces `text` exactly.
  * TOTAL - every branch of the scan advances `pos` by at least one
    codepoint (the fallback always consumes at least one grapheme cluster,
    which is always at least one codepoint), so the scan always terminates
    with the whole input covered. `aksharas` never raises for any `str`
    input; it raises `errors.NormalizationError` only if `text` is not a
    `str` at all, mirroring `normalize.py`'s own guard.
  * DET - no randomness, no I/O, a fixed left-to-right scan: the same input
    always produces the same output.

`aksharas` does NOT normalise its input. Matching the rest of this package's
pipeline convention (`normalize()` is a separate, explicit stage callers run
first), it segments whatever string it is given; pre-normalisation and
post-normalisation calls can legitimately produce different chunk boundaries
on non-canonical input; see tests/test_akshara.py class 2.

One deliberate, documented divergence from `\X`: khanda-ta (ৎ) is treated as
an ordinary consonant here (matching the design doc's own "39 consonants...
ক through ৎ" listing), so `ৎ + virama + consonant` is parsed as one
continuing akshara. `\X` itself does not let khanda-ta chain this way (it
clusters `ৎ্` alone, the next consonant separately) - khanda-ta is not
supposed to take a conjunct in real orthography, so this sequence is
malformed either way, but the two are asserted to disagree on it explicitly
in tests/test_akshara.py rather than silently matching or silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as _re

from . import substrate
from .errors import NormalizationError

_GRAPHEME = _re.compile(r"\X")


@dataclass(frozen=True)
class Akshara:
    """One segmented chunk: an akshara, or one grapheme cluster of anything else.

    `text` is the exact surface substring (segmentation never rewrites), and
    `start`/`end` are codepoint offsets into the original input string, so
    `text == original[start:end]` always holds - carried directly on the
    chunk rather than reconstructed afterwards from cumulative lengths,
    which is a mistake this project has already made once (see
    docs/known-issues.md point 8).
    """

    text: str
    kind: str  # "akshara" | "other"
    start: int
    end: int


def aksharas(text: str) -> list[Akshara]:
    """Segment `text` into akshara and "other" chunks. Total, deterministic.

    Raises:
      NormalizationError: if `text` is not a `str`. Never raises otherwise.
    """
    if not isinstance(text, str):
        raise NormalizationError(f"expected str, got {type(text).__name__}")

    n = len(text)
    out: list[Akshara] = []
    pos = 0
    while pos < n:
        start = pos
        ch = text[pos]

        if ch in substrate.CONSONANTS:
            pos += 1
            pos = _consume_nukta(text, pos, n)
            pos = _consume_conjunct_tail(text, pos, n)
            pos = _consume_trailing_virama(text, pos, n)
            pos = _consume_matra(text, pos, n)
            pos = _consume_modifiers(text, pos, n)
            out.append(Akshara(text[start:pos], "akshara", start, pos))

        elif ch in substrate.VOWELS:
            pos += 1
            pos = _consume_modifiers(text, pos, n)
            out.append(Akshara(text[start:pos], "akshara", start, pos))

        else:
            end = _GRAPHEME.match(text, pos).end()
            out.append(Akshara(text[start:end], "other", start, end))
            pos = end

    return out


def _consume_nukta(text: str, pos: int, n: int) -> int:
    while pos < n and text[pos] == substrate.NUKTA:
        pos += 1
    return pos


def _consume_conjunct_tail(text: str, pos: int, n: int) -> int:
    """(Virama ZWJ? Consonant Nukta*)* - greedily extend an unbounded conjunct."""
    while pos < n and text[pos] == substrate.VIRAMA:
        lookahead = pos + 1
        if lookahead < n and text[lookahead] == substrate.ZWJ:
            lookahead += 1
        if lookahead < n and text[lookahead] in substrate.CONSONANTS:
            pos = _consume_nukta(text, lookahead + 1, n)
        else:
            break  # this virama does not continue a conjunct; leave it for the trailing slot
    return pos


def _consume_trailing_virama(text: str, pos: int, n: int) -> int:
    """(Virama ZWNJ?)? - an explicit/dangling virama that ends the conjunct here."""
    if pos < n and text[pos] == substrate.VIRAMA:
        pos += 1
        if pos < n and text[pos] == substrate.ZWNJ:
            pos += 1
    return pos


def _consume_matra(text: str, pos: int, n: int) -> int:
    if pos < n and text[pos] in substrate.MATRAS:
        pos += 1
    return pos


def _consume_modifiers(text: str, pos: int, n: int) -> int:
    while pos < n and text[pos] in substrate.MODIFIERS:
        pos += 1
    return pos
