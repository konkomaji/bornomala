r"""
The akshara finite-state parser (v2 design step 3,
docs/design/reading-bengali-on-its-own-terms.md section 6-7,
docs/design/FORMAL_SPEC.md sections 2-4).

Where the shipped v1 tokenizer (tokenizer.py) discovers Bengali's structure
statistically (BPE merges over grapheme-cluster atoms), this module parses it
directly from Bengali's own generative grammar: the akshara (orthographic
syllable) is a consonant, optionally extended by further virama-joined
consonants (an unbounded conjunct), decorated along the way by nukta, matra,
and modifier marks - or, in the other branch, an independent vowel with the
same trailing decorations, minus the ability to chain into a further
consonant. That grammar is regular (only Kleene-star, no recursion), so it is
recognisable by a single left-to-right scan with no backtracking: O(n) in the
length of the input (FORMAL_SPEC section 4).

The grammar below went through three empirical passes against `regex`'s own
UAX #29 `\X`, not assumption:

  1. A first pass (ZWJ right after a virama continues a conjunct, ZWNJ right
     after one terminates it explicitly; Nukta can repeat) was checked
     against the design doc's simplified grammar and synthetic edge cases.
  2. Running the parser against real Wikipedia held-out text (v2 design
     step 4's own measurement) surfaced two real, previously-untested
     divergences: an independent vowel followed by a virama does NOT chain
     into a further consonant the way a consonant does (`\X` clusters
     "vowel+virama" as its own pair, then starts fresh) - Unicode's
     Indic_Conjunct_Break (InCB) rule requires InCB=Consonant on *both*
     sides of the virama, and vowels are not InCB=Consonant; and a Modifier
     (candrabindu/anusvara/visarga/sandhi mark) breaks chain eligibility
     even though Matra and Nukta do not.
  3. A second held-out run (after fixing pass 2) surfaced a third: ZWJ and
     ZWNJ are not tied to a fixed position relative to the virama the way
     pass 1 assumed - `regex` clusters "Consonant ZWJ Virama Consonant" as
     one continuing conjunct exactly like "Consonant Virama ZWJ Consonant",
     and ZWNJ blocks chain continuation the same way a Modifier does,
     wherever it falls in the run. The grammar below treats virama, nukta,
     matra, modifier, ZWJ, and ZWNJ as one mixed, order-agnostic run (see
     `_scan_tail`), tracking only *whether* a virama and a blocking
     character (Modifier or ZWNJ) occurred anywhere in it, not their
     positions - which is what actually matches `\X`, verified against
     every combination tried, not guessed at.

    Akshara := Consonant Tail
             | Vowel Tail

    Tail := a run mixing {Virama, Nukta, Matra, Modifier, ZWJ, ZWNJ} in any
            order or repetition; if that run contained at least one Virama,
            no Modifier or ZWNJ, the akshara started with a Consonant (never
            a Vowel), and a Consonant immediately follows, that Consonant is
            consumed too and the whole process repeats for its own tail
            (an unbounded conjunct chain) - otherwise the run's absorption
            already happened and the scan tries once more for further
            trailing decoration.

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

One deliberate, documented divergence from `\X` remains: khanda-ta (ৎ) is
treated as an ordinary consonant here (matching the design doc's own "39
consonants... ক through ৎ" listing), so `ৎ + virama + consonant` is parsed as
one continuing akshara. `\X` itself does not let khanda-ta chain this way (it
clusters `ৎ্` alone, the next consonant separately) - khanda-ta is not
supposed to take a conjunct in real orthography, so this sequence is
malformed either way, but the two are asserted to disagree on it explicitly
in tests/test_akshara.py rather than silently matching or silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as _re

from . import akshara_vec as _akshara_vec
from . import substrate
from .errors import NormalizationError

_GRAPHEME = _re.compile(r"\X")


@dataclass(frozen=True, slots=True)
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

    bounds, is_akshara = _scan(text)
    out: list[Akshara] = []
    start = 0
    for end, akshara in zip(bounds, is_akshara):
        out.append(Akshara(text[start:end], "akshara" if akshara else "other", start, end))
        start = end
    return out


def akshara_bounds(text: str) -> list[int]:
    """End offsets of each chunk `aksharas` would return, and nothing else.

    Chunk `i` spans `text[bounds[i - 1]:bounds[i]]` (with an implicit 0 before
    the first). Same scan, same grammar, same boundaries - it just skips
    building an `Akshara` object per chunk, which profiling showed to be ~38%
    of `aksharas`' total cost on real Bengali. For callers that only need the
    surface substrings (BMBT's `encode`, which never reads `kind`/`start`/
    `end`), this is the cheaper entry point; `aksharas` remains the full,
    object-returning API and is defined in terms of the same `_scan`, so the
    two can never disagree about where a boundary is.

    Raises:
      NormalizationError: if `text` is not a `str`. Never raises otherwise.
    """
    if not isinstance(text, str):
        raise NormalizationError(f"expected str, got {type(text).__name__}")
    return _scan(text)[0]


def akshara_bounds_batch(texts: list[str]) -> list[list[int]]:
    """`akshara_bounds` over many strings at once, vectorized where possible.

    Returns one list of end offsets per input string, always identical to
    `[akshara_bounds(t) for t in texts]`. The difference is only speed: when
    numpy is installed (`pip install "bntok[speed]"`) and every string falls
    inside the subset `akshara_vec` can prove equivalent to `\\X`, the whole
    batch is segmented with array operations instead of the per-character
    Python scan.

    Falls back silently and completely to the scalar scan when numpy is
    absent, or when the batch contains anything the vectorized model does not
    handle (other scripts' conjuncts, emoji ZWJ sequences, Hangul, CRLF).
    The fallback is correctness-preserving by construction: the guard is
    conservative, so an ineligible batch is never segmented by the fast path.

    Raises:
      NormalizationError: if any element is not a `str`.
    """
    for text in texts:
        if not isinstance(text, str):
            raise NormalizationError(f"expected str, got {type(text).__name__}")

    fast = _akshara_vec.bounds_batch(texts)
    if fast is None:  # numpy absent: whole batch takes the scalar scan
        return [_scan(text)[0] for text in texts]
    # Otherwise the batch was partitioned: `None` marks a string the fast path
    # declined, which the proven scalar scan answers instead.
    return [
        _scan(text)[0] if bounds is None else bounds
        for text, bounds in zip(texts, fast)
    ]


def _scan(text: str) -> tuple[list[int], list[bool]]:
    """The single left-to-right scan both public entry points share.

    Returns each chunk's end offset and whether it took an akshara branch
    (Consonant/Vowel) rather than the one-grapheme-cluster fallback. Locals
    are hoisted out of the loop deliberately: at one iteration per chunk over
    corpus-scale text, the module-attribute lookups these replace were
    measurable.
    """
    n = len(text)
    bounds: list[int] = []
    kinds: list[bool] = []
    push_bound = bounds.append
    push_kind = kinds.append
    consonants = substrate.CONSONANTS
    vowels = substrate.VOWELS
    grapheme_match = _GRAPHEME.match
    scan_tail = _scan_tail
    pos = 0

    while pos < n:
        ch = text[pos]

        if ch in consonants:
            pos = scan_tail(text, pos + 1, n, True)
            push_kind(True)

        elif ch in vowels:
            pos = scan_tail(text, pos + 1, n, False)
            push_kind(True)

        else:
            pos = grapheme_match(text, pos).end()
            push_kind(False)

        push_bound(pos)

    return bounds, kinds


# Characters absorbable into the current chunk purely as decoration, plus
# ZWJ/ZWNJ, which additionally interact with virama-chain continuation below.
# Built once at import time: `substrate.NUKTA` is a single codepoint, the
# rest are already frozenset-like collections.
_LINK_CHARS = substrate.MATRAS | substrate.MODIFIERS | {substrate.NUKTA, substrate.ZWJ, substrate.ZWNJ}

# Characters that, if seen anywhere since the last consonant, block a virama
# in the same run from continuing the conjunct chain (Unicode's
# Indic_Conjunct_Break rule: only Nukta/Matra/ZWJ are transparent to chain
# continuation; a Modifier or an explicit ZWNJ are not, verified empirically
# against `regex`'s own `\\X`, in either position relative to the virama).
_CHAIN_BLOCKERS = substrate.MODIFIERS | {substrate.ZWNJ}


def _scan_tail(text: str, pos: int, n: int, may_chain: bool) -> int:
    """Consume everything after an akshara's initial Consonant/Vowel.

    Repeatedly: greedily absorb a single mixed run of virama/nukta/matra/
    modifier/ZWJ/ZWNJ, in any order or repetition (all of these are generic
    Extend-class characters to `\\X`, so real and malformed text alike can
    interleave them in ways a rigid ordered grammar would miss - a modifier
    before its matra, a repeated matra, a ZWJ before rather than after a
    virama, all real cases found by running this parser against held-out
    Wikipedia text, not hypothetical). After that run, if it contained at
    least one virama, no `_CHAIN_BLOCKERS` character, `may_chain` is true
    (a Consonant start; Vowels never chain), and a Consonant immediately
    follows, consume that consonant too and repeat the whole process for
    its own tail. Otherwise the run's absorption already happened and the
    loop tries once more for further trailing decoration. Stops the moment
    an iteration makes no progress.
    """
    # Hoisted out of the loop: these are module/global lookups otherwise
    # repeated on every absorbed codepoint. Each `text[pos]` is also read
    # once into a local rather than re-indexed per test - indexing a
    # non-Latin-1 character allocates a fresh 1-character str in CPython,
    # so the repeated reads were real allocations, not just lookups.
    virama = substrate.VIRAMA
    link_chars = _LINK_CHARS
    chain_blockers = _CHAIN_BLOCKERS
    consonants = substrate.CONSONANTS

    while True:
        before = pos
        saw_virama = saw_blocker = False
        while pos < n:
            ch = text[pos]
            if ch == virama:
                saw_virama = True
            elif ch in link_chars:
                if ch in chain_blockers:
                    saw_blocker = True
            else:
                break
            pos += 1

        if may_chain and saw_virama and not saw_blocker and pos < n and text[pos] in consonants:
            pos += 1  # consume the chaining consonant; loop back for its own tail

        if pos == before:
            break
    return pos
