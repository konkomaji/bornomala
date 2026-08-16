r"""Vectorized backend for `akshara.py`'s segmentation (optional, needs numpy).

`akshara.py`'s `_scan` is a sequential left-to-right FSM: correct, proven, and
about 0.7-1.3 M codepoints/s in interpreted Python. This module computes the
identical boundaries with array operations instead, reaching roughly 15 M
codepoints/s on batched input - measured, see `tests/test_akshara_vec.py`.

WHY THIS IS SOUND
-----------------
The akshara grammar is regular (`akshara.py`'s module docstring), so the parse
state at any position is determined by two quantities that are *segmented
reductions* rather than a sequential dependency chain:

  * where the current run began - the most recent Consonant / Vowel / other
    character, i.e. a running maximum over reset indices;
  * whether a virama and/or a chain-blocker occurred since then - a difference
    of two prefix sums.

Both are O(n) over contiguous memory. A conjunct continuation is then a purely
local predicate over those two reductions, so no character-by-character
recurrence is needed at all.

An earlier version of this file instead computed a parallel prefix scan over
the transition monoid (compose f_c : State -> State pairwise, Hillis-Steele).
That is also correct - function composition is associative - but it is
O(n log n) in fancy-index gathers, which are not SIMD-friendly, and it measured
only 1.33x the scalar scan. The formulation below measured 19-23x. Recorded
here because the faster algorithm is the less obvious one.

WHY THE GUARD EXISTS
--------------------
`akshara.py` falls back to one UAX #29 grapheme cluster (`\X`) for anything
that is not a Bengali consonant or vowel. Reproducing all of `\X` in array
form would mean reimplementing regional-indicator pairing, Hangul jamo
composition, emoji ZWJ sequences, CRLF, and every other script's conjunct
rules - each an opportunity to be subtly wrong.

So this module does not try. `is_vectorizable` admits only a subset in which
the array model is provably equivalent to `\X`, and everything else falls back
to the scalar scan. Correctness therefore rests on the guard being
conservative, not on this file handling all of Unicode. A buffer that mixes
Bengali with Devanagari conjuncts, emoji, or Hangul takes the scalar path and
is still exactly right, just not faster.
"""

from __future__ import annotations

import unicodedata

# Character classes.
_C, _V, _R, _N, _M, _D, _J, _Z, _O, _E = range(10)

_BENGALI_START, _BENGALI_END = 0x0980, 0x0A00
_DANDA, _DOUBLE_DANDA = 0x0964, 0x0965

_TABLES: tuple | None = None


def available() -> bool:
    """True if the vectorized backend can be used (numpy importable)."""
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


def _tables():
    """Build (and cache) the class and safe-set lookup tables.

    Built lazily rather than at import time so that importing `bntok` stays
    cheap for callers that never segment anything, and so that a missing numpy
    is only an error for callers that actually asked for this backend.
    """
    global _TABLES
    if _TABLES is not None:
        return _TABLES

    import numpy as np

    from . import substrate

    cls = np.full(0x110000, _O, dtype=np.uint8)
    for ch in substrate.CONSONANTS:
        cls[ord(ch)] = _C
    for ch in substrate.VOWELS:
        cls[ord(ch)] = _V
    for ch in substrate.MATRAS:
        cls[ord(ch)] = _M
    for ch in substrate.MODIFIERS:
        cls[ord(ch)] = _D
    cls[ord(substrate.VIRAMA)] = _R
    cls[ord(substrate.NUKTA)] = _N
    cls[ord(substrate.ZWJ)] = _J
    cls[ord(substrate.ZWNJ)] = _Z

    # Safe subset: the array model is equivalent to `\X` only here.
    #   * the whole Bengali block;
    #   * printable ASCII, tab and newline - every one a singleton cluster.
    #     CR is deliberately EXCLUDED: `\r\n` is a single cluster to `\X`,
    #     which this model would split;
    #   * danda and double danda, which live in the Devanagari block but are
    #     Bengali's own sentence punctuation (see docs/bengali-script-reference.md).
    #   * Latin-1 Supplement, Latin Extended-A: letters and punctuation only,
    #     no combining marks in either range, so every one is a singleton;
    #   * General Punctuation: quotes, dashes and the invisible formatting
    #     characters. Every Cf in it other than ZWJ/ZWNJ (already classified
    #     above) is Grapheme_Cluster_Break=Control to `\X`, i.e. a singleton,
    #     which is what this model produces for them too;
    #   * currency symbols, and Devanagari's digits alongside its dandas.
    #     The rest of the Devanagari block is excluded: its virama drives
    #     conjunct rules this model does not implement.
    safe = np.zeros(0x110000, dtype=bool)
    safe[_BENGALI_START:_BENGALI_END] = True
    safe[0x20:0x7F] = True
    safe[0x09] = safe[0x0A] = True
    safe[0xA0:0x180] = True
    safe[0x2000:0x2070] = True
    safe[0x20A0:0x20C0] = True
    safe[_DANDA] = safe[_DOUBLE_DANDA] = True
    safe[0x0966:0x0970] = True

    # Any combining mark inside the safe subset that the substrate sets do not
    # already name (e.g. the AU length mark, the vocalic vowel signs) still
    # glues onto what precedes it, exactly as `\X` treats it. Checked against
    # unicodedata rather than assumed - the same discipline akshara.py's own
    # grammar was built with.
    for cp in np.flatnonzero(safe):
        if cls[cp] == _O and unicodedata.category(chr(int(cp))) in ("Mn", "Mc", "Me"):
            cls[cp] = _E

    marks = np.zeros(10, dtype=bool)
    marks[[_R, _N, _M, _D, _J, _Z, _E]] = True
    resets = np.zeros(10, dtype=bool)
    resets[[_C, _V, _O]] = True

    _TABLES = (cls, safe, marks, resets)
    return _TABLES


def _codes(text: str):
    import numpy as np

    return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)


def is_vectorizable(text: str) -> bool:
    """True if `text` lies entirely within the subset this backend handles."""
    if not text:
        return True
    _, safe, _, _ = _tables()
    return bool(safe[_codes(text)].all())


def segment_starts(text: str):
    """Chunk START offsets for `text`, as a numpy array.

    Caller must have checked `is_vectorizable(text)`; the result is undefined
    (not merely slower) otherwise.
    """
    return _starts_from_codes(_codes(text))


def _starts_from_codes(codes):
    """`segment_starts` on an already-encoded codepoint array.

    Split out so `bounds_batch` can encode the joined buffer once and reuse it
    for both the eligibility check and the segmentation itself.
    """
    import numpy as np

    cls_lut, _, marks_lut, resets_lut = _tables()
    n = len(codes)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    cls = cls_lut[codes]
    is_mark = marks_lut[cls]
    is_reset = resets_lut[cls]

    idx = np.arange(n)
    # Index of the most recent run-starting character at or before each position.
    last_reset = np.maximum.accumulate(np.where(is_reset, idx, -1))

    # cum[k] = occurrences strictly before position k.
    cum_vir = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(cls == _R, out=cum_vir[1:])
    cum_blk = np.zeros(n + 1, dtype=np.int32)
    np.cumsum((cls == _D) | (cls == _Z), out=cum_blk[1:])

    # For a consonant at i, the run it might continue began at a = last_reset[i-1].
    a = np.empty(n, dtype=np.int64)
    a[0] = -1
    a[1:] = last_reset[:-1]
    valid = a >= 0
    a_safe = np.where(valid, a, 0)
    after_a = np.minimum(a_safe + 1, n)

    owner_is_consonant = valid & (cls[a_safe] == _C)
    saw_virama = (cum_vir[idx] - cum_vir[after_a]) > 0
    saw_blocker = (cum_blk[idx] - cum_blk[after_a]) > 0

    chains = (cls == _C) & owner_is_consonant & saw_virama & ~saw_blocker

    starts = ~is_mark & ~chains
    starts[0] = True
    return np.flatnonzero(starts)


def bounds_batch(texts: list[str]) -> list[list[int] | None] | None:
    """Chunk END offsets per string, with None for any the fast path declines.

    Returns None outright only when numpy is missing. Otherwise the batch is
    PARTITIONED: eligible strings are segmented together in one pass, and each
    ineligible string yields None for `akshara.py` to fill in with the scalar
    scan.

    Partitioning rather than rejecting the whole batch is not a refinement, it
    is what makes this backend worth having. Measured on 6,000 held-out
    Wikipedia lines, 89.4% of lines are individually eligible but *zero* of the
    4096-line batches were eligible as a whole - a single Devanagari quotation
    anywhere in a block was enough to send all 4,096 lines down the slow path.
    An all-or-nothing guard is correct and useless.

    Eligible strings are joined with a newline before the single pass, because
    per-call array setup dominates on short inputs. A newline is a valid
    separator precisely because it always starts a chunk and never chains, so
    boundaries at the joins are unaffected.
    """
    if not available():
        return None
    if not texts:
        return []

    import numpy as np

    _, safe, _, _ = _tables()
    out: list[list[int] | None] = [None] * len(texts)

    # Encode the whole batch ONCE and derive eligibility from that single
    # array, rather than calling numpy per string: at 4096 short lines per
    # call, per-line setup was costing more than the segmentation itself.
    joined = "\n".join(texts)
    codes = _codes(joined)
    offending = np.flatnonzero(~safe[codes])

    lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    line_starts = np.zeros(len(texts), dtype=np.int64)
    if len(texts) > 1:
        np.cumsum(lengths[:-1] + 1, out=line_starts[1:])

    if offending.size == 0:
        # Everything is eligible: segment the buffer already encoded above.
        eligible = range(len(texts))
        starts = _starts_from_codes(codes)
    else:
        bad_lines = np.searchsorted(line_starts, offending, side="right") - 1
        keep = np.ones(len(texts), dtype=bool)
        keep[np.unique(bad_lines)] = False
        eligible = np.flatnonzero(keep).tolist()
        if not eligible:
            return out
        starts = segment_starts("\n".join(texts[i] for i in eligible))

    offset = 0
    for i in eligible:
        length = len(texts[i])
        if length == 0:
            out[i] = []
            offset += 1
            continue
        # Interior boundaries only: everything STRICTLY after this string's
        # own start. `side="right"` rather than `left` + 1 is load-bearing -
        # a string beginning with a combining mark glues onto the newline
        # separator, so there is no boundary at `offset` to skip past, and
        # the old form silently dropped a real boundary instead. The string's
        # own start needs no entry here because these are END offsets.
        lo = np.searchsorted(starts, offset, side="right")
        hi = np.searchsorted(starts, offset + length, side="left")
        local = starts[lo:hi] - offset
        out[i] = [*local.tolist(), length]
        offset += length + 1
    return out
