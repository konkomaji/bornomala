r"""
Grapheme cluster segmentation and Bengali script structure (Track A core).

The single most important unit in this tokenizer is the grapheme cluster, not the
codepoint (spec section 3.1.1). A Bengali grapheme cluster is what a reader
perceives as one written unit: a base consonant or vowel, plus any conjunct
(consonant + virama + consonant), reph, phalas, vowel sign (matra), nukta, and
sign (candrabindu / anusvara / visarga). In codepoints it can be 1 to roughly 8
long; to a reader it is one symbol.

We segment with UAX #29 extended grapheme clusters via the `regex` module's \X,
which handles Bengali conjuncts and reordered marks correctly, unlike iterating
over `str` (codepoints).

This module also exposes structural predicates (is this cluster a conjunct? does
it carry a reph? a phala?) that the evaluator uses to prove the headline
property: a trained tokenizer must never split a grapheme cluster across token
boundaries (conjunct fragmentation rate = 0), spec section 9.2 step 4.
"""

from __future__ import annotations

import regex as _re

_GRAPHEME = _re.compile(r"\X")

# --- Bengali codepoints (U+0980 block) -----------------------------------
VIRAMA = "্"        # hasanta, triggers conjunct formation
NUKTA = "়"
RA = "র"            # র, the consonant that forms reph / ra-phala
YA = "য"            # য, forms ya-phala
ZWJ = "‍"
ZWNJ = "‌"

# Consonants (main range plus the nukta-formed rha/yya letters).
_CONSONANTS = set(chr(c) for c in range(0x0995, 0x09B9 + 1)) | {"ড়", "ঢ়", "য়", "ৎ"}
# Independent vowels.
_IND_VOWELS = set(chr(c) for c in range(0x0985, 0x098C + 1)) | {"এ", "ঐ", "ও", "ঔ"}
# Dependent vowel signs (matra / kar).
_VOWEL_SIGNS = set(chr(c) for c in range(0x09BE, 0x09CC + 1)) | {"ৗ"}
# Signs.
_SIGNS = {"ঁ", "ং", "ঃ"}  # candrabindu, anusvara, visarga

# Bengali's own sentence- and verse-terminating punctuation (দাঁড়ি / দ্বিদাঁড়ি) is
# NOT in the Bengali Unicode block: it is the shared, script-neutral DANDA and
# DOUBLE DANDA (U+0964/U+0965), a Devanagari-block codepoint pair every Indic
# script that uses this punctuation reuses rather than re-encoding. Without
# these, the single most common punctuation mark in real Bengali text would
# only get an atom by corpus frequency, not by guarantee. (There is also a
# widespread, non-standard practice, especially in older/OCR'd text, of
# reusing U+09F7 BENGALI CURRENCY NUMERATOR FOUR as a visual danda substitute
# on fonts/typewriters that lacked the real one; it is covered anyway since
# it is inside BENGALI_BLOCK.)
DANDA = "।"
DOUBLE_DANDA = "॥"
SHARED_INDIC_PUNCTUATION = frozenset({DANDA, DOUBLE_DANDA})

# Full coverage sets. The tokenizer guarantees an atom for every codepoint in the
# Bengali block, Bengali's shared (non-Bengali-block) punctuation, and ASCII
# printable characters, regardless of what the induction corpus happened to
# contain. This makes encode/decode round-trip for any Bengali or code-mixed
# English text, not only text resembling the corpus.
BENGALI_BLOCK = frozenset(chr(c) for c in range(0x0980, 0x09FF + 1))
ASCII_PRINTABLE = frozenset(chr(c) for c in range(0x20, 0x7E + 1))
GUARANTEED_CODEPOINTS = BENGALI_BLOCK | SHARED_INDIC_PUNCTUATION | ASCII_PRINTABLE


def grapheme_clusters(text: str) -> list[str]:
    """Segment text into UAX #29 extended grapheme clusters."""
    return _GRAPHEME.findall(text)


def num_clusters(text: str) -> int:
    return len(_GRAPHEME.findall(text))


# --- structural predicates (used by the evaluator and analysis) ----------

def is_conjunct(cluster: str) -> bool:
    """True if the cluster contains a virama-joined consonant sequence (juktakkhor)."""
    return VIRAMA in cluster and any(c in _CONSONANTS for c in cluster)


def has_reph(cluster: str) -> bool:
    """True if the cluster begins with ra + virama (reph), reordered above the base."""
    return cluster.startswith(RA + VIRAMA)


def has_ya_phala(cluster: str) -> bool:
    return (VIRAMA + YA) in cluster


def has_ra_phala(cluster: str) -> bool:
    # ra-phala is a subjoined ra: base + virama + ra, where ra is not the first consonant.
    return (VIRAMA + RA) in cluster and not cluster.startswith(RA)


def has_nukta(cluster: str) -> bool:
    return NUKTA in cluster


def is_bengali(cluster: str) -> bool:
    """True if the cluster's first codepoint is in the Bengali block."""
    return bool(cluster) and 0x0980 <= ord(cluster[0]) <= 0x09FF


def codepoints(cluster: str) -> list[str]:
    """The individual codepoints of a cluster, for the decomposition fallback."""
    return list(cluster)


def describe(cluster: str) -> dict:
    """Structural description of one cluster, for analysis and tests."""
    return {
        "cluster": cluster,
        "codepoints": len(cluster),
        "conjunct": is_conjunct(cluster),
        "reph": has_reph(cluster),
        "ya_phala": has_ya_phala(cluster),
        "ra_phala": has_ra_phala(cluster),
        "nukta": has_nukta(cluster),
        "bengali": is_bengali(cluster),
    }
