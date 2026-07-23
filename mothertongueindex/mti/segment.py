r"""
Text segmentation primitives.

Three units matter for tokenizer analysis and they are routinely conflated:

  * word            - whitespace-delimited run. Fertility is defined over this
                      (spec §4.1: words(D) is whitespace segmentation).
  * codepoint       - a single Unicode scalar. What naive tools count as a
                      "character". Misleading for Bengali (§3.1.2).
  * grapheme cluster- what a human reads as one written unit. For Bengali a
                      base consonant plus its conjuncts, matras, reph, phalas.
                      This is the correct notion of "character" (§3.1.1).

We use the `regex` module's `\X` for UAX #29 extended grapheme clusters, which
handles Bengali conjunct + matra + reph sequences correctly, unlike str
iteration over codepoints.
"""

from __future__ import annotations

import regex as _re

# UAX #29 extended grapheme cluster.
_GRAPHEME = _re.compile(r"\X")

# Whitespace-delimited word (spec's fertility denominator).
_WS = _re.compile(r"\S+")


def words(text: str) -> list[str]:
    """Whitespace-delimited tokens. The denominator in fertility."""
    return _WS.findall(text)


def grapheme_clusters(text: str) -> list[str]:
    """UAX #29 extended grapheme clusters. The correct 'characters' for Bengali."""
    return _GRAPHEME.findall(text)


# --- Script identification -------------------------------------------------
#
# Per-script breakdown lets the tool answer "which language/script is expensive
# in this model?". We classify each grapheme cluster by the Unicode script of
# its first strong (letter) codepoint, using regex script properties.

# Ordered probes: first match wins. Latin last so combining/format chars in
# other scripts do not steal clusters.
_SCRIPT_PROBES: list[tuple[str, _re.Pattern[str]]] = [
    ("Bengali", _re.compile(r"\p{Bengali}")),
    ("Devanagari", _re.compile(r"\p{Devanagari}")),
    ("Tamil", _re.compile(r"\p{Tamil}")),
    ("Telugu", _re.compile(r"\p{Telugu}")),
    ("Kannada", _re.compile(r"\p{Kannada}")),
    ("Malayalam", _re.compile(r"\p{Malayalam}")),
    ("Gujarati", _re.compile(r"\p{Gujarati}")),
    ("Gurmukhi", _re.compile(r"\p{Gurmukhi}")),
    ("Oriya", _re.compile(r"\p{Oriya}")),
    ("Arabic", _re.compile(r"\p{Arabic}")),
    ("Hebrew", _re.compile(r"\p{Hebrew}")),
    ("Han", _re.compile(r"\p{Han}")),
    ("Hiragana", _re.compile(r"\p{Hiragana}")),
    ("Katakana", _re.compile(r"\p{Katakana}")),
    ("Hangul", _re.compile(r"\p{Hangul}")),
    ("Thai", _re.compile(r"\p{Thai}")),
    ("Cyrillic", _re.compile(r"\p{Cyrillic}")),
    ("Greek", _re.compile(r"\p{Greek}")),
    ("Latin", _re.compile(r"\p{Latin}")),
]

_DIGIT = _re.compile(r"\p{Nd}")


def script_of(cluster: str) -> str:
    """Best-effort script label for a grapheme cluster.

    Returns a Unicode script name, or 'Number', or 'Other' (punctuation,
    symbols, whitespace, emoji, uncovered scripts).
    """
    for name, probe in _SCRIPT_PROBES:
        if probe.search(cluster):
            return name
    if _DIGIT.search(cluster):
        return "Number"
    return "Other"


def script_histogram(text: str) -> dict[str, int]:
    """Count grapheme clusters by script. Excludes pure-whitespace clusters."""
    hist: dict[str, int] = {}
    for g in grapheme_clusters(text):
        if g.isspace():
            continue
        s = script_of(g)
        hist[s] = hist.get(s, 0) + 1
    return hist


def dominant_script(text: str) -> str:
    """The script covering the most grapheme clusters in `text`."""
    hist = script_histogram(text)
    if not hist:
        return "Other"
    return max(hist, key=lambda k: hist[k])
