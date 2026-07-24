r"""
Bengali Unicode substrate: the character inventory everything else sits on
(v2 design roadmap steps 1-2, docs/design/reading-bengali-on-its-own-terms.md
section 8, docs/design/FORMAL_SPEC.md).

Pure data, no functions. `graphemes.py` (UAX #29 segmentation and structural
predicates) and `akshara.py` (the finite-state akshara grammar) both import
their Bengali codepoint sets from here rather than defining their own, so
there is exactly one place that says what counts as a consonant, a vowel, a
matra, or a modifier.

Nothing here is normative beyond what the Unicode Standard's Bengali block
(U+0980-U+09FF) already specifies; see docs/bengali-script-reference.md for
the full annotated inventory this was checked against.
"""

from __future__ import annotations

# --- primitives ------------------------------------------------------------
VIRAMA = "্"        # hasanta (U+09CD), triggers conjunct formation
NUKTA = "়"         # U+09BC, forms ড়/ঢ়/য় with a base consonant
RA = "র"            # the consonant that forms reph / ra-phala
YA = "য"            # forms ya-phala
ZWJ = "‍"            # U+200D, zero-width joiner
ZWNJ = "‌"           # U+200C, zero-width non-joiner

# --- the four akshara-grammar character classes ----------------------------
# Consonants: the main U+0995-U+09B9 range, plus the three nukta-formed
# letters that have their own dedicated codepoints (ড়/ঢ়/য়), plus khanda-ta
# (ৎ, U+09CE) which is an ordinary consonant in modern usage, not a ta+virama
# ligature (see normalize.py's legacy-khanda-ta handling for the historical
# spelling this supersedes). Built from explicit chr() codepoints, not
# literal characters: typing RRA/RHA/YYA directly into source can silently
# round-trip through an editor/tool pipeline that re-decomposes them onto
# their two-codepoint NFC-exclusion spelling, turning this set quietly wrong
# (see normalize.py's docstring for the same caution; it happened once there
# and happened again here while writing this file).
CONSONANTS = ({chr(c) for c in range(0x0995, 0x09B9 + 1)}
              | {chr(0x09DC), chr(0x09DD), chr(0x09DF), chr(0x09CE)}
              # RA WITH MIDDLE DIAGONAL / RA WITH LOWER DIAGONAL: rare in
              # standard Bengali, used in Assamese (a script Unicode shares
              # this same block with) and occasionally in older Bengali text.
              | {chr(0x09F0), chr(0x09F1)})

# Independent vowels: used word-initially or after another vowel/modifier,
# never directly attached to a consonant (that role belongs to MATRAS).
# VOCALIC RR / VOCALIC LL (U+09E0/U+09E1) are obsolete in modern Bengali
# (Sanskrit-loanword vowels not used in everyday spelling) but are real,
# assigned Bengali-block letters; included for completeness rather than left
# to fall through to the "other" bucket unnamed.
VOWELS = ({chr(c) for c in range(0x0985, 0x098C + 1)} | {"এ", "ঐ", "ও", "ঔ"}
          | {chr(0x09E0), chr(0x09E1)})

# Dependent vowel signs (matra / kar): each is a single Unicode codepoint,
# including ো and ৌ, which are visually two-part (a component before the
# consonant and one after) but are still one codepoint each in the Bengali
# block, unlike some other Brahmic scripts' equivalent signs. VOWEL SIGN
# VOCALIC L / VOCALIC LL (U+09E2/U+09E3) are the matra forms of the obsolete
# vowels above, included for the same completeness reason.
MATRAS = ({chr(c) for c in range(0x09BE, 0x09CC + 1)} | {"ৗ"}
          | {chr(0x09E2), chr(0x09E3)})

# Modifiers: candrabindu (nasalization), anusvara, visarga, plus the SANDHI
# MARK (U+09FE, a nonspacing combining mark used to indicate a sandhi/vowel-
# fusion boundary - included here rather than treated as punctuation since
# Unicode categorises it Mn, same as the other three). Attach after a vowel
# or a completed consonant-conjunct-matra sequence.
MODIFIERS = {"ঁ", "ং", "ঃ", chr(0x09FE)}  # candrabindu, anusvara, visarga, sandhi mark

# BENGALI LETTER VEDIC ANUSVARA (U+09FC) is deliberately NOT in MODIFIERS
# despite its name: Unicode categorises it Lo (a standalone letter), not Mn/Mc
# like the combining modifiers above, and it is Vedic-recitation-only, not
# used in running Bengali text. It correctly falls to akshara.py's "other"
# bucket. BENGALI ABBREVIATION SIGN (U+09FD) is punctuation (Po), also
# correctly left to fall through to "other"; both are named here so neither
# is an unnoticed gap.
VEDIC_ANUSVARA = chr(0x09FC)
ABBREVIATION_SIGN = chr(0x09FD)

# --- matra visual position (informational only) -----------------------------
# A simplified approximation of Unicode's own Indic_Positional_Category:
# where a matra renders relative to its base consonant. This is NOT consumed
# by the akshara grammar in akshara.py (segmentation only needs matra
# *membership*, never reorder direction) - it exists for future featural
# encoding (v2 roadmap step 5), where a model may want to know that ি visually
# precedes its consonant even though it is stored after it.
#   "before": renders to the left of the base consonant
#   "after":  renders to the right of the base consonant
#   "below":  renders under the base consonant
#   "split":  renders as two visual components, one on each side
MATRA_POSITION: dict[str, str] = {
    "া": "after",       # AA
    "ি": "before",      # I
    "ী": "after",       # II
    "ু": "below",       # U
    "ূ": "below",       # UU
    "ৃ": "below",       # VOCALIC R
    "ৄ": "below",       # VOCALIC RR (rare)
    chr(0x09E2): "below",  # VOWEL SIGN VOCALIC L (obsolete)
    chr(0x09E3): "below",  # VOWEL SIGN VOCALIC LL (obsolete)
    "৅": "unassigned",  # no character assigned at this codepoint; MATRAS
                              # includes it only because it falls inside the
                              # 0x09BE-0x09CC blanket range (pre-existing, carried
                              # over unchanged from graphemes.py, not new to this PR)
    "৆": "unassigned",  # same as above
    "ে": "before",      # E
    "ৈ": "before",      # AI
    "৉": "unassigned",  # same as above
    "৊": "unassigned",  # same as above
    "ো": "split",       # O (e-matra component before, aa-matra component after)
    "ৌ": "split",       # AU (e-matra component before, au-length-mark after)
    "ৗ": "after",       # AU LENGTH MARK (after-component of the decomposed AU spelling)
}

# --- shared punctuation ------------------------------------------------------
# Bengali's own sentence- and verse-terminating punctuation (দাঁড়ি / দ্বিদাঁড়ি) is
# NOT in the Bengali Unicode block: it is the shared, script-neutral DANDA and
# DOUBLE DANDA (U+0964/U+0965), a Devanagari-block codepoint pair every Indic
# script that uses this punctuation reuses rather than re-encoding.
DANDA = "।"
DOUBLE_DANDA = "॥"
SHARED_INDIC_PUNCTUATION = frozenset({DANDA, DOUBLE_DANDA})

# --- coverage sets -----------------------------------------------------------
# Bengali block also contains digits, currency signs (৲৳৴-৹), fraction signs,
# and a handful of historical/rare letters that are none of consonant, vowel,
# matra, or modifier above; the akshara grammar in akshara.py correctly falls
# through to its "other" bucket for all of these, no special-casing needed.
BENGALI_BLOCK = frozenset(chr(c) for c in range(0x0980, 0x09FF + 1))
ASCII_PRINTABLE = frozenset(chr(c) for c in range(0x20, 0x7E + 1))
GUARANTEED_CODEPOINTS = BENGALI_BLOCK | SHARED_INDIC_PUNCTUATION | ASCII_PRINTABLE
