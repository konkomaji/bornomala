r"""
Normalisation for Bengali text (Project Bornomala, Track A).

Every Bengali string is normalised before anything else touches it. This is not
optional cleanup, it is correctness: the same visible word can be encoded in
several codepoint sequences, and any metric or merge computed without prior
normalisation is measuring encoding noise rather than language (spec section 3.1,
requirement B-1).

Three steps:

  1. NFC (Unicode Normalization Form C, UAX #15). Canonical composition. This
     unifies most canonically-equivalent spellings of the same visible text into
     one codepoint sequence, INCLUDING, perhaps surprisingly, Bengali's three
     "composition exclusions" (see step 2): NFC always fully decomposes first
     and then recomposes everything except exclusions, so both spellings of
     those three letters already converge under plain NFC. They just converge
     onto the two-codepoint decomposed form, not the single dedicated
     codepoint Unicode also provides for them. Verified empirically against
     Python's own `unicodedata` (see docs/bengali-script-reference.md); there
     was no actual cross-source encoding-inconsistency bug here.

  2. Explicit re-composition of RRA, RHA, YYA (ড়/ঢ়/য়, U+09DC/U+09DD/U+09DF)
     back to their single dedicated codepoint. This is a minor efficiency
     choice, not a correctness fix: it spends one atom instead of two on three
     letters that are common in everyday Bengali (e.g. বড়, পড়া, গড়ে), and
     matches the form the Bengali Unicode block itself was designed around.
     Skipping this step would still be correct, just very slightly less
     efficient.

  3. An explicit, documented ZWJ / ZWNJ policy. Zero-width joiner (U+200D) and
     non-joiner (U+200C) are used inconsistently in real Bengali text to force or
     suppress ligature formation. The whitepaper is explicit: do not silently
     strip them, they carry intent (spec section 9.2, step 2). So the default
     policy preserves them. A 'strip' policy is offered for ablation only, and it
     is documented as lossy.

Nothing here is Bengali-only in a way that would corrupt other scripts: NFC is
universal, the exclusion re-composition is a no-op unless those three specific
sequences appear, and the ZWJ / ZWNJ handling is a no-op unless those characters
appear.

A cautionary note for anyone editing this file: typing the Bengali literals for
these three letters directly into source can silently round-trip through an
editor/tool pipeline that re-decomposes them, turning a fix into an accidental
no-op (this happened once during development). `_NFC_EXCLUSIONS` below is built
from explicit `chr()` codepoints for exactly that reason; keep it that way
rather than reverting to literal characters.
"""

from __future__ import annotations

import unicodedata

from .errors import ConfigError, NormalizationError

ZWJ = "‍"   # zero-width joiner
ZWNJ = "‌"  # zero-width non-joiner

# Bengali khanda ta has a dedicated codepoint (U+09CE) in modern Unicode. Legacy
# text sometimes spells it as ta + virama + ZWJ. We can optionally canonicalise
# that legacy sequence to the dedicated codepoint. This is a documented, opt-in
# normalisation, not silent.
_TA = "ত"       # ত
_VIRAMA = "্"   # ্ (hasanta)
_KHANDA_TA = "ৎ"  # ৎ
_LEGACY_KHANDA_TA = _TA + _VIRAMA + ZWJ

# Bengali's NFC composition exclusions (decomposed spelling -> precomposed
# singleton). Built from explicit chr() codepoints, not literal Bengali text,
# to guarantee correctness regardless of source-file/editor normalisation; see
# the docstring above.
_NFC_EXCLUSIONS = {
    chr(0x09A1) + chr(0x09BC): chr(0x09DC),  # DDA + NUKTA -> RRA (ড় as one codepoint)
    chr(0x09A2) + chr(0x09BC): chr(0x09DD),  # DDHA + NUKTA -> RHA (ঢ় as one codepoint)
    chr(0x09AF) + chr(0x09BC): chr(0x09DF),  # YA + NUKTA -> YYA (য় as one codepoint)
}


def normalize(text: str, zwnj_policy: str = "preserve", canonical_khanda_ta: bool = True) -> str:
    """Normalise Bengali text.

    Args:
      text: input string.
      zwnj_policy: 'preserve' (default, keeps ZWJ/ZWNJ, they carry intent) or
        'strip' (removes them; lossy, for ablation only).
      canonical_khanda_ta: if True, rewrite the legacy ta+virama+ZWJ sequence to
        the dedicated khanda ta codepoint U+09CE before NFC.

    Returns:
      The normalised string.

    Raises:
      NormalizationError: if `text` is not a string.
      ConfigError: if `zwnj_policy` is not 'preserve' or 'strip'.
    """
    if not isinstance(text, str):
        raise NormalizationError(f"expected str, got {type(text).__name__}")
    if zwnj_policy not in ("preserve", "strip"):
        raise ConfigError(f"unknown zwnj_policy: {zwnj_policy!r} (use 'preserve' or 'strip')")

    if not text:
        return ""

    try:
        if canonical_khanda_ta and _LEGACY_KHANDA_TA in text:
            text = text.replace(_LEGACY_KHANDA_TA, _KHANDA_TA)

        text = unicodedata.normalize("NFC", text)
        for decomposed, singleton in _NFC_EXCLUSIONS.items():
            if decomposed in text:
                text = text.replace(decomposed, singleton)

        if zwnj_policy == "strip":
            text = text.replace(ZWJ, "").replace(ZWNJ, "")
    except (UnicodeError, ValueError) as e:  # pragma: no cover - defensive
        raise NormalizationError(f"normalisation failed: {e}") from e

    return text


def is_nfc(text: str) -> bool:
    """True if `text` is already in NFC form."""
    return unicodedata.is_normalized("NFC", text)
