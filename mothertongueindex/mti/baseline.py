"""
English baseline anchoring.

The tool's headline question is: *how efficient is my language, on this model,
compared to English?* English is the anchor every other language is measured
against, because it is the language every mainstream tokenizer is optimised for.

Two ways to answer it, both provided:

  1. Reference mode (zero effort, approximate).
     Each model has an English *reference fertility*, computed once from a fixed
     public-domain English sample. For the user's pasted text we report
     fertility(text) / english_reference_fertility(model). Reading: "on GPT-4o
     your language averages N tokens per word; English averages M; that is N/M x
     English." It is approximate because it compares *your words* to *English
     words*, not the same content.

  2. Parallel mode (exact).
     The user supplies the same meaning in English and in their language. We
     tokenize both and report token(their) / token(english). This is the honest,
     content-controlled ratio and is what belongs in a benchmark table.

The reference text is UDHR Article 1 (public domain, and the canonical anchor
for cross-language script studies).
"""

from __future__ import annotations

# UDHR Article 1 - public domain. Stable, register-neutral, ~34 words.
ENGLISH_REFERENCE = (
    "All human beings are born free and equal in dignity and rights. "
    "They are endowed with reason and conscience and should act towards "
    "one another in a spirit of brotherhood."
)

# Cache: model_id -> english reference fertility (tokens/word).
_EN_FERT: dict[str, float] = {}


def english_reference_fertility(model_id: str) -> float | None:
    """Fertility of the fixed English reference under `model_id`, cached.

    Returns None if the model's tokenizer is unavailable.
    """
    if model_id in _EN_FERT:
        return _EN_FERT[model_id]
    # Imported here to avoid a circular import at module load.
    from .analyze import analyze_one

    r = analyze_one(ENGLISH_REFERENCE, model_id, want_tokens=False)
    if not r.available or r.metrics is None:
        return None
    _EN_FERT[model_id] = r.metrics.fertility
    return r.metrics.fertility


def vs_english_ratio(model_id: str, text_fertility: float) -> float | None:
    """text_fertility / english_reference_fertility(model). None if unavailable.

    >1.0 means the language spends more tokens per word than English does on the
    same model - i.e. costs more for equivalent structure.
    """
    ef = english_reference_fertility(model_id)
    if not ef:
        return None
    return text_fertility / ef


def parallel_ratio(model_id: str, english_text: str, other_text: str) -> dict | None:
    """Exact content-controlled ratio from a parallel pair.

    Returns token counts for both sides and the ratio other/english, or None if
    the tokenizer is unavailable.
    """
    from .analyze import analyze_one

    en = analyze_one(english_text, model_id, want_tokens=False)
    ot = analyze_one(other_text, model_id, want_tokens=False)
    if not (en.available and ot.available):
        return None
    en_tok = en.metrics.n_tokens
    ot_tok = ot.metrics.n_tokens
    return {
        "model_id": model_id,
        "english_tokens": en_tok,
        "other_tokens": ot_tok,
        "ratio": (ot_tok / en_tok) if en_tok else 0.0,
        "estimated": ot.metrics.estimated or en.metrics.estimated,
    }
