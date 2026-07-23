"""
High-level analysis API.

`analyze(text, models)` runs each requested tokenizer over the text and returns
a list of per-model results: exact metrics plus the token surface strings needed
to *show* how the text was split. `analyze_many` does the same for a labelled set
of texts (e.g. one per language) and is what produces the cross-tokenizer /
cross-language comparison tables.

Results carry availability status so callers (CLI, web) can show which models
loaded and which need a token or a download, rather than failing the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import get_model, DEFAULT_MODELS
from .backends import BackendError, Encoding
from .metrics import Metrics, compute, relative_fertility
from .segment import script_histogram, dominant_script


@dataclass
class Result:
    model_id: str
    display: str
    available: bool
    metrics: Metrics | None = None
    tokens: list[str] = field(default_factory=list)
    error: str = ""
    english_fertility: float | None = None  # this model's fertility on the English anchor
    vs_english: float | None = None         # fertility(text) / english_fertility

    def as_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "display": self.display,
            "available": self.available,
            "error": self.error,
            "tokens": self.tokens,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "english_fertility": self.english_fertility,
            "vs_english": self.vs_english,
        }
        return d


def analyze_one(text: str, model_id: str, want_tokens: bool = True, anchor_english: bool = True) -> Result:
    model = get_model(model_id)
    try:
        backend = model.backend()
        enc: Encoding = backend.encode(text)
    except BackendError as e:
        return Result(model_id, model.display, available=False, error=str(e))
    except Exception as e:  # pragma: no cover - defensive
        return Result(model_id, model.display, available=False, error=f"{type(e).__name__}: {e}")

    m = compute(
        model=model_id,
        text=text,
        n_tokens=enc.n_tokens,
        single_token_words=enc.single_token_words,
        estimated=enc.estimated,
        note=enc.note,
    )
    result = Result(
        model_id=model_id,
        display=model.display,
        available=True,
        metrics=m,
        tokens=enc.tokens if want_tokens else [],
    )
    if anchor_english:
        # Lazy import avoids a circular dependency (baseline imports analyze).
        from .baseline import english_reference_fertility, ENGLISH_REFERENCE

        # Do not anchor the anchor to itself.
        if text.strip() != ENGLISH_REFERENCE.strip():
            ef = english_reference_fertility(model_id)
            result.english_fertility = ef
            if ef:
                result.vs_english = m.fertility / ef
    return result


def analyze(text: str, models: list[str] | None = None, want_tokens: bool = True,
            anchor_english: bool = True) -> list[Result]:
    """Analyse one text across several models. Order follows `models`."""
    ids = models or DEFAULT_MODELS
    return [analyze_one(text, mid, want_tokens=want_tokens, anchor_english=anchor_english) for mid in ids]


@dataclass
class LangResult:
    label: str
    text: str
    dominant_script: str
    script_histogram: dict[str, int]
    results: list[Result]


def analyze_many(
    texts: dict[str, str],
    models: list[str] | None = None,
) -> list[LangResult]:
    """Analyse a labelled set of texts (label -> text) across models.

    Produces the material for a cross-language x cross-tokenizer table. Labels
    are usually language names; the dominant script is auto-detected per text.
    """
    out: list[LangResult] = []
    for label, text in texts.items():
        out.append(
            LangResult(
                label=label,
                text=text,
                dominant_script=dominant_script(text),
                script_histogram=script_histogram(text),
                results=analyze(text, models, want_tokens=False),
            )
        )
    return out


def cost_explanation(results: list[Result], baseline_model_english_fertility: float | None = None) -> list[str]:
    """Plain-language 'why the cost moves' notes derived from the metrics.

    Compares each available model's fertility to the best (lowest) among them and
    states the multiplier - the number that directly explains a token-count, and
    therefore cost, difference on the same content.
    """
    avail = [r for r in results if r.available and r.metrics]
    if not avail:
        return ["No tokenizer produced a result."]
    best = min(avail, key=lambda r: r.metrics.fertility)
    lines = []
    for r in avail:
        mult = relative_fertility(r.metrics, best.metrics.fertility)
        tag = " (estimate)" if r.metrics.estimated else ""
        if r is best:
            lines.append(
                f"{r.display}{tag}: most efficient here - {r.metrics.fertility:.2f} tokens/word, "
                f"{r.metrics.bytes_per_token:.1f} bytes/token."
            )
        else:
            lines.append(
                f"{r.display}{tag}: {mult:.2f}x more tokens than {best.display} "
                f"({r.metrics.fertility:.2f} vs {best.metrics.fertility:.2f} tokens/word) "
                f"→ ~{mult:.2f}x the cost for this text."
            )
    return lines
