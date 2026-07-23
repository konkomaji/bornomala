r"""
Reasoning-capability impact of tokenization.

Token *inefficiency* is not only a cost problem. It is a capability problem, and
this module makes that link explicit and quantitative - while being scrupulous
about what is measured versus derived.

The mechanism (spec §4.1, IndicGenBench arXiv:2404.16816):

  A fixed context window holds a fixed number of *tokens*, not words. If your
  language has fertility 4x English, then the same context window holds ~1/4 the
  *content*. That means:
    * fewer few-shot examples fit -> in-context learning degrades;
    * long documents truncate sooner;
    * chain-of-thought reasoning has less room before the window fills.

  IndicGenBench shows empirically that higher fertility correlates with degraded
  downstream performance at fixed context. So a language that tokenizes badly is
  handicapped on reasoning tasks *independently* of the model's underlying
  ability, purely through the token budget.

What this module computes (DERIVED, not measured):
  * effective_context_ratio = english_fertility / language_fertility
        The fraction of English's content capacity your language gets in the
        same window. 1.0 = parity; 0.25 = a quarter.
  * usable_tokens_in_window  = how much of an N-token window your content
        actually occupies vs English.
  * a coarse risk band (LOW/MODERATE/HIGH/SEVERE) from the fertility ratio.

What this module does NOT do:
  It does not measure reasoning accuracy. That requires running a model on a
  benchmark - see `eval/reasoning_probe.py`, which is designed to run on a
  separate machine with an API key. Nothing here is presented as a measured
  capability score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Risk bands by fertility ratio vs English. Thresholds are interpretive, chosen
# to match the qualitative breaks seen in the fertility literature.
_BANDS = [
    (1.5, "LOW", "Near-English token budget. Little capability handicap from tokenization."),
    (2.5, "MODERATE", "Noticeably fewer examples/context fit. Some in-context degradation likely."),
    (4.0, "HIGH", "Context effectively shrinks to a fraction of English. Few-shot and long-context reasoning materially handicapped."),
    (float("inf"), "SEVERE", "Context budget dominated by tokenization overhead. Reasoning at fixed context is severely constrained before the model's own ability is even reached."),
]


@dataclass
class CapabilityImpact:
    model_id: str
    vs_english: float                  # fertility ratio (language / english)
    effective_context_ratio: float     # english / language  (content capacity)
    risk_band: str
    explanation: str
    # For a few reference window sizes, English-equivalent content that fits.
    window_equiv: dict                  # e.g. {"8k": {"your_tokens_of_english_content": ...}}
    estimated: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


_WINDOWS = {"8k": 8192, "128k": 131072, "1M": 1_048_576}


def _band(ratio: float) -> tuple[str, str]:
    for thresh, name, expl in _BANDS:
        if ratio < thresh:
            return name, expl
    return _BANDS[-1][1], _BANDS[-1][2]


def assess(model_id: str, vs_english: float | None, estimated: bool = False) -> CapabilityImpact | None:
    """Derive a capability-impact record from a vs-English fertility ratio.

    Returns None if no ratio is available (e.g. the text is English, or the
    tokenizer did not load).
    """
    if not vs_english or vs_english <= 0:
        return None

    eff = 1.0 / vs_english
    band, expl = _band(vs_english)

    window_equiv = {}
    for name, size in _WINDOWS.items():
        # Of an N-token window, how many tokens' worth of *English-equivalent
        # content* your language actually gets to use.
        window_equiv[name] = {
            "window_tokens": size,
            "english_equiv_content_tokens": round(size * eff),
        }

    return CapabilityImpact(
        model_id=model_id,
        vs_english=vs_english,
        effective_context_ratio=eff,
        risk_band=band,
        explanation=expl,
        window_equiv=window_equiv,
        estimated=estimated,
    )


def summary_line(ci: CapabilityImpact) -> str:
    pct = ci.effective_context_ratio * 100
    tag = " (from estimated tokens)" if ci.estimated else ""
    return (
        f"{ci.model_id}: {ci.risk_band} capability risk{tag} - your language gets "
        f"~{pct:.0f}% of English's usable context on this model "
        f"(fertility {ci.vs_english:.2f}x English). {ci.explanation}"
    )
