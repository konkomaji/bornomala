"""
Tokenizer efficiency metrics.

All metrics are computed from two inputs: the source text and the list of tokens
a tokenizer produced for it. Definitions follow the Bornomala spec and the
tokenization literature (§4.1, §9.4).

  fertility        = |tokens| / |words|
                     Average subword tokens per whitespace word. Lower = better.
                     Controls training/inference cost and effective context.

  strr             = |{word : tokenizer(word) == 1 token}| / |words|
                     Single Token Retention Rate. Exposes vocabulary allocation
                     where fertility alone cannot (spec §4.1).

  bytes_per_token  = utf8_bytes(text) / |tokens|
                     Script-independent compression. Higher = better.

  cp_per_token     = codepoints(text) / |tokens|
  gc_per_token     = grapheme_clusters(text) / |tokens|
                     Two honest notions of "characters per token". For Bengali
                     the grapheme-cluster figure is the meaningful one (§3.1.2).

Relative fertility (vs an English baseline) is what actually explains cost: a
model with English fertility 1.2 and Bengali fertility 4.8 charges 4x as many
tokens for the same *content*. That ratio is the headline the tool exists to
surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .segment import grapheme_clusters as _gcs
from .segment import words as _words


@dataclass
class Metrics:
    model: str
    n_tokens: int
    n_words: int
    n_codepoints: int
    n_graphemes: int
    n_bytes: int
    fertility: float
    strr: float
    bytes_per_token: float
    cp_per_token: float
    gc_per_token: float
    estimated: bool = False  # True when the token count is a heuristic, not exact
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute(
    model: str,
    text: str,
    n_tokens: int,
    single_token_words: int | None = None,
    estimated: bool = False,
    note: str = "",
) -> Metrics:
    """Build a Metrics record.

    `single_token_words` is the count of whitespace words that the tokenizer
    encodes as exactly one token (needed for STRR). When a backend cannot supply
    it (e.g. a pure estimate), pass None and STRR is reported as 0.0 with the
    fact carried by `estimated`.
    """
    wlist = _words(text)
    n_words = len(wlist)
    n_cp = len(text)
    n_gc = len(_gcs(text))
    n_bytes = len(text.encode("utf-8"))

    strr = _safe_div(single_token_words or 0, n_words)

    return Metrics(
        model=model,
        n_tokens=n_tokens,
        n_words=n_words,
        n_codepoints=n_cp,
        n_graphemes=n_gc,
        n_bytes=n_bytes,
        fertility=_safe_div(n_tokens, n_words),
        strr=strr,
        bytes_per_token=_safe_div(n_bytes, n_tokens),
        cp_per_token=_safe_div(n_cp, n_tokens),
        gc_per_token=_safe_div(n_gc, n_tokens),
        estimated=estimated,
        note=note,
    )


def relative_fertility(m: Metrics, baseline_fertility: float) -> float:
    """How many times more tokens per word this model uses vs a baseline.

    e.g. baseline (English) fertility 1.2, this 4.8 -> 4.0x. This is the number
    that explains a cost difference between languages on the same model.
    """
    return _safe_div(m.fertility, baseline_fertility)
