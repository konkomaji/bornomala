r"""
Evaluation for the Track A tokenizer.

Reports the metrics the whitepaper requires (spec section 9.2 step 4, section
9.4), disaggregated so nothing hides in an aggregate:

  fertility            tokens / whitespace-words. Lower is better.
  strr                 fraction of words encoded as exactly one token.
  bytes_per_token      UTF-8 bytes / tokens. Script-independent compression.
  gc_per_token         grapheme clusters / tokens. True characters per token.
  conjunct_fragmentation_rate
                       fraction of grapheme-cluster boundaries that a token
                       boundary splits. The headline correctness metric. With the
                       atom scheme this must be 0 for text whose clusters were
                       seen in training; rare unseen clusters may decompose.
  roundtrip_ok         did encode then decode reproduce the normalised text.

All inputs are NFC-normalised before measurement (requirement B-1). The
fragmentation measure is computed by checking, at every adjacent token boundary,
whether joining the two token surfaces yields fewer grapheme clusters than the
two separately: if so, a cluster was split across that boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .graphemes import grapheme_clusters
from .normalize import normalize
from .substrate import VIRAMA
from .tokenizer import BengaliTokenizer


@dataclass
class Report:
    n_texts: int
    n_words: int
    n_tokens: int
    n_grapheme_clusters: int
    n_bytes: int
    fertility: float
    strr: float
    bytes_per_token: float
    gc_per_token: float
    conjunct_fragmentation_rate: float
    n_fragmented: int
    conjunct_broken_rate: float
    n_conjunct_broken: int
    roundtrip_ok: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _conjunct_broken_boundaries(token_surfaces: list[str]) -> int:
    """Count boundaries that sever a virama-joined consonant cluster.

    Strictly narrower than `_fragmented_boundaries`, and the difference is the
    point. That function counts every split grapheme cluster, which includes
    separating a consonant cluster from its matra (`শ্বে` -> `শ্ব` + `ে`). This
    one counts only splits that leave a virama stranded from the consonant it
    joins (`ক্ষ` -> `ক্` + `ষ`).

    The distinction was academic while every atom was a whole akshara. It stops
    being academic once BMBT's morphology layer deliberately splits at
    onset/rime seams: a morphology-enabled artifact has a nonzero
    grapheme-cluster fragmentation rate by design, while this rate must remain
    exactly zero. Reporting only the first number would misread that trade as a
    regression; reporting only the second would hide a real cost. Both are
    reported. See `docs/bmbt-morphology.md`.
    """
    broken = 0
    for i in range(len(token_surfaces) - 1):
        a, b = token_surfaces[i], token_surfaces[i + 1]
        if not a or not b:
            continue
        # A conjunct is severed exactly when the virama is parted from the
        # consonant it joins, in either direction.
        if a.endswith(VIRAMA) or b.startswith(VIRAMA):
            broken += 1
    return broken


def _fragmented_boundaries(token_surfaces: list[str]) -> int:
    """Count adjacent token boundaries that split a grapheme cluster.

    NOTE: despite the `conjunct_fragmentation_rate` field this feeds, it counts
    ANY split grapheme cluster, not only severed conjuncts. The name predates
    the distinction and is kept so that every already-published number stays
    comparable; `_conjunct_broken_boundaries` is the stricter measure.
    """
    frag = 0
    for i in range(len(token_surfaces) - 1):
        a, b = token_surfaces[i], token_surfaces[i + 1]
        if not a or not b:
            continue
        if len(grapheme_clusters(a)) + len(grapheme_clusters(b)) != len(grapheme_clusters(a + b)):
            frag += 1
    return frag


def evaluate(tok: BengaliTokenizer, texts: list[str]) -> Report:
    """Evaluate a tokenizer over a list of held-out texts."""
    n_words = n_tokens = n_gc = n_bytes = n_frag = n_broken = 0
    roundtrip = True

    for raw in texts:
        if not isinstance(raw, str) or not raw.strip():
            continue
        nfc = normalize(raw, zwnj_policy=tok.config.get("zwnj_policy", "preserve"))
        words = nfc.split()
        ids = tok.encode(raw)
        surfaces = tok.encode_tokens(raw)

        n_words += len(words)
        n_tokens += len(ids)
        n_gc += len(grapheme_clusters(nfc))
        n_bytes += len(nfc.encode("utf-8"))
        n_frag += _fragmented_boundaries(surfaces)
        n_broken += _conjunct_broken_boundaries(surfaces)

        if roundtrip and not tok.roundtrip_ok(raw):
            roundtrip = False

    def div(a, b):
        return a / b if b else 0.0

    # Per-word single-token retention needs a second pass over words.
    single = total_words = 0
    for raw in texts:
        if not isinstance(raw, str) or not raw.strip():
            continue
        for w in normalize(raw, zwnj_policy=tok.config.get("zwnj_policy", "preserve")).split():
            total_words += 1
            if len(tok.encode(w)) == 1:
                single += 1

    return Report(
        n_texts=sum(1 for t in texts if isinstance(t, str) and t.strip()),
        n_words=n_words,
        n_tokens=n_tokens,
        n_grapheme_clusters=n_gc,
        n_bytes=n_bytes,
        fertility=div(n_tokens, n_words),
        strr=div(single, total_words),
        bytes_per_token=div(n_bytes, n_tokens),
        gc_per_token=div(n_gc, n_tokens),
        conjunct_fragmentation_rate=div(n_frag, n_gc),
        n_fragmented=n_frag,
        conjunct_broken_rate=div(n_broken, n_gc),
        n_conjunct_broken=n_broken,
        roundtrip_ok=roundtrip,
    )
