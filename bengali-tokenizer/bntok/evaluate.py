r"""
Evaluation for the Track A tokenizer.

Reports the metrics the whitepaper requires (spec section 9.2 step 4, section
9.4), disaggregated so nothing hides in an aggregate:

  fertility            tokens / whitespace-words. Lower is better.
  strr                 fraction of words encoded as exactly one token.
  bytes_per_token      UTF-8 bytes / tokens. Script-independent compression.
  gc_per_token         grapheme clusters / tokens. True characters per token.
  conjunct_fragmentation_rate
                       LEGACY. Fraction of ALL grapheme clusters that a token
                       boundary splits. Retained unchanged for comparability
                       with published numbers, but it is misnamed and its
                       denominator is wrong: it counts any split cluster rather
                       than only severed conjuncts, and it divides by clusters
                       that cannot be split. See fragmentation.py.
  destructive_rate     HEADLINE. Splits that strand a virama or detach a nukta,
                       over clusters that could have been split. This is what
                       the legacy field was always meant to say.
  any_split_rate       every intra-cluster split, corrected denominator.
  n_destructive / n_modifier / n_onset_rime
                       the graded counts, reported separately rather than
                       collapsed behind a severity weight (rule E4: a weight is
                       a judgement presented as a measurement).
  roundtrip_ok         did encode then decode reproduce the normalised text.

All inputs are NFC-normalised before measurement (requirement B-1). The
fragmentation measure is computed by checking, at every adjacent token boundary,
whether joining the two token surfaces yields fewer grapheme clusters than the
two separately: if so, a cluster was split across that boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .fragmentation import count_splits
from .graphemes import grapheme_clusters
from .normalize import normalize
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
    # Graded replacement, see fragmentation.py. `destructive_rate` is what
    # `conjunct_fragmentation_rate` was always meant to say; the legacy field
    # is kept so every already-published number stays comparable.
    destructive_rate: float
    any_split_rate: float
    n_destructive: int
    n_modifier: int
    n_onset_rime: int
    splittable_clusters: int
    roundtrip_ok: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _fragmented_boundaries(token_surfaces: list[str]) -> int:
    """Count adjacent token boundaries that split a grapheme cluster.

    NOTE: despite the `conjunct_fragmentation_rate` field this feeds, it counts
    ANY split grapheme cluster, not only severed conjuncts, and it divides by
    ALL clusters including the 61% that are a single codepoint and cannot be
    split at all. Both are kept exactly as they were so that every
    already-published number stays comparable. `fragmentation.count_splits` is
    the corrected, graded measure and `destructive_rate` is the headline.
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
    n_words = n_tokens = n_gc = n_bytes = n_frag = 0
    splits = []
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
        splits.append(count_splits(surfaces, nfc))

        if roundtrip and not tok.roundtrip_ok(raw):
            roundtrip = False

    n_destructive = sum(s.destructive for s in splits)
    n_modifier = sum(s.modifier for s in splits)
    n_onset_rime = sum(s.onset_rime for s in splits)
    n_splittable = sum(s.splittable_clusters for s in splits)

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
        destructive_rate=div(n_destructive, n_splittable),
        any_split_rate=div(n_destructive + n_modifier + n_onset_rime, n_splittable),
        n_destructive=n_destructive,
        n_modifier=n_modifier,
        n_onset_rime=n_onset_rime,
        splittable_clusters=n_splittable,
        roundtrip_ok=roundtrip,
    )
