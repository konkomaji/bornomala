r"""What a token boundary actually costs, graded by what it severs.

`evaluate.py`'s original `conjunct_fragmentation_rate` had three defects that
only became visible once BMBT's morphology layer started splitting grapheme
clusters deliberately. All three are measurement problems, not tokenizer
problems, and every published number this project has is affected by the
second one.

  1. IT DOES NOT MEASURE CONJUNCTS. It counts any split grapheme cluster.
     Separating a consonant cluster from its matra (`শ্বে` -> `শ্ব` + `ে`) is
     counted identically to severing the cluster itself (`ক্ষ` -> `ক্` + `ষ`).
     The first is the decomposition Bengali literacy teaches; the second
     produces a fragment with a dangling virama that occurs nowhere in real
     text.

  2. THE DENOMINATOR INCLUDES CLUSTERS THAT CANNOT BE SPLIT. Measured on
     697,048 held-out clusters, 61.0% are a single codepoint: a bare
     consonant, a digit, a space, an ASCII letter. No tokenizer can split
     them, and counting them rewards every tokenizer for not doing the
     impossible. The inflation is 2.56x over all clusters and 1.79x over
     Bengali ones. AI4Bharat IndicBERTv2's published 4.40% on Wikipedia is
     7.88% of the clusters it could actually have broken.

  3. IT IS BINARY. A split is a split, so breaking a three-consonant conjunct
     down the middle scores the same as clipping a trailing anusvara.

GRADING WITHOUT ARBITRARY WEIGHTS
---------------------------------
The obvious fix, weighting split types by severity, was rejected: a weight is
a judgement presented as a measurement, and this project's rule E4 forbids
exactly that. Instead each split is classified by an objective structural
test, and the three counts are reported separately so a reader can apply their
own judgement to numbers that are all real.

  DESTRUCTIVE - the split produces a piece that occurs nowhere in Bengali, or
                changes a letter's identity:
                  * a virama is stranded from the consonant it joins
                    (`ক্` + `ষ`). `ক্` is not a unit of the language;
                  * a nukta is detached from its base (`ড` + `়`). This is not
                    damage to a cluster, it is a different letter: `ড` and
                    `ড়` are distinct, and NFC keeps them decomposed because
                    the composed forms are a permanent Unicode composition
                    exclusion (see docs/bengali-script-reference.md).

  MODIFIER      - a trailing modifier is detached (anusvara `ং`, visarga `ঃ`,
                chandrabindu `ঁ`). These are separate phonemes and only weakly
                bound to the akshara. Recoverable, not meaningless.

  ONSET_RIME  - a consonant cluster parted from its matra (`শ্ব` + `ে`). Both
                pieces are units Bengali literacy names, so nothing is
                destroyed. Counting it as damage is what made the original
                metric report a 3.3% "regression" for a change that severs no
                conjuncts at all.

                The name is deliberately descriptive rather than approving.
                An earlier draft called this class "justified", which claims
                the split was warranted: true of BMBT, which targets morpheme
                seams on purpose, but false of a byte-level BPE that lands
                there by frequency accident. This measure grades the OUTCOME
                of a split and cannot see intent, so it must not imply it.
                Whether a split was linguistically motivated is a separate
                question, answered by morphological alignment, not here.

The headline number is `destructive_rate`: destructive splits over clusters
that could have been split. That is what "conjunct fragmentation" was always
meant to say.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import substrate
from .graphemes import grapheme_clusters

DESTRUCTIVE = "destructive"
MODIFIER = "modifier"
ONSET_RIME = "onset_rime"


@dataclass(frozen=True, slots=True)
class SplitCounts:
    """Intra-cluster token boundaries, graded."""

    destructive: int
    modifier: int
    onset_rime: int
    splittable_clusters: int
    total_clusters: int

    @property
    def total_splits(self) -> int:
        return self.destructive + self.modifier + self.onset_rime

    @property
    def destructive_rate(self) -> float:
        """Destructive splits per cluster that could have been split."""
        return self.destructive / self.splittable_clusters if self.splittable_clusters else 0.0

    @property
    def any_split_rate(self) -> float:
        """Every intra-cluster split, over splittable clusters.

        The corrected-denominator counterpart of the legacy
        `conjunct_fragmentation_rate`.
        """
        return self.total_splits / self.splittable_clusters if self.splittable_clusters else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["destructive_rate"] = self.destructive_rate
        d["any_split_rate"] = self.any_split_rate
        return d


def classify_split(tail: str, head: str) -> str:
    """Grade a boundary that falls between `tail` and `head` inside one cluster.

    `tail` is the part staying with the earlier token, `head` the part starting
    the later one. Order of tests matters: a stranded virama is checked before
    anything else, because that is the failure this tokenizer exists to
    prevent.
    """
    if not tail or not head:
        return ONSET_RIME  # nothing was actually severed
    if tail.endswith(substrate.VIRAMA) or head.startswith(substrate.VIRAMA):
        return DESTRUCTIVE
    if head.startswith(substrate.NUKTA):
        return DESTRUCTIVE
    if head[0] in substrate.MATRAS:
        return ONSET_RIME
    if head[0] in substrate.MODIFIERS:
        return MODIFIER
    # Anything else severs a cluster in a way none of the structural cases
    # describe. Counted as destructive rather than waved through: an unclassified
    # split is not evidence of a harmless one.
    return DESTRUCTIVE


def count_splits_from_offsets(text: str, boundaries) -> SplitCounts:
    """Grade every intra-cluster boundary given token boundary offsets.

    The offset form exists because a byte-level BPE's token surfaces are not
    text: `scripts/compare.py` already reads external tokenizers through
    `offset_mapping` for exactly that reason. `boundaries` is any iterable of
    codepoint offsets where a token starts; 0 and len(text) are ignored.
    """
    clusters = grapheme_clusters(text)
    total = len(clusters)
    splittable = sum(1 for c in clusters if len(c) > 1)

    # Map each codepoint offset to the cluster containing it.
    starts = []
    pos = 0
    for c in clusters:
        starts.append((pos, pos + len(c)))
        pos += len(c)
    span_of = {}
    for lo, hi in starts:
        for i in range(lo, hi):
            span_of[i] = (lo, hi)

    counts = {DESTRUCTIVE: 0, MODIFIER: 0, ONSET_RIME: 0}
    for cut in boundaries:
        if cut <= 0 or cut >= len(text):
            continue
        lo, hi = span_of.get(cut, (cut, cut))
        if cut == lo:
            continue  # boundary sits between clusters: nothing severed
        counts[classify_split(text[lo:cut], text[cut:hi])] += 1

    return SplitCounts(
        destructive=counts[DESTRUCTIVE],
        modifier=counts[MODIFIER],
        onset_rime=counts[ONSET_RIME],
        splittable_clusters=splittable,
        total_clusters=total,
    )


def count_splits(token_surfaces: list[str], text: str | None = None) -> SplitCounts:
    """Grade every intra-cluster boundary across a token sequence.

    `text` supplies the denominator when given; otherwise the surfaces are
    concatenated, which is equivalent whenever the tokenizer round-trips.
    """
    whole = text if text is not None else "".join(token_surfaces)
    clusters = grapheme_clusters(whole)
    total = len(clusters)
    splittable = sum(1 for c in clusters if len(c) > 1)

    counts = {DESTRUCTIVE: 0, MODIFIER: 0, ONSET_RIME: 0}
    for i in range(len(token_surfaces) - 1):
        a, b = token_surfaces[i], token_surfaces[i + 1]
        if not a or not b:
            continue
        tail = grapheme_clusters(a)[-1]
        head = grapheme_clusters(b)[0]
        # They only severed a cluster if the two pieces re-cluster into one.
        if len(grapheme_clusters(tail + head)) != 1:
            continue
        counts[classify_split(tail, head)] += 1

    return SplitCounts(
        destructive=counts[DESTRUCTIVE],
        modifier=counts[MODIFIER],
        onset_rime=counts[ONSET_RIME],
        splittable_clusters=splittable,
        total_clusters=total,
    )
