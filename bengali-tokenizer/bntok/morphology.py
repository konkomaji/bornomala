r"""Bengali inflectional morphology: the second half of BMBT's v2 design step 5.

`akshara.py` parses Bengali's *orthographic* grammar (the syllable). This module
parses part of its *morphological* grammar: the suffix chain a Bengali word
carries after its stem. Both are rule-based and inspectable on purpose. A
statistical segmenter (Morfessor and friends) would have been less work, but
BMBT's whole claim is that it reads Bengali by its own rules rather than
inferring them from counts, and a morphology layer learned from frequency would
give that up at exactly the point where it matters most.

WHAT THIS IS FOR, AND WHAT IT WILL COST
---------------------------------------
`docs/design/FORMAL_SPEC.md` proves that a BPE constrained to respect extra
boundaries cannot beat an unconstrained one on raw token count. Morpheme
boundaries are extra boundaries. So this layer is expected to make fertility
slightly WORSE, not better, and that is written down here before anything is
measured so the result cannot be presented as a surprise either way.

What it is meant to buy is different: boundaries that fall where Bengali's
morphemes actually fall, so that a rare inflected form shares its stem with
every other form of the same word instead of being an unrelated token. That is
a generalisation property, and it is invisible to fertility. It is measured by
morphological-alignment metrics (MorphScore) and, ultimately, by downstream
modelling quality, never by tokens per word.

WHAT IT COVERS, AND WHAT IT DOES NOT
------------------------------------
Bengali inflection is overwhelmingly suffixal, which is why right-to-left
longest-match stripping is a reasonable first approximation. Covered: nominal
case, plural, classifiers/determiners, the common verb-conjugation endings, the
productive derivational suffixes, and the emphatic clitics.

NOT covered, and none of it is pretended otherwise:
  * prefixes (অ-, নি-, বি-, সু-, প্রতি-, উপ-, অনু-), which are largely
    tatsama borrowings from Sanskrit and would need a stem lexicon to strip
    safely;
  * sandhi, where the stem itself changes shape at the boundary
    (করা + ছি -> করছি loses a vowel; this module finds `-ছি` and leaves the
    altered stem alone rather than trying to restore it);
  * compounding (সমাস), which needs a lexicon this project does not have;
  * genuine ambiguity. `-এ` is a locative case marker and also the final
    akshara of many uninflected stems. Nothing here can tell those apart
    without a lexicon or context, so the guards below are deliberately
    conservative: over-stripping a real stem is a worse error than leaving an
    affix attached, because it invents a morpheme boundary that is not there
    and MorphScore counts that as a false split.

The guards are: a minimum stem length measured in ASKHARAS rather than
codepoints (an akshara is Bengali's real unit, and a two-codepoint "stem" like
`ক` + matra is one akshara, not two), and a cap on suffix-chain depth.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import substrate
from .akshara import akshara_bounds
from .errors import ConfigError, NormalizationError

# Minimum aksharas a stem must retain. Below this, stripping is refused.
#
# This was set to 2 on the assumption that it kept the common monosyllabic verb
# roots intact. Running the layer against real forms falsified that immediately:
# `কর` is two aksharas, but `যা` and `দে` are ONE each (consonant plus matra is
# a single akshara, not two), so a floor of 2 refused to strip `যাবেন` into
# `যা` + `বেন` and mis-segmented it as `যাব` + `েন` instead. `বাংলায়` failed
# the same way. The floor is 1; the real protection against over-stripping is
# `_RISKY` below, which is targeted at the short suffixes that are genuinely
# ambiguous rather than applied blindly to every stem.
MIN_STEM_AKSHARAS = 1

# Longest attested inflectional chain in the coverage below is roughly
# stem + classifier + plural + case (`বইগুলোদেরকে`-shaped). 4 is slack enough
# for real text without letting a long stem be shredded into affixes.
MAX_SUFFIX_CHAIN = 4


@dataclass(frozen=True, slots=True)
class Morph:
    """One morphological segment of a word.

    `text` is the exact surface substring, so concatenating every segment's
    `.text` in order reproduces the input word exactly - the same losslessness
    contract `akshara.py` holds itself to. `kind` is "stem" or the suffix class
    that matched. `start`/`end` are codepoint offsets into the word.
    """

    text: str
    kind: str
    start: int
    end: int


# --- the suffix inventory ---------------------------------------------------
#
# Ordered by class. Within `morph_split` the whole inventory is tried
# longest-first regardless of class, so ordering here is documentation, not
# precedence. Every entry is a surface form as it appears in NFC text.

CASE_SUFFIXES = (
    # genitive: -র after a vowel-final stem, -এর after a consonant-final one
    "ের",          # -er
    "র",                # -r
    # objective / dative
    "কে",          # -ke
    # locative
    "তে",          # -te
    "য়",                # -y
    "ে",                # -e   (also a common stem-final akshara: guarded)
    # ablative / comparative, written as separate words as often as not
    "থেকে",  # -theke
    "চেয়ে",  # -cheye
)

PLURAL_SUFFIXES = (
    "দের",                  # -der  (animate genitive/objective)
    "গুলো",            # -gulo
    "গুলি",            # -guli
    "গুলোর",      # -gulor
    "গুলির",      # -gulir
    "রা",                        # -ra   (animate nominative)
    "সমূহ",            # -samuha
    "গণ",                        # -gan
    "বৃন্দ",      # -brinda
)

CLASSIFIER_SUFFIXES = (
    "টা",                  # -ta
    "টি",                  # -ti
    "খানা",      # -khana
    "খানি",      # -khani
    "টুকু",      # -tuku
    "জন",                  # -jon
)

VERB_SUFFIXES = (
    # present continuous
    "ছি", "ছেন", "ছে", "ছ",
    # present perfect
    "েছি", "েছেন", "েছে",
    # simple past
    "লাম", "লেন", "লে", "ল",
    # past continuous / past perfect
    "ছিলাম", "ছিলেন",
    "ছিলে", "ছিল",
    "েছিলাম", "েছিল",
    # future
    "বেন", "বে", "বি", "ব",
    # habitual past
    "তাম", "তেন", "ত",
    # non-finite / participial
    "য়ে", "ইয়া",
    # honorific present
    "েন",
)

DERIVATIONAL_SUFFIXES = (
    "তা",                  # -ta   (nominaliser: স্বাধীন -> স্বাধীনতা)
    "ত্ব",            # -tva
    "পনা", "পন",
    "বান", "মান",
    "ময়", "হীন",
    "িক",                  # -ik   (ঐতিহাসিক)
    "কারী", "কার",
    "ালি", "ামি",
)

# Emphatic clitics. Always outermost when present, and only ever one.
CLITIC_SUFFIXES = (
    "ই",   # -i  emphatic
    "ও",   # -o  also/too
)

_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("clitic", CLITIC_SUFFIXES),
    ("case", CASE_SUFFIXES),
    ("plural", PLURAL_SUFFIXES),
    ("classifier", CLASSIFIER_SUFFIXES),
    ("verb", VERB_SUFFIXES),
    ("derivational", DERIVATIONAL_SUFFIXES),
)

# Flattened, longest surface form first, so matching is unambiguous.
_ORDERED: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((suffix, kind) for kind, group in _CLASSES for suffix in group),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

# Suffixes short enough to be a stem's own final akshara. These are only
# stripped when the remaining stem is comfortably above the minimum, because
# a wrong split here is exactly the false-boundary error MorphScore punishes.
_RISKY = frozenset({
    "ে",   # -e
    "র",   # -r
    "ই",   # -i
    "ও",   # -o
    "য়",   # -y
    "ব",   # -b
    "ল",   # -l
    "ত",   # -t
})
_RISKY_MIN_STEM_AKSHARAS = 2

# Bengali suffixes do not attach in arbitrary order. Stripping right to left,
# the chain runs from outermost to innermost: an emphatic clitic sits outside
# case, case outside plural, plural outside a classifier. A rank may repeat or
# increase as stripping moves inward, never decrease.
#
# This is not a tidiness rule, it is what stops over-stripping. Lowering
# MIN_STEM_AKSHARAS to 1 (needed for one-akshara verb roots like `যা`) made
# `ছেলেরা` segment as `ছে` + `লে`[verb] + `রা`[plural], because `লে` really is
# a past-tense ending and `ছেলে` really does end with those codepoints. The
# ordering constraint rejects it on grammatical grounds instead of by tuning a
# length threshold: a finite verb cannot take a plural marker.
_RANK = {
    "clitic": 0,
    "case": 1,
    "plural": 2,
    "classifier": 3,
    "derivational": 4,
    "verb": 4,
}

# A verb ending attaches directly to the root. Nothing nominal may sit outside
# it - only an emphatic clitic (`করছিও` is well formed, `করছিরা` is not).
_ALLOWED_OUTSIDE_VERB = frozenset({"clitic"})


def _akshara_count(text: str) -> int:
    return len(akshara_bounds(text))


def cuts_inside_conjunct(word: str, cut: int) -> bool:
    """Would splitting `word` at `cut` land inside a virama-joined cluster?

    A Bengali morpheme essentially never begins in the middle of a conjunct,
    so an analysis that requires such a cut is almost always the analyser
    over-stripping rather than a real seam it cannot reach. Measured on 80,000
    held-out Wikipedia words, every intra-conjunct "boundary" the inventory
    proposed was a false positive of exactly this kind: `জাতিরাষ্ট্র` and
    `একমাত্র` split before a stem-final `র`, `স্তোত্র` likewise, and `বিশ্বে`
    was read as stem plus a future-tense `বে` rather than `বিশ্ব` plus the
    locative `ে`.

    Rejecting these raises precision, and it also removes the only class of
    morpheme boundary a conjunct-preserving tokenizer genuinely could not
    reach. See `docs/bmbt-morphology.md`.
    """
    if cut <= 0 or cut >= len(word):
        return False
    # Cutting immediately after a virama severs it from the consonant it
    # joins; cutting immediately before one hands the virama to the suffix.
    return word[cut - 1] == substrate.VIRAMA or word[cut] == substrate.VIRAMA


def morph_split(word: str) -> list[Morph]:
    """Segment one word into a stem followed by its suffix chain.

    Suffixes are stripped right to left, longest match first, until no rule
    applies, the chain cap is reached, or stripping would leave a stem shorter
    than the guard allows. A word with no recognised suffix comes back as a
    single "stem" segment, which is the common case and the intended default:
    this layer only claims boundaries it can justify.

    Concatenating the returned segments' `.text` reproduces `word` exactly.

    Raises:
      NormalizationError: if `word` is not a `str`.
    """
    if not isinstance(word, str):
        raise NormalizationError(f"expected str, got {type(word).__name__}")
    if not word:
        return []

    end = len(word)
    found: list[tuple[str, int, int]] = []  # (kind, start, end), outermost first
    stripped_kinds: list[str] = []
    last_rank = -1

    for _ in range(MAX_SUFFIX_CHAIN):
        stem = word[:end]
        match = None
        for suffix, kind in _ORDERED:
            if not stem.endswith(suffix) or len(suffix) >= len(stem):
                continue
            # Ordering: ranks may repeat or increase moving inward, never drop.
            if _RANK[kind] < last_rank:
                continue
            # A verb ending takes nothing nominal outside it.
            if kind == "verb" and any(
                k not in _ALLOWED_OUTSIDE_VERB for k in stripped_kinds
            ):
                continue
            remaining = stem[: len(stem) - len(suffix)]
            # A morpheme does not begin inside a conjunct. An analysis that
            # requires such a cut is over-stripping, not a real seam.
            if cuts_inside_conjunct(word, len(remaining)):
                continue
            floor = _RISKY_MIN_STEM_AKSHARAS if suffix in _RISKY else MIN_STEM_AKSHARAS
            if _akshara_count(remaining) < floor:
                continue
            match = (kind, len(remaining), end)
            break
        if match is None:
            break
        found.append(match)
        stripped_kinds.append(match[0])
        last_rank = _RANK[match[0]]
        end = match[1]

    segments = [Morph(word[:end], "stem", 0, end)] if end else []
    for kind, s, e in reversed(found):
        segments.append(Morph(word[s:e], kind, s, e))
    return segments


def morph_bounds(word: str) -> list[int]:
    """Interior morpheme boundary offsets in `word`, ascending.

    Empty when the word carries no recognised suffix. This is the form BMBT's
    training path wants: it needs to know where it may not merge, not what the
    segments are called.
    """
    segments = morph_split(word)
    return [m.start for m in segments[1:]]


def morph_split_text(text: str) -> list[list[Morph]]:
    """`morph_split` over every whitespace-delimited word in `text`."""
    if not isinstance(text, str):
        raise NormalizationError(f"expected str, got {type(text).__name__}")
    return [morph_split(w) for w in text.split()]


def coverage(words: list[str]) -> dict:
    """How often the inventory fires, and on what.

    Reported rather than assumed: a morphology layer that almost never matches
    is not adding boundaries, and one that matches on nearly everything is
    probably over-stripping. Both failure modes are visible here.
    """
    if not isinstance(words, list):
        raise ConfigError(f"expected a list of words, got {type(words).__name__}")
    by_kind: dict[str, int] = {}
    segmented = 0
    total_suffixes = 0
    for word in words:
        segments = morph_split(word)
        if len(segments) > 1:
            segmented += 1
            for m in segments[1:]:
                by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
                total_suffixes += 1
    return {
        "words": len(words),
        "words_segmented": segmented,
        "segmented_fraction": segmented / len(words) if words else 0.0,
        "suffixes_found": total_suffixes,
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
    }
