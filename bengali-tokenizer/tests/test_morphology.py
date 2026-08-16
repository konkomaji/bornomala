"""Tests for the Bengali inflectional morphology layer.

Every "must not split" case below is at least as important as the positive
ones. A false morpheme boundary is worse than a missed one: it invents
structure that is not there, and morphological-alignment scoring counts it
against us.
"""

import pytest

from bntok.errors import NormalizationError
from bntok.morphology import (
    MAX_SUFFIX_CHAIN,
    coverage,
    morph_bounds,
    morph_split,
)
from bntok.normalize import normalize

# (word, expected stem, expected suffix kinds outermost-first)
SPLITS = [
    ("বাংলায়", "বাংলা", ["case"]),
    ("ছেলেরা", "ছেলে", ["plural"]),
    ("ছেলেদের", "ছেলে", ["plural"]),
    ("বইগুলো", "বই", ["plural"]),
    ("বইটি", "বই", ["classifier"]),
    ("মানুষকে", "মানুষ", ["case"]),
    ("দেশের", "দেশ", ["case"]),
    ("স্বাধীনতা", "স্বাধীন", ["derivational"]),
    ("ঐতিহাসিক", "ঐতিহাস", ["derivational"]),
    ("করছি", "কর", ["verb"]),
    ("করেছিলাম", "কর", ["verb"]),
    ("যাবেন", "যা", ["verb"]),
    ("বলতাম", "বল", ["verb"]),
]

# Words with no recognised suffix: the whole word is the stem.
NO_SPLIT = ["বই", "নদী", "জল", "ঘর", "মা"]


@pytest.mark.parametrize("word,stem,kinds", SPLITS)
def test_known_inflected_forms(word, stem, kinds):
    segments = morph_split(normalize(word))
    assert segments[0].text == normalize(stem)
    assert segments[0].kind == "stem"
    assert [m.kind for m in segments[1:]] == kinds


@pytest.mark.parametrize("word", NO_SPLIT)
def test_uninflected_words_are_one_stem(word):
    segments = morph_split(normalize(word))
    assert len(segments) == 1
    assert segments[0].kind == "stem"


@pytest.mark.parametrize("word,stem,_kinds", SPLITS)
def test_lossless(word, stem, _kinds):
    w = normalize(word)
    assert "".join(m.text for m in morph_split(w)) == w


@pytest.mark.parametrize("word", NO_SPLIT + [w for w, _, _ in SPLITS])
def test_offsets_match_surface(word):
    w = normalize(word)
    for m in morph_split(w):
        assert w[m.start:m.end] == m.text


def test_one_akshara_verb_root_is_reachable():
    """Regression: `যা` and `দে` are ONE akshara, not two.

    A two-akshara minimum-stem floor looked safe and was not: it refused
    `যা` + `বেন` and produced `যাব` + `েন` instead, inventing a boundary in
    the middle of the future-tense marker. Found by running the layer against
    real forms, not by unit tests written from the same assumption.
    """
    segments = morph_split(normalize("যাবেন"))
    assert [m.text for m in segments] == [normalize("যা"), normalize("বেন")]


def test_verb_ending_takes_no_plural():
    """Regression: `ছেলেরা` must not become `ছে` + `লে`[verb] + `রা`[plural].

    `লে` is a real past-tense ending and `ছেলে` really does end with those
    codepoints, so length guards alone cannot separate them. The ordering
    constraint rejects it grammatically: a finite verb cannot be pluralised.
    """
    segments = morph_split(normalize("ছেলেরা"))
    assert [m.kind for m in segments] == ["stem", "plural"]
    assert "verb" not in [m.kind for m in segments]


def test_suffix_ranks_never_decrease_moving_inward():
    from bntok.morphology import _RANK

    for word, _, _ in SPLITS:
        kinds = [m.kind for m in morph_split(normalize(word))[1:]]
        ranks = [_RANK[k] for k in kinds]
        assert ranks == sorted(ranks), f"{word}: {kinds} out of order"


def test_chain_is_capped():
    for word, _, _ in SPLITS:
        assert len(morph_split(normalize(word))) - 1 <= MAX_SUFFIX_CHAIN


def test_morph_bounds_matches_split():
    for word, _, _ in SPLITS:
        w = normalize(word)
        assert morph_bounds(w) == [m.start for m in morph_split(w)[1:]]
    for word in NO_SPLIT:
        assert morph_bounds(normalize(word)) == []


def test_empty_and_type_guards():
    assert morph_split("") == []
    assert morph_bounds("") == []
    with pytest.raises(NormalizationError):
        morph_split(None)
    with pytest.raises(NormalizationError):
        morph_split(5)


def test_coverage_reports_shape():
    words = [normalize(w) for w, _, _ in SPLITS] + [normalize(w) for w in NO_SPLIT]
    report = coverage(words)
    assert report["words"] == len(words)
    assert report["words_segmented"] == len(SPLITS)
    assert 0.0 < report["segmented_fraction"] < 1.0
    assert set(report["by_kind"]) <= {
        "clitic", "case", "plural", "classifier", "verb", "derivational",
    }


def test_known_limitation_proper_noun_over_strips():
    """`কলকাতা` is wrongly split into `কলকা` + `তা`[derivational].

    Kept as a passing test that asserts the CURRENT behaviour, so that the
    limitation is visible in the suite rather than buried in prose, and so
    that a future stem lexicon shows up here as a deliberate change rather
    than a surprise. Documented in the module docstring and known-issues.md.
    """
    segments = morph_split(normalize("কলকাতা"))
    assert len(segments) == 2
    assert segments[1].kind == "derivational"
