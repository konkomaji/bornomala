"""Tests for the graded fragmentation measure.

Each grade is defined by a structural test rather than a severity weight, so
these tests pin the structure, not a judgement.
"""

import pytest

from bntok.fragmentation import (
    DESTRUCTIVE,
    MODIFIER,
    ONSET_RIME,
    classify_split,
    count_splits,
)
from bntok.normalize import normalize

VIRAMA = "্"
NUKTA = "়"


def test_stranded_virama_is_destructive():
    """`ক্ষ` -> `ক্` + `ষ` leaves a virama with nothing to join."""
    assert classify_split(normalize("ক") + VIRAMA, normalize("ষ")) == DESTRUCTIVE
    # ...and in the other direction, where the virama opens the second piece.
    assert classify_split(normalize("ক"), VIRAMA + normalize("ষ")) == DESTRUCTIVE


def test_detached_nukta_is_destructive():
    """`ড` + `়` is not a damaged cluster, it is a different letter.

    NFC keeps these decomposed: the composed forms are a permanent Unicode
    composition exclusion, which this project documented while fixing an
    unrelated normalisation bug.
    """
    assert classify_split(normalize("ড"), NUKTA) == DESTRUCTIVE


def test_onset_rime_seam_is_onset_rime():
    """`শ্ব` + `ে` yields a valid consonant cluster and a valid vowel sign."""
    onset = normalize("শ") + VIRAMA + normalize("ব")
    assert classify_split(onset, "ে") == ONSET_RIME


def test_detached_modifier_is_modifier():
    """Anusvara, visarga and chandrabindu are separate phonemes."""
    for modifier in ("ং", "ঃ", "ঁ"):
        assert classify_split(normalize("ক"), modifier) == MODIFIER


def test_unclassified_split_defaults_to_destructive():
    """An unrecognised severing is not evidence of a harmless one."""
    assert classify_split(normalize("ক"), normalize("খ")) == DESTRUCTIVE


def test_empty_side_severs_nothing():
    assert classify_split("", normalize("ক")) == ONSET_RIME
    assert classify_split(normalize("ক"), "") == ONSET_RIME


# --- counting -------------------------------------------------------------

def test_unsplit_text_scores_zero():
    text = normalize("আমি বাংলায় গান গাই")
    counts = count_splits([text], text)
    assert counts.total_splits == 0
    assert counts.destructive_rate == 0.0


def test_denominator_excludes_unsplittable_clusters():
    """A bare consonant cannot be split, so it must not pad the denominator.

    Measured on 697,048 held-out clusters, 61.0% are a single codepoint. The
    legacy metric divides by all of them, which inflates every tokenizer's
    apparent quality by 2.56x.
    """
    text = normalize("কখগ")  # three single-codepoint clusters
    counts = count_splits([text], text)
    assert counts.total_clusters == 3
    assert counts.splittable_clusters == 0
    # No division by zero, and no credit claimed for the impossible.
    assert counts.destructive_rate == 0.0


def test_splittable_count_is_multi_codepoint_clusters():
    text = normalize("কে")  # one two-codepoint cluster
    counts = count_splits([text], text)
    assert counts.total_clusters == 1
    assert counts.splittable_clusters == 1


def test_destructive_split_is_counted_and_rated():
    whole = normalize("ক্ষ")
    surfaces = [whole[:2], whole[2:]]  # ক + virama | ষ
    assert "".join(surfaces) == whole
    counts = count_splits(surfaces, whole)
    assert counts.destructive == 1
    assert counts.onset_rime == 0
    assert counts.splittable_clusters == 1
    assert counts.destructive_rate == 1.0


def test_onset_rime_split_does_not_count_as_destructive():
    whole = normalize("শ্বে")
    cut = whole.index("ে")
    surfaces = [whole[:cut], whole[cut:]]
    assert "".join(surfaces) == whole
    counts = count_splits(surfaces, whole)
    assert counts.destructive == 0
    assert counts.onset_rime == 1
    assert counts.destructive_rate == 0.0
    assert counts.any_split_rate == 1.0


def test_boundary_between_separate_clusters_is_not_a_split():
    """Splitting between two whole clusters severs nothing."""
    a, b = normalize("কে"), normalize("খা")
    counts = count_splits([a, b], a + b)
    assert counts.total_splits == 0


def test_as_dict_exposes_derived_rates():
    whole = normalize("ক্ষ")
    d = count_splits([whole[:2], whole[2:]], whole).as_dict()
    assert d["destructive"] == 1
    assert d["destructive_rate"] == pytest.approx(1.0)
    assert "any_split_rate" in d
