"""Tests for bntok.dedup (Track A2, Gate G3 pipeline)."""

import pytest

from bntok import errors
from bntok.dedup import (
    _is_low_quality,
    _shingles,
    exact_dedup,
    near_dedup,
    quality_filter,
    survival_report,
)

CLEAN_BENGALI = [
    "আমি বাংলায় গান গাই আমি বাংলার গান গাই",
    "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে",
    "বাংলা আমার মাতৃভাষা আমি বাংলায় কথা বলি এবং লিখি",
    "পশ্চিমবঙ্গের উপভাষা নিয়ে গবেষণা জরুরি ও গুরুত্বপূর্ণ",
    "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের শ্রেষ্ঠ কবি ছিলেন",
]


# --- exact_dedup ---

def test_exact_dedup_removes_byte_identical_repeats():
    lines = [CLEAN_BENGALI[0], CLEAN_BENGALI[1], CLEAN_BENGALI[0], CLEAN_BENGALI[0]]
    out, removed = exact_dedup(lines)
    assert out == [CLEAN_BENGALI[0], CLEAN_BENGALI[1]]
    assert removed == 2


def test_exact_dedup_keeps_first_occurrence_order():
    lines = ["c", "a", "b", "a", "c"]
    out, removed = exact_dedup(lines)
    assert out == ["c", "a", "b"]
    assert removed == 2


def test_exact_dedup_empty_input():
    out, removed = exact_dedup([])
    assert out == []
    assert removed == 0


# --- near_dedup ---

def test_near_dedup_catches_minor_edit():
    base = CLEAN_BENGALI[1]
    near_identical = base + " ।"  # trailing punctuation added, shingles barely change
    out, removed = near_dedup([base, near_identical] + CLEAN_BENGALI[2:], threshold=0.5)
    assert removed >= 1
    assert base in out


def test_near_dedup_keeps_genuinely_different_lines():
    out, removed = near_dedup(list(CLEAN_BENGALI), threshold=0.8)
    assert removed == 0
    assert out == CLEAN_BENGALI


def test_near_dedup_short_lines_get_a_signature_not_skipped():
    # fewer words than shingle_size=5: _shingles falls back to whole-line
    out, removed = near_dedup(["ছোট বাক্য", "ছোট বাক্য"], threshold=0.5)
    assert len(out) == 1
    assert removed == 1


def test_near_dedup_invalid_threshold_raises():
    with pytest.raises(errors.ConfigError):
        near_dedup(CLEAN_BENGALI, threshold=0.0)
    with pytest.raises(errors.ConfigError):
        near_dedup(CLEAN_BENGALI, threshold=1.5)


def test_shingles_short_line_fallback():
    assert _shingles("এক দুই") == {"এক দুই"}
    assert _shingles("") == set()


def test_shingles_normal_line():
    s = _shingles("a b c d e f", n=5)
    assert s == {"a b c d e", "b c d e f"}


# --- quality_filter / _is_low_quality ---

def test_quality_filter_keeps_clean_bengali():
    out, removed = quality_filter(CLEAN_BENGALI)
    assert out == CLEAN_BENGALI
    assert removed == 0


def test_quality_filter_rejects_short_and_foreign_script():
    junk = ["a", "こんにちは世界", "12345678901234"]
    out, removed = quality_filter(junk)
    assert out == []
    assert removed == len(junk)


def test_is_low_quality_digit_dominated():
    assert _is_low_quality("০১২৩৪৫৬৭৮৯ পাতা ১২৩")


def test_is_low_quality_repeated_character():
    assert _is_low_quality("বাংলা ============================")


def test_is_low_quality_false_for_clean_line():
    assert not _is_low_quality(CLEAN_BENGALI[0])


def test_is_low_quality_empty_line():
    assert _is_low_quality("")


# --- survival_report ---

def test_survival_report_end_to_end():
    raw = CLEAN_BENGALI + [CLEAN_BENGALI[0]] + ["===", "12345", "x"]
    report = survival_report(raw)
    assert report["raw_lines"] == len(raw)
    assert report["surviving_lines"] <= report["raw_lines"]
    assert report["removed_exact_dup"] == 1  # the duplicated CLEAN_BENGALI[0]
    assert 0.0 <= report["survival_ratio_lines"] <= 1.0
    assert 0.0 <= report["survival_ratio_words"] <= 1.0
    assert report["surviving_lines"] == len(CLEAN_BENGALI)  # junk + the exact dup all removed


def test_survival_report_empty_raises():
    with pytest.raises(errors.ConfigError):
        survival_report([])


def test_survival_report_all_junk_survives_nothing():
    report = survival_report(["===", "12345", "x", "==="])
    assert report["surviving_lines"] == 0
    assert report["survival_ratio_lines"] == 0.0
    assert report["survival_ratio_words"] == 0.0
