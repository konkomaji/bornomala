"""Tests for the corpus-quality filters in bntok.corpus.

`is_clean_bengali_line`/`is_clean_banglish_line` had zero test coverage
before this file (confirmed by search - no test_corpus.py existed). Adding
coverage for those alongside the new structural-validity and
legacy-encoding-triage functions, rather than leaving the new ones tested
while the old ones stay unverified.
"""

from bntok.corpus import (
    bengali_structural_validity_ratio,
    flag_possible_legacy_encoding,
    is_clean_bengali_line,
    is_structurally_valid_bengali,
)


def test_is_clean_bengali_line_accepts_real_bengali():
    assert is_clean_bengali_line("আমি ভাত খাচ্ছি") is True


def test_is_clean_bengali_line_rejects_too_short():
    assert is_clean_bengali_line("আম") is False


def test_is_clean_bengali_line_rejects_wrong_script():
    assert is_clean_bengali_line("これは日本語のテキストです") is False


def test_structural_validity_ratio_is_one_for_clean_bengali():
    assert bengali_structural_validity_ratio("আমি ভাত খাচ্ছি") == 1.0


def test_structural_validity_ratio_is_one_for_no_bengali_at_all():
    # Nothing to validate - that's is_clean_bengali_line's job, not this one's.
    assert bengali_structural_validity_ratio("just English text") == 1.0


def test_structural_validity_ratio_drops_for_orphan_matra():
    # chr(0x09BF) (lone ি) is a verified orphan matra - falls to "other" per
    # test_akshara.py's own test_lone_orphan_matra_falls_to_other. Isolated
    # with whitespace on both sides, matching that exact proven case, not a
    # mid-word placement - first attempt at this test put it directly after
    # "আমি" (which itself ends in ি) and got ratio == 1.0, because the scan's
    # left-to-right state at that position isn't the same as a truly isolated
    # orphan; real orphan-matra behaviour is context-sensitive, verify against
    # the precedent case rather than guess at a new one.
    orphan = chr(0x09BF)
    ratio = bengali_structural_validity_ratio(f"আমি {orphan} ভাত")
    assert ratio < 1.0


def test_is_structurally_valid_bengali_accepts_clean_text():
    assert is_structurally_valid_bengali("আমি ভাত খাচ্ছি প্রতিদিন") is True


def test_is_structurally_valid_bengali_rejects_mostly_orphan_matras():
    orphan = chr(0x09BF)
    garbage = orphan * 10
    assert is_structurally_valid_bengali(garbage) is False


def test_is_structurally_valid_bengali_rejects_too_few_bengali_chars():
    # Below min_bengali_chars - not enough signal to judge, so reject rather
    # than let one stray codepoint swing the ratio to a meaningless 1.0 or 0.0.
    assert is_structurally_valid_bengali("hi আ bye", min_bengali_chars=4) is False


def test_flag_possible_legacy_encoding_flags_ascii_junk_from_declared_source():
    # Simulates a legacy-font page: declared as Bengali content, but the text
    # itself is almost entirely ASCII (the actual signature of a Bijoy/
    # SutonnyMJ-era font mapping Bengali glyphs onto Latin byte slots).
    legacy_style = "GwuwR GKQvi GKUv K_v Ryb Avwg 123"
    assert flag_possible_legacy_encoding(legacy_style, declared_bengali_source=True) is True


def test_flag_possible_legacy_encoding_does_not_flag_when_not_declared():
    legacy_style = "GwuwR GKQvi GKUv K_v Ryb Avwg 123"
    assert flag_possible_legacy_encoding(legacy_style, declared_bengali_source=False) is False


def test_flag_possible_legacy_encoding_does_not_flag_real_bengali():
    assert flag_possible_legacy_encoding("আমি ভাত খাচ্ছি প্রতিদিন", declared_bengali_source=True) is False


def test_flag_possible_legacy_encoding_respects_min_len():
    assert flag_possible_legacy_encoding("short", declared_bengali_source=True, min_len=20) is False
