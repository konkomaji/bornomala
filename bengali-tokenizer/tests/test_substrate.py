r"""
Regression tripwire for bntok/substrate.py.

Independently re-derives the Bengali Unicode ranges rather than trusting the
import machinery: a typo'd hex constant in substrate.py would not necessarily
break any behavioral test in test_tokenizer.py, since those only exercise a
handful of common words. This pins the actual codepoint sets.
"""

from __future__ import annotations

from bntok import substrate


def test_consonants_range_and_extras():
    # 0x0995..0x09B9 inclusive (37) + RRA/RHA/YYA/khanda-ta (4) = 41.
    expected = {chr(c) for c in range(0x0995, 0x09B9 + 1)} | {
        chr(0x09DC), chr(0x09DD), chr(0x09DF), chr(0x09CE),
    }
    assert substrate.CONSONANTS == expected
    assert len(substrate.CONSONANTS) == 41


def test_vowels_range_and_extras():
    # 0x0985..0x098C inclusive (8, includes the rare/archaic VOCALIC L) + E/AI/O/AU (4) = 12.
    expected = {chr(c) for c in range(0x0985, 0x098C + 1)} | {
        chr(0x098F), chr(0x0990), chr(0x0993), chr(0x0994),
    }
    assert substrate.VOWELS == expected
    assert len(substrate.VOWELS) == 12


def test_matras_range_and_extra():
    # 0x09BE..0x09CC inclusive (15, includes 4 Unicode-unassigned codepoints
    # inside the range) + AU LENGTH MARK (1) = 16.
    expected = {chr(c) for c in range(0x09BE, 0x09CC + 1)} | {chr(0x09D7)}
    assert substrate.MATRAS == expected
    assert len(substrate.MATRAS) == 16


def test_modifiers():
    assert substrate.MODIFIERS == {chr(0x0981), chr(0x0982), chr(0x0983)}


def test_primitives_are_the_expected_codepoints():
    assert substrate.VIRAMA == chr(0x09CD)
    assert substrate.NUKTA == chr(0x09BC)
    assert substrate.RA == chr(0x09B0)
    assert substrate.YA == chr(0x09AF)
    assert substrate.ZWJ == chr(0x200D)
    assert substrate.ZWNJ == chr(0x200C)
    assert substrate.DANDA == chr(0x0964)
    assert substrate.DOUBLE_DANDA == chr(0x0965)


def test_danda_is_outside_bengali_block_but_in_guaranteed_coverage():
    assert substrate.DANDA not in substrate.BENGALI_BLOCK
    assert substrate.DOUBLE_DANDA not in substrate.BENGALI_BLOCK
    assert substrate.DANDA in substrate.GUARANTEED_CODEPOINTS
    assert substrate.DOUBLE_DANDA in substrate.GUARANTEED_CODEPOINTS


def test_the_four_grammar_classes_are_pairwise_disjoint():
    classes = [substrate.CONSONANTS, substrate.VOWELS, substrate.MATRAS, substrate.MODIFIERS]
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert a.isdisjoint(b)


def test_matra_position_covers_every_matra_with_a_known_value():
    assert set(substrate.MATRA_POSITION) == substrate.MATRAS
    assert set(substrate.MATRA_POSITION.values()) <= {"before", "after", "below", "split", "unassigned"}


def test_rra_rha_yya_are_precomposed_singletons_not_decomposed_pairs():
    # Regression guard for the exact silent-decomposition trap normalize.py's
    # own docstring warns about: typing RRA/RHA/YYA as literal characters can
    # round-trip through an editor/tool pipeline that re-decomposes them onto
    # base+nukta. It happened once while writing substrate.py itself.
    for cp in (0x09DC, 0x09DD, 0x09DF):
        ch = chr(cp)
        assert ch in substrate.CONSONANTS
        assert len(ch) == 1
