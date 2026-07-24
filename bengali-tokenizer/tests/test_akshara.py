r"""
Property/round-trip tests for bntok/akshara.py, the v2 finite-state akshara
parser (roadmap step 3). Mirrors FORMAL_SPEC.md section 7.1's fuzzer input
classes, with per-class assertion scoping: SEG/TOTAL/DET are asserted for
every class (they must hold universally, by construction), but boundary
alignment with `grapheme_clusters()` is asserted only where the two are
actually expected to agree (well-formed, normalized Bengali) - asserting it
everywhere would either be false or accidentally untested on the classes
where it doesn't hold.
"""

from __future__ import annotations

import random
import time

import pytest

from bntok import errors
from bntok.akshara import Akshara, aksharas
from bntok.graphemes import grapheme_clusters
from bntok.normalize import normalize

# --- shared helpers ----------------------------------------------------------


def assert_seg_total_det(text: str) -> list[Akshara]:
    """SEG (lossless segmentation), TOTAL (never raises), DET (repeatable)."""
    chunks = aksharas(text)  # must not raise for any str input (TOTAL)
    assert "".join(c.text for c in chunks) == text  # SEG / LOSSLESS
    for c in chunks:
        assert text[c.start:c.end] == c.text  # offsets are exact, not reconstructed
    assert aksharas(text) == chunks  # DET
    return chunks


def cluster_boundaries(text: str) -> set[int]:
    """Every valid cut point between UAX #29 grapheme clusters in `text`, plus 0 and len(text)."""
    bounds = {0}
    pos = 0
    for cluster in grapheme_clusters(text):
        pos += len(cluster)
        bounds.add(pos)
    return bounds


def assert_aligned_with_grapheme_clusters(chunks: list[Akshara], text: str) -> None:
    bounds = cluster_boundaries(text)
    for c in chunks:
        assert c.start in bounds
        assert c.end in bounds


# --- class 1: valid Bengali, including the design doc's own named hard words -


HARD_WORDS = [
    "স্ত্রী",       # strii, "wife" - the paper's own lead example
    "ক্ষ্ম",        # as in লক্ষ্মী
    "আকাঙ্ক্ষা",   # 3-consonant conjunct chain
    "ঋত্বিক",       # independent vowel + conjunct
]

REAL_SENTENCES = [
    "আমি বাংলায় ক্ষুদ্র গান গাই",
    "রবীন্দ্রনাথের কবিতা",
    "পরীক্ষা",
    "যুক্তাক্ষর ক্ষ জ্ঞ ত্র দ্ধ ঙ্ক্ষ পরীক্ষা করা হচ্ছে",
]


@pytest.mark.parametrize("word", HARD_WORDS)
def test_hard_words_seg_total_det_and_aligned(word):
    chunks = assert_seg_total_det(word)
    assert_aligned_with_grapheme_clusters(chunks, word)
    assert all(c.kind == "akshara" for c in chunks)


@pytest.mark.parametrize("sentence", REAL_SENTENCES)
def test_real_sentences_seg_total_det_and_aligned(sentence):
    chunks = assert_seg_total_det(sentence)
    assert_aligned_with_grapheme_clusters(chunks, sentence)


def test_hard_words_group_into_few_aksharas_not_many_codepoints():
    # The whole point of the grammar: a multi-consonant conjunct is ONE
    # akshara, not one chunk per codepoint.
    chunks = aksharas("স্ত্রী")
    assert len(chunks) == 1
    assert len(chunks[0].text) == 6  # স ্ ত ্ র ী


# --- class 2: variant encodings (nukta composed vs. decomposed) -------------


def test_decomposed_nukta_letter_seg_total_det_without_normalization():
    # ড় spelled as its NFC-exclusion decomposed pair (DDA + NUKTA), fed
    # directly, NOT pre-normalized. SEG/TOTAL/DET must hold regardless;
    # boundary-alignment with grapheme_clusters() is NOT asserted here on
    # purpose (see module docstring) since normalize() is a separate stage
    # this parser deliberately does not run internally.
    decomposed = chr(0x09A1) + chr(0x09BC)  # DDA + NUKTA, not the RRA singleton
    text = decomposed + "া"  # + AA-matra, e.g. as in "বড়া"-like sequences
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"


def test_decomposed_nukta_letter_after_normalize_is_aligned():
    decomposed = chr(0x09A1) + chr(0x09BC)
    normalized = normalize(decomposed)
    assert normalized == chr(0x09DC)  # normalize() recomposes to the RRA singleton
    chunks = assert_seg_total_det(normalized)
    assert_aligned_with_grapheme_clusters(chunks, normalized)


def test_precomposed_rra_rha_yya_seg_total_det():
    for cp in (0x09DC, 0x09DD, 0x09DF):
        assert_seg_total_det(chr(cp) + "া")


# --- class 3: mixed script / Banglish ----------------------------------------


BANGLISH = [
    "আমি busy আছি",
    "কি খবর? Hello World 123 ঠিক আছে",
    "আমার ফোন নাম্বার +৮৮০-১২৩৪৫৬৭৮৯",
]


@pytest.mark.parametrize("text", BANGLISH)
def test_banglish_seg_total_det(text):
    assert_seg_total_det(text)


def test_banglish_english_words_fall_to_other_kind():
    chunks = aksharas("আমি busy আছি")
    other_texts = [c.text for c in chunks if c.kind == "other"]
    assert "b" in other_texts  # English falls through one grapheme cluster (= 1 ASCII char) at a time


# --- class 4: adversarial Unicode ---------------------------------------------
#
# The cases below (vowel+virama non-chaining, matra not blocking a conjunct
# chain, modifier blocking one, repeated/reordered matra and modifier) were
# not hypothesized - they were found by running the parser against real
# Wikipedia held-out text (v2 roadmap step 4's own measurement surfaced 5
# lines with a genuine boundary-inside-cluster bug), then verified one at a
# time against regex's own \\X before the grammar in akshara.py was fixed.


def test_vowel_virama_does_not_chain_into_a_further_consonant():
    # regex's \\X clusters an independent vowel + virama as its own pair, and
    # does NOT let it continue into a following consonant the way a
    # consonant-initiated virama does (Unicode's Indic_Conjunct_Break rule
    # requires InCB=Consonant on both sides; vowels are not InCB=Consonant).
    a, virama, ta = chr(0x0985), chr(0x09CD), chr(0x09A4)
    text = a + virama + ta
    chunks = assert_seg_total_det(text)
    assert [c.kind for c in chunks] == ["akshara", "akshara"]
    assert [(c.start, c.end) for c in chunks] == [(0, 2), (2, 3)]
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_matra_does_not_block_conjunct_chain_continuation():
    # Consonant+matra+virama+consonant clusters as ONE unit in \\X: a matra
    # between the first consonant and the continuing virama does not break
    # chain eligibility, unlike a modifier (see the next test).
    ka, aa_matra, virama, ta = chr(0x0995), chr(0x09BE), chr(0x09CD), chr(0x09A4)
    text = ka + aa_matra + virama + ta
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_modifier_blocks_conjunct_chain_continuation():
    # A modifier (anusvara here) between the first consonant and a following
    # virama DOES break chain eligibility in \\X, unlike matra/nukta: the
    # modifier+virama are absorbed into the first chunk (both are generic
    # Extend characters), but the chain does not continue to the next
    # consonant, which starts its own chunk instead.
    ka, anusvara, virama, ta = chr(0x0995), chr(0x0982), chr(0x09CD), chr(0x09A4)
    text = ka + anusvara + virama + ta
    chunks = assert_seg_total_det(text)
    assert [c.kind for c in chunks] == ["akshara", "akshara"]
    assert [(c.start, c.end) for c in chunks] == [(0, 3), (3, 4)]
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_modifier_before_matra_unusual_order_still_clusters_as_one():
    # Linguistically backwards (a modifier before its matra), but \\X still
    # absorbs both into one cluster regardless of order.
    ka, anusvara, aa_matra = chr(0x0995), chr(0x0982), chr(0x09BE)
    text = ka + anusvara + aa_matra
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_repeated_matra_absorbed_like_repeated_nukta():
    # Malformed (a matra doesn't normally repeat), found for real in
    # Wikipedia held-out text: base consonant followed by the same matra
    # several times in a row. \\X absorbs all of them into one cluster.
    da, u_matra = chr(0x09A6), chr(0x09C1)
    text = da + u_matra * 4
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_vowel_nukta_absorbed_even_though_not_real_orthography():
    # Not real Bengali (nukta modifies consonants), but \\X absorbs any
    # Extend-class character generically, vowels included.
    text = chr(0x0985) + chr(0x09BC)
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_dangling_hasanta_seg_total_det():
    # Consonant + virama with nothing after: an explicit/dangling hasanta.
    text = chr(0x0995) + chr(0x09CD)  # ক + ্
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"


def test_lone_orphan_matra_falls_to_other():
    # A matra with no preceding consonant: does not start either grammar
    # branch, so it correctly falls to the "other" bucket rather than being
    # silently dropped or crashing.
    text = chr(0x09BF)  # ি alone
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "other"


def test_zwj_continues_conjunct_zwnj_terminates_it():
    # Empirically verified against regex's own \\X: ZWJ after a virama
    # continues the conjunct (one cluster); ZWNJ after a virama terminates
    # it explicitly (two clusters). The grammar must reproduce this exactly,
    # not just "not crash" - these are two different, correct segmentations.
    ka, virama, ta = chr(0x0995), chr(0x09CD), chr(0x09A4)

    zwj_text = ka + virama + chr(0x200D) + ta
    chunks = assert_seg_total_det(zwj_text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, zwj_text)

    zwnj_text = ka + virama + chr(0x200C) + ta
    chunks = assert_seg_total_det(zwnj_text)
    assert len(chunks) == 2
    assert [c.kind for c in chunks] == ["akshara", "akshara"]
    assert_aligned_with_grapheme_clusters(chunks, zwnj_text)


def test_zwj_before_virama_also_continues_the_conjunct():
    # Found for real in Wikipedia held-out text: ZWJ is not tied to a fixed
    # position relative to the virama. "Consonant ZWJ Virama Consonant"
    # clusters as one continuing conjunct in \\X exactly like
    # "Consonant Virama ZWJ Consonant" above.
    ka, zwj, virama, ta = chr(0x0995), chr(0x200D), chr(0x09CD), chr(0x09A4)
    text = ka + zwj + virama + ta
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_zwnj_before_virama_blocks_chain_but_is_still_absorbed():
    # Found for real in Wikipedia held-out text: a ZWNJ before the virama
    # (not just after) still blocks chain continuation to a following
    # consonant, but the ZWNJ+virama themselves are absorbed into the
    # current chunk either way (verified both with and without a following
    # consonant).
    na, zwnj, virama, ta = chr(0x09A8), chr(0x200C), chr(0x09CD), chr(0x09A4)

    no_following_consonant = na + zwnj + virama
    chunks = assert_seg_total_det(no_following_consonant)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, no_following_consonant)

    with_following_consonant = na + zwnj + virama + ta
    chunks = assert_seg_total_det(with_following_consonant)
    assert len(chunks) == 2
    assert [c.kind for c in chunks] == ["akshara", "akshara"]
    assert_aligned_with_grapheme_clusters(chunks, with_following_consonant)


def test_double_nukta_combining_overflow_seg_total_det():
    # Malformed (nukta doesn't normally repeat), but Nukta* must absorb it
    # without crashing, matching \\X's own behaviour of clustering even a
    # doubled nukta with its base consonant.
    text = chr(0x0995) + chr(0x09BC) + chr(0x09BC)
    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"
    assert_aligned_with_grapheme_clusters(chunks, text)


def test_khanda_ta_virama_consonant_is_a_documented_divergence_from_uax29():
    # Deliberate, documented divergence (see akshara.py's module docstring
    # and docs/known-issues.md): khanda-ta is an ordinary consonant in this
    # grammar, so it chains into a further conjunct here, producing ONE
    # akshara for all 3 codepoints. regex's \\X does NOT let khanda-ta chain
    # this way and produces TWO clusters instead. Both are asserted
    # explicitly, not silently matched or silently skipped.
    khanda_ta, virama, ka = chr(0x09CE), chr(0x09CD), chr(0x0995)
    text = khanda_ta + virama + ka

    chunks = assert_seg_total_det(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"

    assert grapheme_clusters(text) == [khanda_ta + virama, ka]
    assert len(grapheme_clusters(text)) == 2  # UAX #29 disagrees with the grammar here, on purpose


def test_many_dangling_viramas_in_a_row_seg_total_det():
    text = chr(0x09CD) * 5  # virama with no preceding consonant, repeated
    assert_seg_total_det(text)


# --- class 5: non-Bengali ------------------------------------------------------


NON_BENGALI = [
    "Hello, World! 123",
    "日本語のテキスト",
    "\U0001F600\U0001F44D",  # emoji
    "\U0001F468‍\U0001F469‍\U0001F467",  # ZWJ emoji family sequence
    "Ελληνικά",
    "العربية",
]


@pytest.mark.parametrize("text", NON_BENGALI)
def test_non_bengali_seg_total_det(text):
    assert_seg_total_det(text)


def test_emoji_zwj_sequence_falls_to_one_other_chunk_not_three():
    # The "other" fallback advances by one grapheme cluster, not one raw
    # codepoint, so a ZWJ emoji family sequence is one chunk, not five.
    text = "\U0001F468‍\U0001F469‍\U0001F467"
    chunks = aksharas(text)
    assert len(chunks) == 1
    assert chunks[0].kind == "other"
    assert chunks[0].text == text


# --- class 6: corrupted / random codepoints (fixed seed, deterministic) ------


def _random_unicode_strings(seed: int, count: int, max_len: int) -> list[str]:
    rng = random.Random(seed)
    strings = []
    for _ in range(count):
        length = rng.randint(0, max_len)
        chars = []
        for _ in range(length):
            cp = rng.randint(0x20, 0x10FFFF)
            while 0xD800 <= cp <= 0xDFFF:  # surrogates are not valid scalar values
                cp = rng.randint(0x20, 0x10FFFF)
            chars.append(chr(cp))
        strings.append("".join(chars))
    return strings


@pytest.mark.parametrize("text", _random_unicode_strings(seed=20260725, count=30, max_len=40))
def test_random_unicode_seg_total_det(text):
    assert_seg_total_det(text)


# --- class 7: degenerate -------------------------------------------------------


DEGENERATE = ["", " ", "\n", "\t", "a", "ক", "   \n\t  "]


@pytest.mark.parametrize("text", DEGENERATE)
def test_degenerate_seg_total_det(text):
    assert_seg_total_det(text)


def test_empty_string_returns_no_chunks():
    assert aksharas("") == []


# --- totality / error contract -------------------------------------------------


def test_non_str_input_raises_normalization_error():
    for bad in (None, 123, b"bytes", ["a", "b"]):
        with pytest.raises(errors.NormalizationError):
            aksharas(bad)


# --- linearity sanity check (catches accidental O(n^2) re-slicing) ------------


def test_linear_time_on_a_large_deeply_conjunct_input():
    # A single ~100k-codepoint conjunct chain is the worst case for the
    # (Virama ZWJ? Consonant Nukta*)* loop: if the implementation ever
    # re-slices `text[pos:]` per step instead of using fixed offsets, this
    # degrades to O(n^2) and the generous bound below fails. This is a
    # correctness-via-performance guard, not a benchmark.
    chain = chr(0x0995) + (chr(0x09CD) + chr(0x0995)) * 50_000  # ~100,001 codepoints
    start = time.perf_counter()
    chunks = aksharas(chain)
    elapsed = time.perf_counter() - start
    assert "".join(c.text for c in chunks) == chain
    assert len(chunks) == 1  # the whole chain is one unbounded conjunct
    assert elapsed < 5.0


def test_linear_time_on_a_large_repeated_sentence():
    sentence = "আমি বাংলায় ক্ষুদ্র গান গাই। "
    text = sentence * 4000  # roughly 100k+ codepoints of realistic mixed content
    start = time.perf_counter()
    chunks = aksharas(text)
    elapsed = time.perf_counter() - start
    assert "".join(c.text for c in chunks) == text
    assert elapsed < 5.0
