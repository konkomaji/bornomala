"""The vectorized backend must be indistinguishable from the scalar scan.

`akshara_vec` exists purely for speed, so every test here is differential: the
scalar `_scan` in `akshara.py` is the oracle, and the only acceptable result is
byte-identical boundaries. Cases that the fast path is NOT expected to handle
(other scripts' conjuncts, emoji, Hangul, CRLF) are tested too - not to check
that it segments them correctly, but to check that it *declines* them and lets
the scalar scan answer.
"""

import pytest

from bntok.akshara import akshara_bounds, akshara_bounds_batch, aksharas
from bntok.akshara_vec import available, bounds_batch, is_vectorizable
from bntok.normalize import normalize

pytestmark = pytest.mark.skipif(not available(), reason="numpy not installed")


BENGALI = [
    "আমি বাংলায় গান গাই",
    "রবীন্দ্রনাথ ঠাকুর পশ্চিমবঙ্গের বিশিষ্ট সাহিত্যিক।",
    "আকাঙ্ক্ষা স্বাধীনতা বিজ্ঞান কৃষ্ণ ঋত্বিক স্ত্রী",
    "মুর্শিদাবাদ বিষ্ণুপুর শান্তিনিকেতন দার্জিলিং বর্ধমান",
    "ভাষা ও সাহিত্যের ইতিহাস অত্যন্ত সমৃদ্ধ এবং বৈচিত্র্যময়।",
    "১৯৭১ সালে ২৬শে মার্চ",           # digits
    "Bengali (বাংলা) is spoken by 230 million people.",  # code-mixed ASCII
    "ক্ষ ঙ্ক্ষ র্ধ",   # explicit viramas
    "ক‍ষ ক‌ষ",           # ZWJ / ZWNJ, escaped: invisible in source
    "অ্ক",                        # vowel + virama: must NOT chain
    "কঁ্ক",                       # modifier blocks chaining
    "",
    " ",
    "\n\t",
    "ক",
    "্",                          # orphan virama
    "া",                          # orphan matra
    "café",                       # precomposed Latin-1 letter: a plain singleton
    "text\u200bwith zwsp",        # ZWSP is GCB=Control, so a singleton to \X too
    "“quoted” – dash",  # General Punctuation
]

# Inputs the guard must REJECT, each for a documented reason.
NOT_VECTORIZABLE = [
    "সংস্কৃত धर्म",           # Devanagari conjunct: another script's virama
    "अञ्चल ভাগ",
    "line\r\nbreak",           # CRLF is one cluster to \X
    "flag 🇮🇳 here",           # regional indicator pair
    "family 👨‍👩‍👧",   # emoji ZWJ sequence
    "한국어 텍스트",            # Hangul
]


def _oracle(text):
    return [c.end for c in aksharas(text)]


@pytest.mark.parametrize("text", BENGALI)
def test_batch_matches_scalar(text):
    text = normalize(text)
    assert akshara_bounds_batch([text]) == [akshara_bounds(text)]
    assert akshara_bounds_batch([text]) == [_oracle(text)]


def test_batch_matches_scalar_all_at_once():
    texts = [normalize(t) for t in BENGALI]
    assert akshara_bounds_batch(texts) == [akshara_bounds(t) for t in texts]


def test_batch_join_does_not_leak_across_strings():
    """A chunk must never span two inputs, however they are grouped."""
    texts = [normalize(t) for t in BENGALI if t]
    one_at_a_time = [akshara_bounds(t) for t in texts]
    assert akshara_bounds_batch(texts) == one_at_a_time
    # Adjacent strings that would chain if concatenated without a separator.
    assert akshara_bounds_batch(["ক্", "ক"]) == [
        akshara_bounds("ক্"),
        akshara_bounds("ক"),
    ]


@pytest.mark.parametrize("text", NOT_VECTORIZABLE)
def test_guard_rejects_and_result_is_still_correct(text):
    assert not is_vectorizable(text)
    assert bounds_batch([text]) == [None]  # declined, not answered wrongly
    # ...and the public API still returns the right answer, via the fallback.
    assert akshara_bounds_batch([text]) == [_oracle(text)]


def test_mixed_batch_is_partitioned_not_rejected():
    """An ineligible string must not drag eligible ones onto the slow path.

    Regression guard: an earlier all-or-nothing version of this backend made
    every real 4096-line batch ineligible, because one Devanagari quotation
    anywhere in the block rejected the whole thing.
    """
    ok = normalize("আমি বাংলায় গান গাই")
    bad = "সংস্কৃত धर्म"
    partitioned = bounds_batch([ok, bad, ok])
    assert partitioned is not None
    assert partitioned[0] == akshara_bounds(ok)
    assert partitioned[1] is None
    assert partitioned[2] == akshara_bounds(ok)
    assert akshara_bounds_batch([ok, bad, ok]) == [_oracle(t) for t in (ok, bad, ok)]


def test_falls_back_when_numpy_absent(monkeypatch):
    """The optional-extra contract: no numpy, same answers, no error."""
    import bntok.akshara_vec as vec

    monkeypatch.setattr(vec, "available", lambda: False)
    texts = [normalize(t) for t in BENGALI if t]
    assert vec.bounds_batch(texts) is None
    assert akshara_bounds_batch(texts) == [_oracle(t) for t in texts]


@pytest.mark.parametrize("lead", ["া", "্", "ঁ", "‍", "‌"])
def test_string_starting_with_a_combining_mark_inside_a_batch(lead):
    """A leading mark must not glue onto the batch's internal separator.

    Regression guard: marks always attach leftwards, so a string that starts
    with one attaches to the newline used to join the batch, meaning there is
    no boundary at that string's own offset. An earlier version assumed there
    always was one and dropped the first real boundary of such a string.
    """
    prev = normalize("আমি বাংলায় গান গাই")
    texts = [prev, lead + "ক", prev]
    assert akshara_bounds_batch(texts) == [akshara_bounds(t) for t in texts]
    assert akshara_bounds_batch(texts) == [_oracle(t) for t in texts]


def test_empty_and_degenerate_batches():
    assert akshara_bounds_batch([]) == []
    assert akshara_bounds_batch([""]) == [[]]
    assert akshara_bounds_batch(["", "ক", ""]) == [[], akshara_bounds("ক"), []]


def test_rejects_non_str():
    from bntok.errors import NormalizationError

    with pytest.raises(NormalizationError):
        akshara_bounds_batch(["ঠিক", 5])


def test_lossless_over_batch():
    texts = [normalize(t) for t in BENGALI if t]
    for text, bounds in zip(texts, akshara_bounds_batch(texts)):
        start, parts = 0, []
        for end in bounds:
            parts.append(text[start:end])
            start = end
        assert "".join(parts) == text
