"""Tests for the Banglish pipeline (bntok.banglish): the tier-1 lookup, the
tier-2 from-scratch classifier, and the transliterate() cascade including
its self-growing cache. Tier 3 itself (the real seq2seq model) does not
exist yet, so its slot is tested here against a stub `tier3_fn` - these
tests prove the cache-growth mechanism works, not that any particular
transliteration model is accurate (that's a separate, real-data validation,
see scripts/build_banglish_lookup.py and docs/known-issues.md).
"""

from bntok.banglish import NgramClassifier, TransliterationResult, transliterate


def _tiny_classifier():
    # A small, fast classifier for tests: real Banglish-shaped words vs
    # real English-shaped words, not the full 125k/10k production lists.
    banglish = ["ami", "tumi", "ache", "korchi", "bhalo", "kemon", "keno", "kotha"] * 5
    english = ["the", "and", "hello", "world", "book", "table", "music", "friend"] * 5
    return NgramClassifier.train(banglish, english)


def test_tier1_lookup_hit_substitutes():
    lookup = {"ami": ("আমি", "real")}
    clf = _tiny_classifier()
    result = transliterate("ami bhalo achi", lookup, clf)
    assert result.text.startswith("আমি ")
    assert result.tier1_hits == 1
    assert result.total_latin_words == 3


def test_tier2_leaves_real_english_untouched():
    lookup = {}
    clf = _tiny_classifier()
    result = transliterate("hello world", lookup, clf)
    assert result.text == "hello world"
    assert result.tier2_english == 2
    assert result.tier1_hits == 0


def test_bengali_script_passes_through_unchanged():
    lookup = {"ami": ("আমি", "real")}
    clf = _tiny_classifier()
    result = transliterate("আমি ভালো আছি", lookup, clf)
    assert result.text == "আমি ভালো আছি"
    assert result.total_latin_words == 0
    assert result.tier1_hits == 0


def test_no_tier3_fn_leaves_unresolved_words_unchanged():
    lookup = {}
    clf = _tiny_classifier()
    result = transliterate("kortesi", lookup, clf)
    # not in lookup, and shaped like Banglish, not English -> unresolved
    assert result.tier3_unresolved == 1
    assert result.tier3_hits == 0
    assert "kortesi" in result.text


def test_tier3_fn_resolves_and_grows_the_cache():
    lookup = {}
    clf = _tiny_classifier()
    calls = []

    def stub_tier3(word: str) -> str:
        calls.append(word)
        return "স্টাব"  # arbitrary placeholder Bengali output, not a real model

    r1 = transliterate("kortesi", lookup, clf, tier3_fn=stub_tier3)
    assert r1.tier3_hits == 1
    assert r1.cache_growth == 1
    assert "স্টাব" in r1.text
    assert calls == ["kortesi"]
    assert "kortesi" in lookup  # written back

    # Second call, same word: now a tier-1 hit, tier3_fn is NOT called again.
    r2 = transliterate("kortesi", lookup, clf, tier3_fn=stub_tier3)
    assert r2.tier1_hits == 1
    assert r2.tier3_hits == 0
    assert r2.cache_growth == 0
    assert calls == ["kortesi"]  # unchanged: no second call


def test_tier3_fn_returning_none_still_counts_as_unresolved():
    lookup = {}
    clf = _tiny_classifier()
    result = transliterate("kortesi", lookup, clf, tier3_fn=lambda w: None)
    assert result.tier3_hits == 0
    assert result.tier3_unresolved == 1
    assert result.cache_growth == 0


def test_result_is_transliteration_result():
    lookup = {}
    clf = _tiny_classifier()
    result = transliterate("hello", lookup, clf)
    assert isinstance(result, TransliterationResult)
