"""
Tests for BMBT (Bornomala's Bengali Tokenizer, v2 roadmap step 5, partial).

Mirrors tests/test_tokenizer.py's coverage 1:1 (headline guarantees, atom
map, error paths, persistence) so the two tokenizers are held to the same
bar, plus a dedicated section for featurize(), the genuinely new capability
this module adds over v1.
"""

import pytest

from bntok import errors
from bntok.akshara import aksharas
from bntok.bmbt import BMBT, AksharaAtomMap, AksharaFeatures, featurize, featurize_akshara
from bntok.evaluate import evaluate
from bntok.normalize import normalize
from bntok.substrate import NUKTA, VIRAMA

CORPUS = [
    "আমি বাংলায় গান গাই আমি বাংলার গান গাই",
    "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে",
    "বাংলা আমার মাতৃভাষা আমি বাংলায় কথা বলি এবং লিখি",
    "পশ্চিমবঙ্গের উপভাষা নিয়ে গবেষণা জরুরি ও গুরুত্বপূর্ণ",
    "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের শ্রেষ্ঠ কবি ছিলেন",
    "যুক্তাক্ষর ক্ষ জ্ঞ ত্র দ্ধ ঙ্ক্ষ পরীক্ষা করা হচ্ছে",
] * 150


@pytest.fixture(scope="module")
def tok():
    return BMBT.train(CORPUS, algo="bpe", vocab_size=1500, min_atom_freq=1)


# --- linguistic primitive ---

def test_akshara_conjunct_is_one_unit():
    chunks = aksharas("ক্ষ")
    assert len(chunks) == 1
    assert chunks[0].kind == "akshara"


# --- the headline guarantees ---

def test_roundtrip_bengali(tok):
    for s in ["আমি বাংলায় ক্ষুদ্র গান গাই", "রবীন্দ্রনাথের কবিতা", "পরীক্ষা"]:
        assert tok.decode(tok.encode(s)) == normalize(s)

def test_roundtrip_code_mixed(tok):
    s = "কি খবর? Hello World 123 ঠিক আছে"
    assert tok.decode(tok.encode(s)) == normalize(s)

def test_zero_conjunct_fragmentation(tok):
    # evaluate.py is v1's own evaluation function, reused here completely
    # unmodified - this test doubles as regression coverage proving that
    # reuse actually works, not just that it type-checks.
    rep = evaluate(tok, CORPUS[:6])
    assert rep.conjunct_fragmentation_rate == 0.0
    assert rep.n_fragmented == 0
    assert rep.roundtrip_ok

def test_unseen_conjunct_still_roundtrips(tok):
    assert tok.decode(tok.encode("ষ্প্রি")) == "ষ্প্রি"


# --- akshara atom map ---

def test_atom_map_reversible():
    am = AksharaAtomMap.build(["বাংলা ভাষা"], min_freq=1)
    assert am.decode(am.encode("বাংলা")) == "বাংলা"

def test_atom_map_empty_raises():
    with pytest.raises(errors.EmptyCorpusError):
        AksharaAtomMap.build([], min_freq=1)


# --- error handling ---

def test_train_rejects_bad_algo():
    with pytest.raises(errors.ConfigError):
        BMBT.train(CORPUS, algo="wordpiece", vocab_size=1000)

def test_train_rejects_tiny_vocab():
    with pytest.raises(errors.VocabSizeError):
        BMBT.train(CORPUS, algo="bpe", vocab_size=100)

def test_train_rejects_single_string():
    with pytest.raises(errors.ConfigError):
        BMBT.train("not a list", algo="bpe", vocab_size=1000)  # type: ignore

def test_train_empty_corpus():
    with pytest.raises(errors.EmptyCorpusError):
        BMBT.train(["   ", ""], algo="bpe", vocab_size=1000)

def test_load_missing_dir():
    with pytest.raises(errors.LoadError):
        BMBT.load("does/not/exist")


# --- persistence ---

def test_save_load_roundtrip(tok, tmp_path):
    d = str(tmp_path / "bmbt")
    tok.save(d)
    t2 = BMBT.load(d)
    assert t2.decode(t2.encode("বাংলা ভাষা")) == "বাংলা ভাষা"
    assert t2.vocab_size == tok.vocab_size
    assert t2.config["format"] == "bornomala-bmbt/1"


# --- featurize() ---

def _reconstruct(f: AksharaFeatures) -> str:
    """Rebuild surface text from a decomposition: the one check that proves
    featurize() is lossless, not just 'looks right on a couple of examples'."""
    parts = []
    for i, c in enumerate(f.onset):
        if i > 0:
            parts.append(VIRAMA)
        parts.append(c)
        if f.nuktas[i]:
            parts.append(NUKTA)
    if f.vowel:
        parts.append(f.vowel)
    parts.extend(f.modifiers)
    return "".join(parts)


HARD_WORDS = ["স্ত্রী", "ক্ষ্ম", "আকাঙ্ক্ষা", "ঋত্বিক"]


@pytest.mark.parametrize("word", HARD_WORDS)
def test_featurize_reconstructs_hard_words_exactly(word):
    for f in featurize(word):
        assert isinstance(f, AksharaFeatures)
        assert _reconstruct(f) == f.text

def test_featurize_simple_consonant_matra():
    f = featurize_akshara(aksharas("কি")[0])
    assert f.onset == ["ক"]
    assert f.nuktas == [False]
    assert f.vowel == "ি"
    assert f.modifiers == []

def test_featurize_multi_consonant_conjunct_onset_order():
    f = featurize_akshara(aksharas("স্ত্রী")[0])
    assert f.onset == ["স", "ত", "র"]
    assert f.vowel == "ী"

def test_featurize_nukta_mid_chain():
    # RRA (ড়) as a decomposed base+nukta pair, chained via virama into a
    # further consonant: nukta must land on the correct onset element.
    da, nukta, virama, ka = chr(0x09A1), chr(0x09BC), chr(0x09CD), chr(0x0995)
    text = da + nukta + virama + ka
    chunk = aksharas(text)[0]
    f = featurize_akshara(chunk)
    assert f.onset == [da, ka]
    assert f.nuktas == [True, False]
    assert _reconstruct(f) == text

def test_featurize_trailing_modifier_not_swallowed_into_vowel():
    f = featurize_akshara(aksharas("আঃ")[0])
    assert f.vowel == "আ"
    assert f.modifiers == [chr(0x0983)]  # visarga

def test_featurize_independent_vowel_branch_has_empty_onset():
    f = featurize_akshara(aksharas("আ")[0])
    assert f.onset == []
    assert f.nuktas == []
    assert f.vowel == "আ"

def test_featurize_akshara_raises_on_other_kind():
    chunk = aksharas("!")[0]
    assert chunk.kind == "other"
    with pytest.raises(errors.BnTokError):
        featurize_akshara(chunk)

def test_featurize_whole_string_mixed_content_does_not_raise():
    result = featurize("আমি busy আছি")
    assert len(result) > 0
    kinds_seen = {type(r).__name__ for r in result}
    assert "AksharaFeatures" in kinds_seen  # Bengali aksharas were featurized
    assert any(getattr(r, "kind", None) == "other" for r in result)  # English/space untouched
