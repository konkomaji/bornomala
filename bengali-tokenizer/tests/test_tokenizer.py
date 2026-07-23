"""
Tests for the Track A Bengali tokenizer.

The headline guarantees are asserted directly: NFC normalisation, grapheme-cluster
integrity (conjunct fragmentation rate = 0), round-trip fidelity for Bengali and
code-mixed text, and that every failure mode raises a typed error.
"""

import pytest

from bntok import BengaliTokenizer, evaluate, grapheme_clusters, normalize
from bntok.atoms import AtomMap
from bntok.graphemes import is_conjunct, has_reph
from bntok import errors


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
    return BengaliTokenizer.train(CORPUS, algo="bpe", vocab_size=1500, min_atom_freq=1)


# --- linguistic primitives ---

def test_grapheme_cluster_is_one_unit():
    assert len(grapheme_clusters("ক্ষ")) == 1
    assert len("ক্ষ") > 1

def test_conjunct_and_reph_detection():
    assert is_conjunct("ক্ষ")
    assert has_reph("র্ম")
    assert not is_conjunct("ক")

def test_nfc_normalisation():
    assert normalize("বাংলা") == "বাংলা"
    with pytest.raises(errors.NormalizationError):
        normalize(123)  # type: ignore


# --- the headline guarantees ---

def test_roundtrip_bengali(tok):
    for s in ["আমি বাংলায় ক্ষুদ্র গান গাই", "রবীন্দ্রনাথের কবিতা", "পরীক্ষা"]:
        assert tok.decode(tok.encode(s)) == normalize(s)

def test_roundtrip_code_mixed(tok):
    s = "কি খবর? Hello World 123 ঠিক আছে"
    assert tok.decode(tok.encode(s)) == normalize(s)

def test_zero_conjunct_fragmentation(tok):
    rep = evaluate(tok, CORPUS[:6])
    assert rep.conjunct_fragmentation_rate == 0.0
    assert rep.n_fragmented == 0
    assert rep.roundtrip_ok

def test_unseen_conjunct_still_roundtrips(tok):
    # A conjunct never present as a whole in the corpus must still round-trip via
    # the codepoint fallback and guaranteed coverage.
    assert tok.decode(tok.encode("ষ্প্রি")) == "ষ্প্রি"


# --- atom map ---

def test_atom_map_reversible():
    am = AtomMap.build(["বাংলা ভাষা"], min_freq=1)
    assert am.decode(am.encode("বাংলা")) == "বাংলা"

def test_atom_map_empty_raises():
    with pytest.raises(errors.EmptyCorpusError):
        AtomMap.build([], min_freq=1)


# --- error handling ---

def test_train_rejects_bad_algo():
    with pytest.raises(errors.ConfigError):
        BengaliTokenizer.train(CORPUS, algo="wordpiece", vocab_size=1000)

def test_train_rejects_tiny_vocab():
    # Below the atom alphabet (Bengali block + ASCII + specials) it is impossible.
    with pytest.raises(errors.VocabSizeError):
        BengaliTokenizer.train(CORPUS, algo="bpe", vocab_size=100)

def test_train_rejects_single_string():
    with pytest.raises(errors.ConfigError):
        BengaliTokenizer.train("not a list", algo="bpe", vocab_size=1000)  # type: ignore

def test_train_empty_corpus():
    with pytest.raises(errors.EmptyCorpusError):
        BengaliTokenizer.train(["   ", ""], algo="bpe", vocab_size=1000)

def test_load_missing_dir():
    with pytest.raises(errors.LoadError):
        BengaliTokenizer.load("does/not/exist")


# --- persistence ---

def test_save_load_roundtrip(tok, tmp_path):
    d = str(tmp_path / "tok")
    tok.save(d)
    t2 = BengaliTokenizer.load(d)
    assert t2.decode(t2.encode("বাংলা ভাষা")) == "বাংলা ভাষা"
    assert t2.vocab_size == tok.vocab_size
