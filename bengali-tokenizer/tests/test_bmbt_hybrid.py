"""
Tests for BMBT-Hybrid (bmbt_hybrid.py, v2 follow-up experiment).

Mirrors tests/test_bmbt.py's coverage (headline guarantees, atom map,
error paths, persistence), plus dedicated tests for the two things this
module adds over BMBT: the fused-vs-factored atom split itself, and the
safety-net merge that guarantees a factored akshara's boundary can never
survive as a real token split -- the property the small-scale experiments
that led to this module found could otherwise regress by 20-70x.
"""

import pytest

from bntok import errors
from bntok.akshara import aksharas
from bntok.bmbt_hybrid import DEFAULT_K_FUSED, BMBTHybrid, HybridAtomMap, _onset_tail_split
from bntok.evaluate import evaluate
from bntok.normalize import normalize

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
    return BMBTHybrid.train(CORPUS, vocab_size=1500, min_atom_freq=1, k_fused=20)


# --- the headline guarantees (same bar as v1/BMBT) ---

def test_roundtrip_bengali(tok):
    for s in ["আমি বাংলায় ক্ষুদ্র গান গাই", "রবীন্দ্রনাথের কবিতা", "পরীক্ষা"]:
        assert tok.decode(tok.encode(s)) == normalize(s)

def test_roundtrip_code_mixed(tok):
    s = "কি খবর? Hello World 123 ঠিক আছে"
    assert tok.decode(tok.encode(s)) == normalize(s)

def test_zero_conjunct_fragmentation(tok):
    # This is the property the safety-net merge exists for: factoring an
    # akshara into onset+tail atoms would otherwise let its boundary
    # survive as a real split whenever the learned BPE merges didn't
    # happen to re-fuse that exact pair (measured 20-70x worse at full
    # scale before this fix). evaluate() is reused completely unmodified.
    rep = evaluate(tok, CORPUS[:6])
    assert rep.conjunct_fragmentation_rate == 0.0
    assert rep.n_fragmented == 0
    assert rep.roundtrip_ok

def test_unseen_conjunct_still_roundtrips(tok):
    assert tok.decode(tok.encode("ষ্প্রি")) == "ষ্প্রি"


# --- onset/tail split primitive ---

def test_onset_tail_split_conjunct():
    onset, tail = _onset_tail_split("ক্ষু")
    assert onset == "ক্ষ"
    assert tail == "ু"

def test_onset_tail_split_no_tail():
    onset, tail = _onset_tail_split("ক")
    assert onset == "ক"
    assert tail == ""

def test_onset_tail_split_vowel_initial_not_split():
    onset, tail = _onset_tail_split("আঃ")
    assert onset == "আঃ"
    assert tail == ""


# --- hybrid atom map: fused vs factored ---

def test_fused_akshara_is_one_piece():
    # "রে" (র + ে matra) is a single akshara chunk with a real tail -- a
    # candidate for factoring unless it's frequent enough to be fused.
    corpus = ["রে রে রে রে রে রে রে রে রে রে"] * 50
    am = HybridAtomMap.build(corpus, k_fused=5, min_freq=1)
    assert "রে" in am.fused_set
    chunk = aksharas("রে")[0]
    assert chunk.text == "রে"
    pieces = am._pieces_for_chunk(chunk)
    assert pieces == ["রে"]

def test_factored_akshara_below_k_splits_into_two():
    # k_fused=0 forces everything with a tail to factor, regardless of frequency.
    am = HybridAtomMap.build(["ক্ষুদ্র বিষয়"], k_fused=0, min_freq=1)
    pieces = am._pieces_for_chunk(aksharas("ক্ষু")[0])
    assert len(pieces) == 2
    assert pieces[0] + pieces[1] == "ক্ষু"

def test_default_k_fused_is_a_positive_int():
    assert isinstance(DEFAULT_K_FUSED, int)
    assert DEFAULT_K_FUSED > 0

def test_atom_map_reversible():
    am = HybridAtomMap.build(["বাংলা ভাষা"], k_fused=0, min_freq=1)
    assert am.decode(am.encode("বাংলা")) == "বাংলা"

def test_atom_map_empty_raises():
    with pytest.raises(errors.EmptyCorpusError):
        HybridAtomMap.build([], k_fused=10, min_freq=1)

def test_encode_collects_multi_atom_chunk_chains():
    am = HybridAtomMap.build(["ক্ষুদ্র বিষয়"], k_fused=0, min_freq=1)
    collected = set()
    am.encode("ক্ষুদ্র", collect=collected)
    assert len(collected) > 0
    for chain in collected:
        assert isinstance(chain, tuple)
        assert len(chain) > 1
        assert all(isinstance(a, str) for a in chain)

def test_single_atom_chunk_not_collected():
    # a chunk that maps to exactly one atom (fused, or no tail to factor)
    # must not appear in the collected chains -- there's nothing to
    # reconstruct a safety net for.
    am = HybridAtomMap.build(["ক ক ক"], k_fused=0, min_freq=1)
    collected = set()
    am.encode("ক", collect=collected)
    assert len(collected) == 0


# --- error handling ---

def test_train_rejects_tiny_vocab():
    with pytest.raises(errors.VocabSizeError):
        BMBTHybrid.train(CORPUS, vocab_size=100)

def test_train_rejects_negative_k_fused():
    with pytest.raises(errors.ConfigError):
        BMBTHybrid.train(CORPUS, vocab_size=1000, k_fused=-1)

def test_train_rejects_single_string():
    with pytest.raises(errors.ConfigError):
        BMBTHybrid.train("not a list", vocab_size=1000)  # type: ignore

def test_train_empty_corpus():
    with pytest.raises(errors.EmptyCorpusError):
        BMBTHybrid.train(["   ", ""], vocab_size=1000)

def test_load_missing_dir():
    with pytest.raises(errors.LoadError):
        BMBTHybrid.load("does/not/exist")


# --- persistence ---

def test_save_load_roundtrip(tok, tmp_path):
    d = str(tmp_path / "bmbt_hybrid")
    tok.save(d)
    t2 = BMBTHybrid.load(d)
    assert t2.decode(t2.encode("বাংলা ভাষা")) == "বাংলা ভাষা"
    assert t2.vocab_size == tok.vocab_size
    assert t2.config["format"] == "bornomala-bmbt-hybrid/1"
    assert t2.config["k_fused"] == 20


# --- featurize() passthrough (identical to BMBT's, reused not duplicated) ---

def test_featurize_passthrough_works(tok):
    result = tok.featurize("ক্ষুদ্র")
    assert len(result) > 0
