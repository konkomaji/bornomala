"""
Smoke tests using only the OpenAI tiktoken backends, so they run offline with no
Hugging Face downloads and no auth. These assert the invariants the whole tool
depends on, not specific token counts (which can shift with library versions).
"""

import pytest

from mti.segment import grapheme_clusters, words, script_of, dominant_script
from mti.metrics import compute
from mti.analyze import analyze_one, analyze
from mti.registry import MODELS, DEFAULT_MODELS, GROUPS


# --- segmentation ---------------------------------------------------------

def test_grapheme_clusters_bengali_conjunct():
    # A Bengali conjunct is several codepoints but one grapheme cluster.
    s = "ক্ষ"  # ka + virama + ssa -> one perceived unit
    assert len(s) > 1                       # multiple codepoints
    assert len(grapheme_clusters(s)) == 1   # one grapheme cluster

def test_words_whitespace():
    assert words("a b  c\n d") == ["a", "b", "c", "d"]

def test_script_detection():
    assert script_of("A") == "Latin"
    assert script_of("অ") == "Bengali"
    assert dominant_script("hello world") == "Latin"


# --- metrics --------------------------------------------------------------

def test_fertility_and_strr_bounds():
    m = compute("x", "one two three", n_tokens=3, single_token_words=3)
    assert m.n_words == 3
    assert m.fertility == 1.0
    assert m.strr == 1.0
    assert m.bytes_per_token > 0

def test_zero_division_safe():
    m = compute("x", "", n_tokens=0, single_token_words=0)
    assert m.fertility == 0.0
    assert m.bytes_per_token == 0.0


# --- analyze via tiktoken (exact, offline) --------------------------------

def test_analyze_english_gpt4o():
    r = analyze_one("the quick brown fox", "gpt-4o")
    assert r.available
    assert r.metrics.n_tokens > 0
    assert r.metrics.estimated is False

def test_vs_english_ratio_present_for_non_english():
    r = analyze_one("আমি বাংলায় গান গাই", "gpt-4o")
    assert r.available
    # Bengali should cost more per word than the English anchor on cl100k/o200k.
    assert r.vs_english is not None and r.vs_english > 1.0

def test_estimate_flagged():
    r = analyze_one("আমি বাংলায় গান গাই", "claude")
    assert r.available
    assert r.metrics.estimated is True

def test_analyze_multiple_default_models_openai_subset():
    results = analyze("hello world", ["gpt-4o", "gpt-4", "claude"])
    assert len(results) == 3
    assert all(r.available for r in results)


# --- registry invariants --------------------------------------------------

def test_registry_tiers_valid():
    for m in MODELS.values():
        assert m.tier in {"ungated", "gated", "estimate"}

def test_groups_reference_known_models():
    for name, ids in GROUPS.items():
        for mid in ids:
            assert mid in MODELS, f"group {name} references unknown model {mid}"

def test_default_models_are_known():
    for mid in DEFAULT_MODELS:
        assert mid in MODELS
