"""
Model registry.

Maps a short model id to a display name and a backend spec. This is the single
place that decides which tokenizers the tool knows about.

Availability tiers:
  * ungated  — loads with no auth. Works out of the box.
  * gated    — needs HF_TOKEN (Llama, Gemma). Supported, documented.
  * estimate — no public tokenizer (Claude). Reported as an estimate.

Add a model by adding one row. `family` groups models that share a tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import TiktokenBackend, HFBackend, EstimateBackend


@dataclass
class Model:
    id: str
    display: str
    family: str
    tier: str  # ungated | gated | estimate
    _factory: object = field(repr=False)
    note: str = ""

    def backend(self):
        return self._factory()


def _tk(enc):  return lambda: TiktokenBackend(enc)
def _hf(repo): return lambda: HFBackend(repo)
def _est(lbl): return lambda: EstimateBackend(lbl)


# The curated mainstream set. Order = default display order.
_ROWS: list[Model] = [
    # --- OpenAI (exact, ungated via tiktoken) ---
    Model("gpt-4o",   "GPT-4o / GPT-4.1", "openai-o200k", "ungated", _tk("o200k_base"),
          "o200k_base — also GPT-4o-mini, o1"),
    Model("gpt-5",    "GPT-5 family",     "openai-o200k", "ungated", _tk("o200k_base"),
          "assumes o200k_base"),
    Model("gpt-4",    "GPT-4 / 3.5",      "openai-cl100k","ungated", _tk("cl100k_base"),
          "cl100k_base"),

    # --- Open-weight LLMs on HF (ungated tokenizers) ---
    Model("qwen3",    "Qwen3 / Qwen2.5",  "qwen",      "ungated", _hf("Qwen/Qwen2.5-7B"), ""),
    Model("deepseek", "DeepSeek-V3",      "deepseek",  "ungated", _hf("deepseek-ai/DeepSeek-V3"), ""),
    Model("mistral",  "Mistral",          "mistral",   "ungated", _hf("mistralai/Mistral-7B-v0.3"), ""),
    Model("bloom",    "BLOOM (multiling)","bloom",     "ungated", _hf("bigscience/bloom"), ""),
    Model("xlmr",     "XLM-RoBERTa",      "xlmr",      "ungated", _hf("FacebookAI/xlm-roberta-base"), ""),
    Model("mbert",    "mBERT (multiling)","mbert",     "ungated", _hf("google-bert/bert-base-multilingual-cased"), ""),

    # --- Indic-focused (the comparison Track A cares about) ---
    Model("sarvam1",  "Sarvam-1 (Indic)", "sarvam",    "ungated", _hf("sarvamai/sarvam-1"),
          "Indic-native, primary Bornomala base"),

    # --- Gated (exact, need HF_TOKEN) ---
    Model("llama4",   "Llama 4",          "llama4",    "gated",   _hf("meta-llama/Llama-4-Scout-17B-16E"), ""),
    Model("llama3",   "Llama 3.1",        "llama3",    "gated",   _hf("meta-llama/Llama-3.1-8B"), ""),
    Model("gemma3",   "Gemma 3",          "gemma3",    "gated",   _hf("google/gemma-3-4b-pt"), ""),

    # --- No public tokenizer (estimate) ---
    Model("claude",   "Claude (estimate)","claude",    "estimate",_est("claude"),
          "Anthropic publishes no exact tokenizer — heuristic estimate"),
]

MODELS: dict[str, Model] = {m.id: m for m in _ROWS}


def list_models(tier: str | None = None) -> list[Model]:
    if tier is None:
        return list(_ROWS)
    return [m for m in _ROWS if m.tier == tier]


def get_model(model_id: str) -> Model:
    try:
        return MODELS[model_id]
    except KeyError:
        raise KeyError(
            f"unknown model '{model_id}'. Known: {', '.join(MODELS)}"
        )


# A safe default set that loads without any auth token.
DEFAULT_MODELS = ["gpt-4o", "gpt-4", "qwen3", "sarvam1", "xlmr", "claude"]
