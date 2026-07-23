"""
Model registry.

Maps a short model id to a display name and a backend spec. This is the single
place that decides which tokenizers the tool knows about.

Availability tiers:
  * ungated  — loads with no auth. Works out of the box.
  * gated    — needs HF_TOKEN (Llama, Gemma, Mistral, Command-R). Supported.
  * estimate — no public tokenizer at all (Claude, Gemini, Grok). Reported as a
               clearly-labelled estimate, never as measured (spec rule E4).

`family` groups models that share a tokenizer. Add a model by adding one row.

A note on tiers being best-effort: gated/ungated status on the Hugging Face Hub
changes over time. If a repo flips, the backend fails soft (reports the model as
unavailable with a hint) rather than crashing the run — so an out-of-date tier
never breaks the tool, it just mislabels one row's expectation.
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


# The catalogue. Order = default display order, grouped by vendor/region.
_ROWS: list[Model] = [
    # ----- OpenAI (exact, ungated via tiktoken) -----
    Model("gpt-5",    "GPT-5 family",      "openai-o200k",  "ungated", _tk("o200k_base"),
          "o200k_base (assumed)"),
    Model("gpt-4o",   "GPT-4o / GPT-4.1",  "openai-o200k",  "ungated", _tk("o200k_base"),
          "o200k_base — also GPT-4o-mini, o1/o3"),
    Model("gpt-4",    "GPT-4 / GPT-3.5",   "openai-cl100k", "ungated", _tk("cl100k_base"),
          "cl100k_base"),

    # ----- Anthropic / Google / xAI (no public tokenizer -> estimate) -----
    Model("claude",   "Claude (estimate)", "claude",  "estimate", _est("claude"),
          "Anthropic publishes no exact tokenizer"),
    Model("gemini",   "Gemini (estimate)", "gemini",  "estimate", _est("gemini"),
          "Gemini tokenizer not downloadable; for exact use Gemma 3 as proxy"),
    Model("grok",     "Grok (estimate)",   "grok",    "estimate", _est("grok"),
          "xAI publishes no tokenizer"),

    # ----- Meta Llama / Google Gemma (exact, gated: set HF_TOKEN) -----
    Model("llama4",   "Llama 4",           "llama4",  "gated", _hf("meta-llama/Llama-4-Scout-17B-16E"), ""),
    Model("llama3",   "Llama 3.1",         "llama3",  "gated", _hf("meta-llama/Llama-3.1-8B"), ""),
    Model("gemma3",   "Gemma 3",           "gemma3",  "gated", _hf("google/gemma-3-4b-pt"),
          "close proxy for Gemini"),

    # ----- Open-weight frontier (ungated tokenizers) -----
    Model("qwen3",    "Qwen3 / Qwen2.5",   "qwen",     "ungated", _hf("Qwen/Qwen2.5-7B"), ""),
    Model("deepseek", "DeepSeek-V3",       "deepseek", "ungated", _hf("deepseek-ai/DeepSeek-V3"), ""),
    Model("deepseek-r1","DeepSeek-R1",     "deepseek", "ungated", _hf("deepseek-ai/DeepSeek-R1"), ""),
    Model("kimi",     "Kimi K2 (Moonshot)","kimi",     "ungated", _hf("moonshotai/Kimi-K2-Instruct"), ""),
    Model("glm4",     "GLM-4 (Zhipu)",     "glm",      "ungated", _hf("THUDM/glm-4-9b"), ""),
    Model("yi",       "Yi 1.5 (01.AI)",    "yi",       "ungated", _hf("01-ai/Yi-1.5-9B"), ""),
    Model("phi4",     "Phi-4 (Microsoft)", "phi",      "ungated", _hf("microsoft/phi-4"), ""),
    Model("falcon",   "Falcon (TII)",      "falcon",   "ungated", _hf("tiiuae/falcon-7b"), ""),

    # ----- Gated Western open-weight -----
    Model("mistral",  "Mistral / Mixtral", "mistral",  "gated", _hf("mistralai/Mistral-7B-v0.3"), ""),
    Model("command-r","Command R (Cohere)","cohere",   "gated", _hf("CohereForAI/c4ai-command-r-v01"), ""),

    # ----- Multilingual encoders (ungated; strong Indic coverage baselines) -----
    Model("bloom",    "BLOOM (multiling)", "bloom",    "ungated", _hf("bigscience/bloom"), ""),
    Model("xlmr",     "XLM-RoBERTa",       "xlmr",     "ungated", _hf("FacebookAI/xlm-roberta-base"), ""),
    Model("mbert",    "mBERT (multiling)", "mbert",    "ungated", _hf("google-bert/bert-base-multilingual-cased"), ""),

    # ----- Indian models (the comparison Track A cares about) -----
    Model("sarvam1",  "Sarvam-1 (Indic)",  "sarvam",   "ungated", _hf("sarvamai/sarvam-1"),
          "Indic-native, primary Bornomala base"),
    Model("sarvam-m", "Sarvam-M (24B)",    "sarvam-m", "ungated", _hf("sarvamai/sarvam-m"),
          "Mistral-based Indic"),
    Model("openhathi","OpenHathi (AI4Bharat)","openhathi","ungated", _hf("sarvamai/OpenHathi-7B-Hi-v0.1-Base"),
          "Llama-2 Hindi continued-pretrain"),
    Model("indicbert","IndicBERT (AI4Bharat)","indicbert","ungated", _hf("ai4bharat/indic-bert"),
          "12 Indian languages"),
    Model("titulm",   "TituLLM (Bengali)", "titulm",   "ungated", _hf("hishab/titulm-llama-3.2-1b-v2.0"),
          "Hishab — Bengali-focused, closest prior art"),
    Model("param1",   "Param-1 (BharatGen)","param1",  "gated",   _hf("bharatgenai/Param-1-2.9B-Instruct"),
          "government-backed, 25% Indic"),
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
        raise KeyError(f"unknown model '{model_id}'. Known: {', '.join(MODELS)}")


# A no-auth default set spanning OpenAI, open-weight frontier, multilingual, and
# Indic — every one loads without an HF token.
DEFAULT_MODELS = ["gpt-4o", "gpt-4", "qwen3", "deepseek", "sarvam1", "xlmr", "claude", "gemini"]

# Convenience groups for the CLI / web presets.
GROUPS = {
    "default": DEFAULT_MODELS,
    "openai": ["gpt-5", "gpt-4o", "gpt-4"],
    "frontier": ["gpt-4o", "claude", "gemini", "grok", "llama4", "deepseek", "qwen3", "kimi"],
    "indian": ["sarvam1", "sarvam-m", "openhathi", "indicbert", "titulm", "param1"],
    "multilingual": ["bloom", "xlmr", "mbert", "sarvam1", "qwen3"],
    "open": ["qwen3", "deepseek", "deepseek-r1", "kimi", "glm4", "yi", "phi4", "falcon"],
}
