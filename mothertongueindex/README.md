# MotherTongueIndex

**How efficient is *your* language on today's AI models — compared to English?**

Paste text in any language, pick the mainstream LLMs you care about, and
MotherTongueIndex runs their *real* tokenizers over it and shows, side by side:

- **how many tokens** your text costs on each model,
- **×English** — the headline: how many times more tokens your language spends
  per word than English does on the *same* model,
- **how the text is split** (see the actual token boundaries),
- **reasoning-capability impact** — because token bloat shrinks the effective
  context window, which measurably handicaps reasoning in non-English languages,
  not just cost.

It is an **understanding** tool, not a price calculator. It answers *why* a
language costs more and *where* the tokenizer breaks.

The same engine produces the cross-tokenizer fertility/STRR comparison tables
that [Project Bornomala](../PROJECT_BORNOMALA_Technical_Specification.md)'s
Track A requires (spec §4.1, §9.4).

---

## Quick start

```bash
pip install -r mothertongueindex/requirements.txt

# Your language vs English across a no-auth model set:
python -m mti "আমি বাংলায় গান গাই, আমি বাংলার গান গাই"

# Choose models, explain the cost, show the capability impact:
python -m mti --models gpt-4o,gpt-4,sarvam1,claude \
    --why --capability "आपकी भाषा यहाँ लिखें"

# See the actual token split:
python -m mti --show --models gpt-4o,sarvam1 "কি খবর"

# List known models / emit JSON:
python -m mti --list
python -m mti --json --file sample.txt
```

## What the columns mean

| Column | Meaning | Better |
|---|---|---|
| `tokens` | Exact token count from the model's real tokenizer | lower |
| `fert.` | Fertility = tokens ÷ whitespace-words | lower |
| `xEN` | Fertility ÷ this model's English fertility. **The headline.** | →1.0 |
| `STRR` | Single-token retention: fraction of words kept as one token | higher |
| `b/tok` | UTF-8 bytes per token (script-independent compression) | higher |
| `gc/tok` | Grapheme clusters per token (true "characters" per token) | higher |

## Models covered

- **OpenAI** GPT-4o / GPT-4.1 / GPT-5 (`o200k_base`), GPT-4 / 3.5 (`cl100k_base`)
  — exact, no auth.
- **Open weights** Qwen3, DeepSeek-V3, Mistral, BLOOM, XLM-R, mBERT, **Sarvam-1**
  (Indic) — exact, no auth.
- **Gated** Llama 4 / 3.1, Gemma 3 — exact, set `HF_TOKEN`.
- **Claude** — Anthropic ships no public tokenizer, so it is a clearly-labelled
  **estimate**, never presented as measured.

## Honesty rules (from the Bornomala spec, §E4)

- Exact counts and estimates are always distinguished (`est` flag).
- The capability-impact numbers are **derived** from tokenization, not measured.
  Measuring real reasoning accuracy is a separate harness
  (`eval/reasoning_probe.py`) meant to run on a machine with a model/API key.

## Layout

```
mothertongueindex/     core engine (CPU only)
  segment.py           grapheme-cluster + script segmentation
  backends.py          tiktoken / HF / estimate tokenizer loaders
  registry.py          the model catalogue
  metrics.py           fertility, STRR, bytes/token, gc/token
  baseline.py          English anchoring (reference + parallel modes)
  capability.py        reasoning-capability impact (derived)
  analyze.py           high-level API
  cli.py               command line
web/                   browser UI (paste + compare visually)
eval/                  reasoning probe (runs on the training/API machine)
train/                 Bengali tokenizer training pipeline (Track A)
```
