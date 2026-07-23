# MotherTongueIndex: Measuring Multilingual Tokenizer Efficiency Against English

**Author.** Konko Maji (work.konkomaji@gmail.com)
**Document.** Technical report, Draft 1.0
**Status.** A standalone subproject of Project Bornomala. Not for external circulation as a finished result. Numbers in this report are reproducible from the bundled code and data; estimates are labelled as estimates.
**Scope note.** MotherTongueIndex (MTI) is a measurement and analysis tool. It does not train, produce, or ship a tokenizer. Project Bornomala Track A builds the actual Bengali tokenizer in the main repository; MTI measures tokenizers, including that one once it exists.

---

## Abstract

Every mainstream large language model reads text through a tokenizer, and every mainstream tokenizer is optimised, in vocabulary allocation and in merge statistics, for English and a handful of high-resource Latin-script languages. The consequence is a measurable inequity: the same meaning, expressed in a different language, is charged a different number of tokens. That difference is not only a billing difference. Because a context window holds a fixed number of tokens rather than a fixed amount of meaning, token inefficiency also shrinks the effective context available to a language, which the literature links to degraded in-context and reasoning performance at fixed budget.

MotherTongueIndex is a small, dependency-light, CPU-only tool that quantifies this. It runs the real tokenizers of 28 mainstream models over text in any language and reports, against an English anchor: token count, fertility, single-token retention, bytes per token, grapheme clusters per token, and a versus-English ratio. It adds a derived reasoning-capability signal (effective-context loss) and ships a separate, opt-in probe for the measured version of that signal. Using a parallel, content-controlled corpus (the opening of Article 1 of the Universal Declaration of Human Rights in 15 languages), we show a large and generation-dependent penalty for non-Latin and especially complex (Indic and abugida) scripts, and we show that a tokenizer generation change alone (OpenAI cl100k to o200k) moves Bengali fertility from 8.40 to 1.90 tokens per word. The tool is language-agnostic by construction and is released under Apache 2.0.

---

## 1. Introduction

There is a widespread and load-bearing assumption that a model treats all of its supported languages roughly equally. It does not. Before a single transformer layer runs, text passes through a tokenizer whose vocabulary was induced from a training mixture dominated by English. A language whose words, characters, or scripts are rare in that mixture is split into more, smaller pieces. This is the tokenizer fertility problem, and it has two distinct downstream effects that are routinely conflated:

1. **Cost and throughput.** More tokens for the same content means more money per request, slower generation, and sooner-exhausted context. This is a linear, obvious effect.
2. **Capability at fixed context.** A context window is denominated in tokens, not in meaning. If a language spends four times as many tokens per unit of content, then a fixed window holds one quarter of the content, one quarter of the few-shot examples, and one quarter of the reasoning scratch space. IndicGenBench [4] reports empirically that higher fertility correlates with degraded downstream performance at fixed context. This is the effect that is usually missed.

MTI exists to make both effects visible and quantitative for anyone, in any language, against the one anchor every tokenizer is tuned for: English. The central user question is deliberately simple. Paste your language. See how many times more tokens it costs than English on GPT-4o, on Claude, on Sarvam, on Llama, and understand why.

The tool is also a component of a larger programme. Project Bornomala argues that the binding constraint on Bengali language technology is data, not compute, and that the correct headline metrics for Bengali are grapheme-aware, not codepoint-aware. MTI operationalises the tokenization half of that argument in a form that is useful to speakers of every language, not only Bengali.

## 2. Background

### 2.1 Subword tokenization and fertility

Modern LLMs use subword tokenizers, typically byte-level Byte Pair Encoding (BPE) or Unigram language model tokenizers. A tokenizer `T` maps a string to a sequence of subword ids. The standard efficiency measure is **fertility**:

```
fertility(T, D) = |T(D)| / |words(D)|
```

for a corpus `D`, where `words` is whitespace segmentation. Lower is better. English fertility for well-matched tokenizers is close to 1.1 to 1.3. For Indic scripts, published multilingual models exhibit fertility of 4 to 8, and purpose-built Indic tokenizers such as Sarvam-1 [2] reach 1.4 to 2.1. IndicSuperTokenizer [1] reports a 39.5 percent average fertility improvement over one baseline family across 22 Indic languages, English, and code.

Fertility alone is insufficient. **Single Token Retention Rate** (STRR), the fraction of words preserved as exactly one token, exposes vocabulary allocation that fertility hides [7]:

```
STRR(T, D) = |{w in words(D) : |T(w)| = 1}| / |words(D)|
```

### 2.2 The second-order capability effect

IndicGenBench [4] measures fertility from about 4.1 (Pashto) to 19.9 (Tibetan) and shows that high fertility means fewer in-context examples fit inside a fixed context window, which measurably degrades downstream performance on the affected language. Tokenization is therefore not only a cost problem but a capability problem at fixed context. MTI encodes this relationship explicitly (Section 6).

### 2.3 Unicode grounding: why codepoint counting misleads

A naive tool counts Unicode codepoints and calls them characters. For complex scripts this is wrong. Bengali, like other Brahmic abugidas, is not an alphabet. A written unit (a grapheme cluster) is a consonant carrying an inherent vowel, modified by diacritics (matras) and combined into conjunct ligatures (juktakkhor). One grapheme cluster can span one to roughly eight codepoints, and visual order is not codepoint order (the vowel sign in `কি` renders to the left of the consonant it follows in memory).

MTI therefore computes characters as **extended grapheme clusters** under Unicode Standard Annex 29 [8], using the `regex` module's `\X`. It also normalises strings with Normalization Form C [9] where a comparison depends on it. Reporting grapheme clusters per token, rather than codepoints per token, gives an honest picture of how much readable text a token actually carries in Bengali, Tamil, Hindi, and similar scripts.

## 3. Method and Metrics

For a text and a tokenizer, MTI computes:

| Metric | Definition | Direction |
|---|---|---|
| tokens | Exact count from the model's real tokenizer | lower is cheaper |
| fertility | tokens / whitespace-words | lower is better |
| STRR | fraction of words that are one token | higher is better |
| bytes_per_token | UTF-8 bytes / tokens | higher is better |
| gc_per_token | grapheme clusters / tokens | higher is better |
| vs_english | fertility / English-anchor fertility on the same model | closer to 1.0 is fairer |

### 3.1 The English anchor: two modes

The headline metric is `vs_english`. It is provided in two modes.

**Reference mode (zero effort, approximate).** Each model has an English reference fertility, computed once from a fixed public-domain English sample (the opening of Article 1 of the Universal Declaration of Human Rights). For a user's pasted text we report `fertility(text) / english_reference_fertility(model)`. The reading is: on this model your language averages N tokens per word, English averages M, so your language is N/M times English. It is approximate because it compares your words to English words, not the same content.

**Parallel mode (exact, content-controlled).** When the same meaning is supplied in English and in the target language, MTI reports the exact token ratio for identical content. The bundled UDHR corpus is parallel across all 15 languages, so the cross-language results in Section 5 are content-controlled: the token counts describe the same sentence rendered in each language, not different sentences.

### 3.2 What MTI is not

MTI is not a price calculator. It reports tokens and ratios, not currency. Prices change per vendor and per tier; the token structure does not. MTI explains why a cost moves, and leaves the multiplication by a per-token price to the reader.

## 4. Architecture

The engine is a small pipeline: segment, encode, measure, anchor, assess. Full module detail and diagrams are in `docs/architecture.md`.

Three tokenizer backends give exact or clearly-labelled results:

- **tiktoken backend (exact, no auth).** OpenAI GPT families via the public BPE ranks `o200k_base` (GPT-4o, GPT-4.1, GPT-5 family, o1, o3) and `cl100k_base` (GPT-4, GPT-3.5).
- **Hugging Face backend (exact).** Any `tokenizers` or `transformers` tokenizer: Qwen, DeepSeek, Mistral, BLOOM, XLM-R, mBERT, Sarvam-1, and the Indian models, plus gated Llama and Gemma when `HF_TOKEN` is set. Unavailable repos fail soft and are reported, never faked.
- **Estimate backend (labelled).** Models with no public tokenizer (Claude, Gemini, Grok) receive a heuristic token estimate from per-script bytes-per-token priors, always flagged `estimated`. These are never presented as measured, in keeping with the honesty rules.

The registry catalogues 28 models across OpenAI, Anthropic, Google, xAI, Meta, Mistral, Cohere, and the open-weight and Indian ecosystems, with an availability tier (ungated, gated, estimate) on each. When the Bornomala Bengali tokenizer is released it will be added here as one more row, and every table in this report will regenerate to include it.

## 5. Results

### 5.1 Setup

We tokenize the parallel UDHR Article 1 opening in 15 languages with four models that require no downloads: GPT-4o and GPT-4 (exact, tiktoken) and Claude and Gemini (estimate). Data and regeneration command:

```
cd mothertongueindex
python data/build_tables.py --models gpt-4o,gpt-4,claude,gemini
```

Outputs are in `data/tables/`. The figures below are copied verbatim from `data/tables/by_language.json`. Claude and Gemini columns are estimates.

### 5.2 Cross-language fertility and versus-English ratio

Fertility (tokens per whitespace word) and, in brackets, the versus-English ratio. Lower is better.

| Language | Script | GPT-4o (o200k) | GPT-4 (cl100k) | Claude (est) | Gemini (est) |
|---|---|---|---|---|---|
| English | Latin | 1.08 (1.0) | 1.08 (1.0) | 1.08 (1.0) | 0.83 (1.0) |
| Spanish | Latin | 1.25 (1.14) | 1.58 (1.44) | 1.25 (1.07) | 0.92 (1.06) |
| French | Latin | 1.31 (1.19) | 1.69 (1.54) | 1.23 (1.06) | 0.92 (1.07) |
| Portuguese | Latin | 1.25 (1.14) | 1.75 (1.59) | 1.25 (1.07) | 0.92 (1.06) |
| German | Latin | 1.27 (1.16) | 1.45 (1.32) | 1.27 (1.09) | 0.91 (1.05) |
| Russian | Cyrillic | 1.64 (1.49) | 3.27 (2.98) | 4.82 (4.13) | 3.55 (4.09) |
| Greek | Greek | 2.58 (2.35) | 6.33 (5.76) | 5.00 (4.29) | 3.67 (4.23) |
| Arabic | Arabic | 2.25 (2.05) | 4.63 (4.21) | 4.88 (4.18) | 3.63 (4.18) |
| Hindi | Devanagari | 1.60 (1.46) | 5.53 (5.03) | 8.93 (7.66) | 6.60 (7.62) |
| Bengali | Bengali | 1.90 (1.73) | 8.40 (7.64) | 11.70 (10.03) | 8.70 (10.04) |
| Tamil | Tamil | 3.75 (3.41) | 15.50 (14.09) | 22.38 (19.18) | 16.50 (19.04) |
| Korean | Hangul | 2.40 (2.18) | 4.10 (3.73) | 4.40 (3.77) | 3.20 (3.69) |

Chinese, Japanese, and Thai are reported separately in Section 5.4 because they have no whitespace, which breaks the word-based fertility denominator.

### 5.3 Findings

**Latin-script languages sit near parity.** Spanish, French, Portuguese, and German land within roughly 1.1 to 1.6 times English on all exact tokenizers. The tokenizer was built for this neighbourhood.

**Complex scripts are penalised heavily, and the penalty is generation-dependent.** The single most important result is the OpenAI cl100k to o200k jump. Holding the model vendor and the content fixed, and changing only the tokenizer generation:

- Bengali fertility falls from 8.40 (GPT-4, cl100k) to 1.90 (GPT-4o, o200k), a versus-English ratio of 7.64 down to 1.73.
- Hindi falls from 5.53 to 1.60, a ratio of 5.03 down to 1.46.
- Tamil falls from 15.50 to 3.75, a ratio of 14.09 down to 3.41.
- Greek falls from 6.33 to 2.58, Arabic from 4.63 to 2.25.

This is a clean, reproducible demonstration that a large part of the multilingual penalty is a tokenizer artifact, not an inherent property of the language, and that it is fixable by vocabulary design. It also warns that any Bengali or Indic efficiency claim must state the exact tokenizer, because the same nominal vendor can differ by more than 4x across generations.

**The estimated frontier tokenizers (Claude, Gemini) show the same qualitative pattern.** Both estimate large penalties for Indic scripts (Claude Bengali 11.70, Gemini 8.70 by estimate). These are heuristic and are not measurements; they are included to give a plausible prior and are flagged as estimates in every table.

### 5.4 Scripts without whitespace

Chinese, Japanese, and Thai do not delimit words with spaces, so `words(D)` collapses toward 1 and fertility becomes tokens per sentence rather than tokens per word. For these languages the honest comparison is **bytes per token**, which is script-independent:

| Language | GPT-4o bytes/token | GPT-4 bytes/token |
|---|---|---|
| English | 4.85 | 4.85 |
| Chinese | 3.80 | 2.71 |
| Japanese | 3.60 | 2.80 |
| Thai | 5.90 | 2.81 |

The same o200k versus cl100k improvement appears: Thai rises from 2.81 to 5.90 bytes per token, Chinese from 2.71 to 3.80. MTI reports fertility, bytes per token, and grapheme clusters per token together precisely so that no single distorted number is read in isolation.

## 6. Capability Impact

### 6.1 Derived signal (not a measurement)

`mti/capability.py` turns a versus-English fertility ratio into an effective-context statement:

```
effective_context_ratio = english_fertility / language_fertility = 1 / vs_english
```

If your language is 4x English, then in the same window you get one quarter of English's usable content, one quarter of the few-shot examples, and one quarter of the reasoning room. MTI maps the ratio to coarse risk bands (LOW, MODERATE, HIGH, SEVERE) and prints, for common window sizes, the English-equivalent content that actually fits.

Worked from the real data: Bengali on GPT-4 (cl100k), vs_english 7.64, yields an effective-context ratio of about 0.13. A nominal 128k window holds roughly 17k tokens of English-equivalent content for that language on that tokenizer. On GPT-4o (o200k), vs_english 1.73, the same window holds roughly 74k. This is a derived consequence of tokenization arithmetic, presented as such, not a measured accuracy.

### 6.2 Measured signal (opt-in, separate machine)

Whether the shrunken context actually degrades reasoning is an empirical question that requires running a model. `eval/reasoning_probe.py` is the harness for that: it runs identical reasoning items in English and in a target language through a pluggable answer function (any OpenAI, Anthropic, or HF endpoint) and reports accuracy in each language, the accuracy gap, and mean tokens per item, so the gap can be read next to the tokenization overhead MTI predicts.

The probe is deliberately kept out of the core package. The Bornomala plan runs this class of work on a rented GPU or an API machine, not on the local CPU box (parent spec Section 15). Run without a wired answer function it reports tokens only, in an explicit dry mode, and fabricates nothing.

## 7. Honesty and Limitations

- **Estimates.** Claude, Gemini, and Grok have no public tokenizer. Their columns are heuristic estimates from per-script bytes-per-token priors, flagged `estimated` everywhere. They must not be cited as measured token counts.
- **Whitespace fertility.** For scripts without spaces (Han, Thai, Khmer, and others) word-based fertility is not meaningful. Use bytes per token or grapheme clusters per token for those languages. MTI reports all three.
- **Reference-mode versus parallel-mode.** Reference mode compares a user's words to English words, not the same content; it is a fast approximation. The Section 5 results use the parallel UDHR corpus, which is content-controlled and therefore stronger.
- **UDHR register.** UDHR Article 1 is one register (formal, legal). Fertility varies by register and domain. A single sentence is a demonstration, not a corpus. The tool is built to run on the user's own text for exactly this reason.
- **Speaker figures.** Language speaker counts in `mti/languages.py` are rounded order-of-magnitude approximations for sorting and context, never precise statistics.
- **Capability numbers.** Section 6.1 figures are derived from tokenization, not measured. Measured accuracy requires the Section 6.2 probe.
- **Tier drift.** Gated versus ungated status on the model hub changes over time. A stale tier never breaks the tool; the backend fails soft.

## 8. Relation to Project Bornomala

Project Bornomala is a Bengali-first language technology programme whose thesis is that the binding constraint is data, not compute, and whose Track A builds a Bengali-only, grapheme-cluster-aware tokenizer in the main repository. MTI is a subproject that supports Track A in three concrete ways:

1. **It produces the comparison table Track A needs.** The parent spec notes that Bengali does not appear in a single published cross-tokenizer fertility comparison. MTI generates exactly such a table, for Bengali and for 14 other languages, from real tokenizers, reproducibly.
2. **It will benchmark the Bornomala tokenizer.** When the Track A tokenizer ships in the main repo, it becomes one more registry row, and the tool immediately reports its fertility, STRR, and versus-English position against all 28 incumbents.
3. **It generalises the equity argument.** Bengali's tokenization penalty is a special case of a global pattern. Presenting the pattern for every language, with Bengali as one clearly disadvantaged instance, is a stronger and more honest framing than a Bengali-only claim.

MTI does not train a tokenizer and does not itself advance the corpus or model tracks. It measures.

## 9. Reproducibility

- Python 3.10 or newer, CPU only. No GPU, no training.
- Install: `pip install -r requirements.txt` (core) and optionally the `hf` extra for Hugging Face tokenizers.
- Deterministic: tokenization is a pure function of text and tokenizer version. There is no sampling, no randomness, no `Date.now`.
- Regenerate the results table: `python data/build_tables.py --models gpt-4o,gpt-4,claude,gemini`.
- Single query: `python -m mti --models gpt-4o,gpt-4,claude "your text" --why --capability`.
- Model tokenizer versions are pinned by repository id in `mti/registry.py`. Record the resolved tokenizer revision when publishing a table.

## 10. Ethics and Language Equity

Tokenizer inequity is a quiet tax on speakers of under-served languages: they pay more per query, wait longer, hit context limits sooner, and receive worse in-context performance, for structural reasons that have nothing to do with what they wrote. Making the tax visible, per language, per model, with an honest separation of measured and estimated numbers, is a small act of accountability. The tool anchors on English not to centre English but because English is, empirically, the reference every mainstream tokenizer is tuned toward; measuring distance from that anchor is the most direct way to quantify who is being under-served and by how much. All results, code, and data are released openly (Apache 2.0 for code, and public-domain UDHR text for the sample corpus) so the measurement can be checked, extended, and argued with.

---

## References

[1] IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs. arXiv:2511.03237.
[2] Sarvam-1 technical write-up. Sarvam AI. https://www.sarvam.ai/blogs/sarvam-1
[3] BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement. arXiv:2605.29379.
[4] IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages. arXiv:2404.16816.
[5] IndicLLMSuite (Sangraha, Setu). arXiv:2403.06350.
[6] Sennrich, Haddow, Birch. Neural Machine Translation of Rare Words with Subword Units (BPE). ACL 2016.
[7] Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation.
[8] Unicode Standard Annex 29: Unicode Text Segmentation. https://unicode.org/reports/tr29/
[9] Unicode Standard Annex 15: Unicode Normalization Forms. https://unicode.org/reports/tr15/
[10] Unicode Character Database (UCD). https://www.unicode.org/ucd/
[11] Unicode Common Locale Data Repository (CLDR). https://cldr.unicode.org/
[12] Universal Declaration of Human Rights, in Unicode. https://www.unicode.org/udhr/
[13] Project Bornomala Technical and Scientific Specification, Draft 1.0. Konko Maji, 2026.

## How to cite

```bibtex
@techreport{maji2026mothertongueindex,
  title       = {MotherTongueIndex: Measuring Multilingual Tokenizer Efficiency Against English},
  author      = {Maji, Konko},
  year        = {2026},
  institution = {Project Bornomala},
  type        = {Technical report},
  note        = {Subproject of Project Bornomala. Version 0.1.0},
  url         = {https://github.com/konkomaji/bornomala/tree/main/mothertongueindex}
}
```
