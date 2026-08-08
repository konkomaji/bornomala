# Project Bornomala

**A Bengali-First Language Technology Stack: Tokenization, Document Recognition, Dialect Documentation, Speech, and Foundation Modelling**

---

| | |
|---|---|
| **Programme** | Project Bornomala (বর্ণমালা) |
| **Principal Investigator** | Konko Maji |
| **Document** | Technical and Scientific Specification, Draft 1.0 |
| **Date** | 10 July 2026 |
| **Status** | Internal planning. Not for external circulation. |
| **Naming** | "Project Bornomala" is the only fixed name in this document. All sub-component, model, dataset, and benchmark names are marked `[TBD]` and will be assigned later. |

> **Verification note.** Every citation, dataset link, and reported metric in this document was checked against a publicly accessible source at time of writing. Figures that could not be verified are explicitly labelled *estimate*. No benchmark score is reported as measured unless it appears in a cited source. No keyword volumes, corpus sizes, or performance numbers are fabricated.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation and Scientific Framing](#2-motivation-and-scientific-framing)
3. [The Linguistic Problem, Formally Stated](#3-the-linguistic-problem-formally-stated)
4. [Prior Art: Systematic Review](#4-prior-art-systematic-review)
5. [Gap Analysis](#5-gap-analysis)
6. [Research Questions and Hypotheses](#6-research-questions-and-hypotheses)
7. [Build-From-Scratch versus Adaptation: Formal Decision](#7-build-from-scratch-versus-adaptation-formal-decision)
8. [Compute Model](#8-compute-model)
9. [Track A: Tokenization](#9-track-a-tokenization)
10. [Track B: Bengali Document Recognition](#10-track-b-bengali-document-recognition)
11. [Track C: West Bengal Dialect Documentation](#11-track-c-west-bengal-dialect-documentation)
12. [Track D: Speech](#12-track-d-speech)
13. [Track E: Foundation Model](#13-track-e-foundation-model)
14. [Evaluation Protocol](#14-evaluation-protocol)
15. [Infrastructure](#15-infrastructure)
16. [Roadmap and Phase Gates](#16-roadmap-and-phase-gates)
17. [Data Ethics, Consent, and Licensing](#17-data-ethics-consent-and-licensing)
18. [Risk Register](#18-risk-register)
19. [Publication and Partnership Strategy](#19-publication-and-partnership-strategy)
20. [Appendix A: Literature](#appendix-a-literature)
21. [Appendix B: Datasets and Access](#appendix-b-datasets-and-access)
22. [Appendix C: Tooling](#appendix-c-tooling)
23. [Appendix D: Organisational Landscape](#appendix-d-organisational-landscape)
24. [Appendix E: Notation and Glossary](#appendix-e-notation-and-glossary)

---

## 1. Abstract

Project Bornomala is a multi-year programme to construct a Bengali-first language technology stack. It rests on a single empirical claim, stated here so that it can be attacked:

> **The binding constraint on Bengali language modelling is neither compute nor architecture. It is (i) the absence of a large, clean, high-register Bengali corpus, because that corpus exists only as page images, and (ii) the total absence of any computational resource for the Bengali dialects spoken in West Bengal.**

Two consequences follow.

First, optical character recognition is not an application built on top of the language model. It is the instrument that manufactures the language model's training corpus. The causal arrow runs OCR → corpus → model, not model → OCR.

Second, dialect documentation is not a cultural gesture appended to the research programme. It is the only asset in the programme that a better-capitalised competitor cannot acquire by spending money, because it requires physical presence, local trust, and native competence in five dialect groups that no lab outside West Bengal can access.

Everything else in the programme (the tokenizer, the speech stack, the foundation model) is downstream of solving those two data problems.

Total GPU compute across all five tracks is estimated at **USD 5,000 to 10,000** over 33 months. This is not a compute-constrained programme. It is a data-labour and data-licensing programme. The compute figure is stated precisely because overstating it is the fastest way to be dismissed by a technically literate reviewer.

---

## 2. Motivation and Scientific Framing

### 2.1 The conflation to be avoided

There is a widespread and load-bearing confusion in discussions of non-English language models:

| Claim | Truth value | What it actually controls |
|---|---|---|
| "English models are better because of the tokenizer" | **Partially false** | Tokenization controls *cost*, *effective context length*, and *inference throughput* |
| "English models are better because of data volume" | **True** | Data volume and quality control *capability* |

Fixing tokenization for Bengali yields a 2x to 4x reduction in cost and a 2x to 4x increase in usable context. It does not yield frontier reasoning. Both problems are real. Conflating them produces a research plan that optimises the cheap problem and ignores the expensive one.

Project Bornomala addresses both, but sequences them correctly: the tokenizer is a four-month CPU deliverable; the corpus is a twenty-month data-engineering programme.

### 2.2 What "Bengali-first" means, operationally

Not: a multilingual model that happens to include Bengali.
Not: an English model with Bengali fine-tuning bolted on.

Rather, a system in which every design decision that could be made English-first or Hindi-first or pan-Indic-average is instead made Bengali-first:

- The tokenizer's vocabulary budget is allocated by Bengali morphology, not by the average of 22 Indic languages.
- The pre-tokenizer segments on Bengali grapheme clusters, not on Unicode codepoints.
- The pretraining corpus is weighted toward Bengali literary and formal register, not toward the natural distribution of Bengali web text.
- The evaluation metric is grapheme-cluster error rate, not codepoint character error rate.
- The dialect coverage includes the five West Bengal dialect groups, which currently have zero coverage anywhere.

### 2.3 Scope, and explicit non-scope

**In scope**

- A Bengali-only, morphology-aware subword vocabulary
- Open state-of-the-art Bengali document OCR, including pre-1950 letterpress
- The first computational resource for the West Bengal dialect groups
- Dialect-aware Bengali ASR and prosodically competent Bengali TTS
- A 2 to 4 billion parameter Bengali model, deployable on-device

**Explicitly out of scope**

- Matching frontier general reasoning capability. Frontier models are trained on tens of trillions of tokens. That is a category difference, not a funding gap.
- Pretraining a foundation model from random initialisation. See §7.2 for the formal argument that this option should be *deleted*, not deprioritised.
- Building novel ASR or TTS architectures. Existing open architectures are adapted.
- General-purpose multilingual OCR. Bengali script only, with code-mixed English as a secondary.
- Competing with Sarvam AI or Hishab. Both are candidate collaborators.

### 2.4 On the phrase "Claude-level efficiency"

This phrase should be retired from all external communication. It is unfalsifiable and it invites the correct rebuttal that the programme's budget is three orders of magnitude short.

The defensible, falsifiable claim is:

> A 2 to 4 billion parameter model that outperforms frontier models on Bengali-specific tasks (dialect comprehension, idiom, script handling, literary register, regional cultural knowledge), runs offline on a mid-range Android device at Q4 quantisation, and costs nothing per query.

Narrow-domain small models outperforming general large models *on their domain* is the expected result in the literature. It is not an aspiration. But the claim is worthless without the benchmark that adjudicates it, which is why every track in this programme ships a benchmark as a first-class deliverable.

---

## 3. The Linguistic Problem, Formally Stated

Bengali is not merely low-resource. Its structural properties violate assumptions embedded in the standard NLP toolchain. Three distinct problem classes, routinely conflated.

### 3.1 Orthographic and script-level (affects Tracks A and B)

Bengali script is an **abugida**, not an alphabet. A written unit is a consonant carrying an inherent vowel, modified by diacritics and combined into ligatures. Consequences:

| Property | Formal description | What it breaks |
|---|---|---|
| **Conjunct ligatures** (যুক্তাক্ষর) | A consonant cluster `C₁ C₂ [C₃]` joined by the virama (hasanta, U+09CD) renders as a single ligature glyph `G`. The visual form of `G` is frequently unrelated to the forms of `C₁`, `C₂`. 300 to 500 in common use. | Character-segmentation OCR; naive BPE merges; character-level CTC heads |
| **Non-linear vowel signs** (matra / kar) | For a base consonant `C` and vowel sign `v`, the rendered position of `v` may be above (`ি` U+09BF renders *left*), below, right, or split around `C`. **Visual order ≠ codepoint order.** e.g. `কি` = `ক` + `ি` in codepoints; `ি` renders to the left of `ক`. | Any OCR assuming reading order equals visual order; any pre-tokenizer splitting on codepoints |
| **Matra headline** (মাত্রা) | A horizontal stroke joins graphemes within an orthographic word. | Projection-profile and connected-component segmentation, which strip it and destroy glyph information |
| **Reph** (র্) | A preceding `র` + virama renders as a diacritic *above* the following consonant, reordered from logical position. | Grapheme cluster boundary detection; naive CER |
| **Phalas** (্য ya-phala, ্র ra-phala) | Subjoined consonant forms hanging below the base. | Vertical segmentation; bounding-box line recognisers |
| **Unicode ambiguity** | The same rendered glyph admits multiple codepoint encodings. ZWJ (U+200D) and ZWNJ (U+200C) are used inconsistently in the wild to force or suppress ligature formation. | **Every metric computed without prior NFC normalisation** |
| **Historical orthography** | Letterpress Bengali c. 1880 to 1950 uses glyph forms and spellings (e.g. older ৎ conventions, pre-reform spellings) absent from every modern font. | Every model trained only on modern-font synthetic data |

#### 3.1.1 The grapheme cluster, formally

Let `Σ` be the set of Bengali Unicode codepoints. A **grapheme cluster** is the maximal substring `g ∈ Σ*` that a native reader perceives as one written unit. Under UAX #29 extended grapheme cluster rules, a Bengali cluster has the schema:

```
g ::= [reph] C (virama C)* [vowel_sign] [nukta] [candrabindu | anusvara | visarga]
```

Crucially, `|g|` in codepoints ranges from 1 to roughly 8, while the reader perceives `g` as a single symbol.

**This is the single most important definition in the programme.**

#### 3.1.2 Consequence for evaluation

Let `H` be a hypothesis string and `R` a reference string.

- **Codepoint CER** = `Lev(H, R) / |R|` where `Lev` is edit distance over codepoints.
- **Grapheme-cluster error rate (GCER)** = `Lev(𝒢(H), 𝒢(R)) / |𝒢(R)|` where `𝒢(·)` segments into grapheme clusters.

A single wrong vowel sign changes one codepoint but changes the perceived word. Codepoint CER therefore **systematically understates Bengali error**. Published Bengali OCR numbers computed on raw codepoints are not comparable across papers and are, in several cases, optimistic by a wide margin.

> **Requirement B-1 (non-negotiable):** All Bengali error rates in this programme are reported as GCER, after NFC normalisation of both `H` and `R`. Codepoint CER may be reported only as a clearly labelled secondary figure.

Prior art exists for the correct treatment. **BnGraphemizer**, a trie-based tokenizer that segments Bengali label text into graphemes rather than characters, was introduced precisely because character-based tokenizers fail on Bengali handwritten text recognition: conjunct graphemes have visual representations "vastly different from the characters they comprise."

### 3.2 Morphological (affects Tracks A and E)

Bengali is moderately agglutinative with rich nominal and verbal inflection. Case suffixes, classifiers (`-টা`, `-টি`, `-খানা`), definiteness markers, and verb conjugations stack. Vocabulary is stratified:

| Stratum | Origin | Orthography | Register |
|---|---|---|---|
| **Tatsama** | Sanskrit, borrowed unchanged | Conservative, conjunct-heavy | Literary, formal, technical |
| **Tadbhava** | Evolved via Prakrit | Simplified | Vernacular, colloquial |
| **Deshi** | Substrate / Austroasiatic | Irregular | Rural, dialectal |
| **Perso-Arabic** | Mughal-era | Adapted | Administrative, everyday |
| **English** | Colonial and modern | Transliterated or code-mixed | Contemporary, technical |

**Consequence.** A BPE vocabulary induced on Bengali *web* text spends its budget on the tadbhava and English-loan strata and starves the tatsama stratum. But tatsama is exactly where literary and formal Bengali lives, and exactly where conjunct density is highest. This is a **corpus-composition problem masquerading as a tokenizer problem**, and it is why Track A's induction corpus must be deliberately over-weighted toward literary register, and why Track A is causally downstream of Track B.

### 3.3 Dialectal (affects everything; is the strategic opportunity)

Phonological classification identifies five principal Bengali dialect groups: **Eastern Bengali (Bangali), Manbhumi, Rangpuri, Varendri, and Rarhi**. Sukumar Sen's canonical five-part regional scheme names them **Rāṛhī, Jhaṛkhaṇḍī, Varendrī, Baṅgālī, Kāmrūpī**.

Standard Literary Bengali, in both its `sadhu` and `chalit` registers, derives from **Rarhi**, the dialect of the Hooghly, Nadia, and Kolkata belt.

Every Bengali NLP system in existence is trained on Standard Bengali. Almost every Bengali dialect resource in existence covers **Bangali** subvarieties spoken inside Bangladesh.

**The dialect that produced the literary standard has no computational resource of its own.** §5.2 quantifies this.

---

## 4. Prior Art: Systematic Review

Full citation tables with links: [Appendix A](#appendix-a-literature). Dataset access: [Appendix B](#appendix-b-datasets-and-access).

### 4.1 Tokenization

**Metric.** Token *fertility* is defined as

```
fertility(T, D) = |T(D)| / |words(D)|
```

for tokenizer `T` and corpus `D`, where `words(·)` is whitespace segmentation. Lower is better.

**Published findings.**

- Existing multilingual models exhibit fertility of **4 to 8 tokens per word** on Indic scripts, against roughly **1.4 for English**. Sarvam-1's purpose-built tokenizer (68,096 vocabulary, of which 4,096 reserved) achieves **1.4 to 2.1** across its ten supported Indic languages.
- **IndicSuperTokenizer** (arXiv:2511.03237) combines linguistically grounded pre-tokenization with a two-stage subword-to-superword induction (following SuperBPE), reporting a **39.5 percent** average fertility improvement over LLaMA4 and **18 percent** over Sutra, with **44 percent** higher inference throughput, across 22 Indic languages, English, and code.
- Cross-tokenizer fertility on curated benchmarks (VerChol, arXiv:2603.05883, Table 2):

| Language | Family | Best BPE | Llama-3.1 | GPT-4o | Gemma-2 |
|---|---|---|---|---|---|
| English | Germanic | 1.24 | 1.24 | 1.23 | 1.23 |
| Hindi | Indo-Aryan | 1.40 | 2.67 | 1.65 | 1.96 |
| Tamil | Dravidian | 2.17 | 12.39 | 3.17 | 4.19 |
| Kannada | Dravidian | 2.37 | 14.95 | 3.29 | 5.55 |
| Telugu | Dravidian | 2.14 | 13.30 | 3.06 | 4.57 |
| Malayalam | Dravidian | 2.85 | 16.26 | 3.52 | 5.88 |
| **Bengali** | **Indo-Aryan** | **absent** | **absent** | **absent** | **absent** |

> **Bengali does not appear in a single published cross-tokenizer fertility comparison.** That absence is Track A's first deliverable.

- **Fertility is insufficient as a sole metric.** *Single Token Retention Rate* (STRR), the proportion of words preserved as a single token, exposes vocabulary allocation across languages and domains where fertility cannot. Analysis across six tokenizers and seven languages finds stable English fertility, high Chinese fertility, and little domain sensitivity, motivating STRR as a complement.

```
STRR(T, D) = |{w ∈ words(D) : |T(w)| = 1}| / |words(D)|
```

- **Second-order effect.** IndicGenBench (arXiv:2404.16816) finds fertility ranging from 4.1 (Pashto) to 19.9 (Tibetan) and shows that high fertility means **fewer in-context examples fit inside a fixed context window**, which measurably degrades downstream performance on the affected language. Tokenization is therefore not only a cost problem but a capability problem at fixed context.

- **BrahmicTokenizer-131K** (arXiv:2605.29379) reports evaluation on a 27.12M-document corpus assembled from the public AI4Bharat and Sarvam stacks; **Bengali is the largest per-language slice at 745 million words**, indicating substantial Bengali text is available for vocabulary induction.

**Gap.** Every published Indic tokenizer optimises the average across 22 languages. Sarvam-1 allocates roughly 8 percent of its Indic data to Bengali against 28 percent for Hindi. No published tokenizer has been induced from Bengali text alone using a Bengali-specific pre-tokenizer that respects grapheme cluster boundaries, preserves conjunct integrity, and models the tatsama/tadbhava split. **BengaliBPE** is the only Bengali-specific tokenizer benchmark in the literature and it is a small study.

### 4.2 Pretraining corpora

| Corpus | Origin | Scale | Bengali share | Status |
|---|---|---|---|---|
| **Sangraha** (IndicLLMSuite) | AI4Bharat | 251B tokens / 22 languages | **~30B tokens** | Open, HF |
| — Sangraha Verified | | 64.3B tokens, 2.6× IndicCorp v2 | | Scraped from human-verified sites, OCR-extracted from Indic PDFs, transcribed from Indic video/podcast |
| — Sangraha Unverified | | | | Extracted from existing multilingual corpora; perplexity-filtered with n-gram LMs trained on Verified, extending CCNet |
| — Sangraha Synthetic | | | | English Wikimedia translated into 14 Indic languages, plus romanised transliteration |
| **IndicAlign** | AI4Bharat | 74.7M instruction pairs, 20 languages | | Open |
| **Bangla2B+** | BUET CSE | 27.5 GB, crawled from 110 Bengali sites | 100 percent | BanglaBERT pretraining set |
| **IndicCorp v2** | AI4Bharat | Superseded by Sangraha | | Open |
| **Common Corpus (English reference)** | | **~2T tokens** | 0 | Comparison baseline |

**The asymmetry:** ~30B cleaned Bengali tokens against ~2T English tokens. A **67× gap**.

**Gap.** Two observations that jointly define this programme.

1. Sangraha Verified explicitly includes *OCR-extracted data from high-quality Indic PDFs*. The quality ceiling of Bengali pretraining data is therefore **already partially set by the quality of Bengali OCR**. Improving Bengali OCR raises that ceiling directly and mechanically.

2. The entire published Bengali corpus stock is **web-derived and contemporary**. Bengali literary corpus from 1850 to 1950, which is where the language's register depth actually resides, is absent from every pretraining set because it exists only as page images. No amount of web crawling recovers it.

### 4.3 Bengali and Indic language models

| Model | Params | Origin | Method | Status | Role for Bornomala |
|---|---|---|---|---|---|
| **BanglaBERT** | ~110M | BUET CSE, Bangladesh | Encoder pretrained on Bangla2B+ | Released | Foundational. Baseline encoder. |
| **BanglaGPT** | Small | Salim et al. (2023) | Generative pretrained transformer | Released | Historical |
| **TituLLM** | 1B, 3B | **Hishab** (Singapore/Dhaka) + QCRI | "First large pretrained Bangla LLMs," released with benchmarking datasets, deliberately small due to compute constraints | Open, ACL 2025 Findings, `github.com/hishab-nlp/titulm` | **Closest prior art.** Direct comparison baseline. |
| **TigerLLM** | Small family | Raihan & Zampieri | Bangla LLM family | arXiv:2503.10995 | Reported to surpass open and proprietary models on standard Bengali benchmarks. Direct baseline. |
| **Sarvam-1** | 2B | Sarvam AI, India | Ground-up Indic pretraining on Sarvam-2T; custom 68k tokenizer; fertility 1.4–2.1 | Open weights | **Primary continued-pretraining base** |
| **Sarvam-30B / 105B** | 30B / 105B MoE | Sarvam AI | Reasoning models; 30B pretrained on ~16T tokens; trained in India | Open weights, released **6 March 2026**, on HF and AI Kosh | Strong. Do not compete directly. |
| **Sarvam-M** | 24B | Sarvam AI | Built on Mistral; ~1/3 of training samples in Indic; Hindi 28 percent, Bengali and 8 others 8 percent each | Open weights | Illustrates the Bengali allocation problem |
| **BharatGen Param-1** | 2.9B | Government-backed, India | Pretrained from scratch, 25 percent Indic data | Released | Alternative base. Useful provenance for public grants. |
| **OpenHathi / Airavata** | 7B | Sarvam / AI4Bharat | Llama-2 continued pretraining | **Superseded** | Obsolete. Do not build on. |

**Gap.** No Bengali-first model exists on the Indian side. Sarvam allocates Bengali 8 percent of Indic data. AI4Bharat treats it as 1 of 22. Hishab and BUET are Bangladesh-focused, using Standard Bangla with Dhaka-region conventions. **No model has been trained on Bengali literary corpus. No model handles West Bengal dialects.**

### 4.4 Dialect resources

| Resource | Type | Scale | Dialects | Origin |
|---|---|---|---|---|
| **Vashantor** (arXiv:2311.11142) | Parallel text, dialect → Standard Bangla | 32,500 sentences | Chittagong, Noakhali, Sylhet, Barishal, Mymensingh | Bangladesh |
| **BanglaDial** (DIB 2025) | Merged dialect text corpus | 60,729 sentence-level entries, 11 dialects + Standard | Chittagong, Sylhet, Barisal, Rangpur, others | Daffodil Intl. University, Bangladesh |
| **ONUBAD** | Parallel dialect text | 4 categories | Barisal, Sylhet, Chittagong, Standard | Bangladesh |
| **Bhashamul** | Competition data | Kaggle | Bangladesh regions | Bangladesh |
| **Ben-10** (arXiv:2510.23252) | Spontaneous speech | 78+ hours, 16,690 clips, 394 speakers, 155 topic stimuli, 34 annotators over 14 months, Silero VAD 30s segmentation, linguist-validated | Rangpur, Kishoreganj, Narail, Chittagong, Narsingdi, Tangail, Barishal, Habiganj, Sylhet, Sandwip | Bangladesh |
| **RegSpeech12** (arXiv:2510.24096) | Spontaneous speech | ~100 hours | Twelve Bangladesh dialects | Bangladesh |
| **BanglaTalk** (arXiv:2510.06188) | Real-time dialect speech assistance system | Uses RegSpeech12 | Bangladesh | Bangladesh |
| **BIDWESH** (arXiv:2507.16183) | Dialect hate-speech detection | 9,183 instances, manual translation from BD-SHS | Chittagong, Noakhali, Barishal | Bangladesh |
| **FeniVerse** | Trilingual parallel corpus | First dataset for the dialect | Feni | Bangladesh |
| **Bangla Regional Dialects Speech** | Speech | 5 regions, 19 speech categories | Bangladesh | Bangladesh |

### 4.5 Optical character recognition

#### 4.5.1 General state of the art, 2026

- **Vision-language models now lead** text extraction on complex documents, reporting **3× to 4× lower CER** than traditional engines on noisy scans, receipts, and distorted text. General-purpose frontier models beat dedicated OCR systems on real documents.
- **Open-weight VLMs are deployable.** dots.ocr (1.7B, 100+ languages), Surya 2 (650M), PaddleOCR-VL, Qwen3-VL (2B–235B). Vendor-reported OmniDocBench scores should be treated as a prior, not a result, and re-run privately.
- **Tiering is the correct architecture.** Fast traditional or small models for clean text; expensive inference reserved for hard cases.
- **Hallucination is the critical failure mode.** Unlike traditional OCR errors, VLM hallucinations produce plausible text that evades spell-checkers. In Bengali this is severe: a hallucinated Bengali sentence is grammatical, and no human reviewer catches it. **No Bengali OCR paper currently measures this.**

#### 4.5.2 Bengali-specific

| Resource | Type | Scale | Reported result | Status |
|---|---|---|---|---|
| **BN-HTRd** (arXiv:2206.08977) | Handwritten, document level | 788 pages, 150 writers, 108,147 words, 13,867 lines, 23,115 unique words, 574,203 characters; ground truth from BBC Bangla News corpus | Benchmark dataset | Public, Mendeley |
| **BanglaWriting** | Handwritten, word level | ~21,234 words; dual-channel scanner + smartphone | — | Public |
| **End-to-End OCR for Bengali Handwritten Words** (arXiv:2105.04020) | Method | Evaluated on BanglaWriting | **0.091 CER, 0.273 WER** (DenseNet121 + GRU) | Published |
| **BN-DRISHTI** | Line/word segmentation | 786 full pages, extended BN-HTRd | **F 99.97% line, 98% word** (YOLO + Hough + affine skew correction) | Springer |
| **BnGraphemizer** (PLATTER / CHIPS line) | Trie-based grapheme tokenizer | Method | Outperforms character-based tokenizers on Bengali HTR | **Adopt directly** |
| **CHIPS** | Page-level Indic handwritten OCR | 10 Indic languages, detection + recognition labels | PLATTER framework, 6 HTR models compared | Public, code + models |
| **Boise State Bangla Handwriting** | Isolated characters | — | Baseline recognizer | Public |
| **BanglaLekha-Isolated** | Isolated characters | — | Classification baselines | Public |
| **Tesseract Bengali** | Printed OCR engine | — | Weighted F1 **84.96%**, raised to **87.26%** by Levenshtein post-correction + character grouping | Public. Unusable for production. |

#### 4.5.3 Transferable precedents from adjacent scripts

| System | Language | Method | Result |
|---|---|---|---|
| **QARI-OCR v0.2** (arXiv:2506.02295) | Arabic | Fine-tuned a multimodal LLM on synthetic + real diacritically-rich text; **models and datasets released** | **WER 0.160, CER 0.061, BLEU 0.737.** Open-source SOTA. |
| **KITAB-Bench** (arXiv:2502.14949) | Arabic | 8,809 samples, 9 domains, 36 sub-domains, handwriting + tables + 21 chart types | VLMs beat traditional OCR by **~60% CER** on average; best model (Gemini-2.0-Flash) reaches only **65%** on PDF→Markdown |
| **ViLanOCR** (PMC11065407) | Urdu | Adapted multilingual vision-language transformers | **1.1% CER** on UHWR handwriting |
| **olmOCR-Bench** | English/Latin | Unit-test-driven, 1,403 pages, 6 categories including hallucinated/repeated n-gram penalisation and reading order | Standard for end-to-end document OCR eval |

> **The QARI-OCR + KITAB-Bench pair is a complete, published, reproducible template.** Arabic shares Bengali's structural problems: cursive joining, diacritics, ligatures, non-linear glyph composition. The Bengali version of that pair does not exist and is the single most tractable high-impact contribution available to this programme.

**Gap.** Nobody has trained a VLM on Bengali *document* text at scale. All Bengali OCR literature is handwriting-focused, character-segmentation-based, or Tesseract-derived. There is no Bengali OCR benchmark comparable to KITAB-Bench. No dataset covers pre-1950 West Bengal letterpress. No Bengali OCR paper reports hallucination rate.

### 4.6 Speech

| Resource | Type | Origin | Note |
|---|---|---|---|
| **IndicConformer** | ASR models, 22 languages | AI4Bharat | Permissive licence, production quality |
| **IndicVoices** | Spontaneous Indic speech | AI4Bharat | Bengali included |
| **Shrutilipi** | Mined ASR corpus from broadcast | AI4Bharat | Large; Bengali included |
| **Kathbath** | Read speech, 22 languages | AI4Bharat | Bengali included |
| **IndicF5 / IndicTTS** | TTS models and corpora | AI4Bharat | Bengali voice available |
| **Whisper-large-v3** | Multilingual ASR | OpenAI, open weights | Standard fine-tuning target |
| **Common Voice Bengali** | Crowd-sourced read speech | Mozilla | CC0 |
| **OpenSLR SLR53** | Bengali ASR corpus | SLR | Read speech |

**Gap.** Bengali ASR is trained overwhelmingly on *read* speech and skews toward Bangladeshi Standard Bengali conventions. WER on spontaneous West Bengal speech, on code-mixed Bengali-English, and on any Rarhi or Manbhumi audio, **is not measured anywhere, because no evaluation set exists.** Bengali TTS is intelligible and prosodically flat; sandhi realisation, interrogative intonation, and emotional register are unaddressed.

### 4.7 Benchmarks

**The claim "no Bengali benchmark exists" is false and should never be repeated.** What exists:

- **IndicGenBench, IndicXTREME, IndicGLUE**, and constituent tasks: IndicSentiment, IndicXNLI, IndicCopa, IndicXParaphrase, IndicWikiBio, IndicQA. All include Bengali.
- **TituLLM** shipped five Bengali benchmarking datasets alongside the models.
- Bengali mathematical reasoning: **BEnQA** (bilingual K-12), **Shomikoron**, **PatiGonit**, **BMWP**, **SOMADHAN**, **GanitLLM**.
- **BNLI** (arXiv:2511.08813), a linguistically refined Bengali NLI dataset.
- **MILU**; **IndQA** (OpenAI, November 2025) for Indian language reasoning.

**What does not exist, and is therefore in scope:**

1. A Bengali **OCR** benchmark with grapheme-aware metrics and hallucination measurement.
2. A **West Bengal dialect** comprehension and translation benchmark.
3. A Bengali **cultural, idiomatic, and literary-register** knowledge benchmark, distinct from translated English tests.

> Narrow the external claim from *"no Bengali benchmark exists"* to *"no Bengali OCR benchmark, no West Bengal dialect benchmark, and no native cultural-register benchmark exist."* The narrower claim is true, defensible, and rhetorically stronger.

---

## 5. Gap Analysis

### 5.1 Consolidated

| Domain | What exists | Who owns it | Unfilled gap | Tractability (2-person team) |
|---|---|---|---|---|
| Tokenizer | IndicSuperTokenizer, Sarvam-1, BrahmicTokenizer-131K, SuperBPE, MYTE, BengaliBPE | Sarvam, AI4Bharat, academic | Bengali-only, morphology-aware, grapheme-boundary-respecting vocabulary with published fertility and STRR | **High.** CPU only. Weeks. |
| Pretraining corpus | Sangraha (~30B Bengali), Bangla2B+, IndicCorp v2, CulturaX, FineWeb2 | AI4Bharat, BUET | Bengali literary corpus 1850–1950, currently trapped in page images | **Medium.** Gated on OCR. |
| OCR | Tesseract, Surya, BN-HTRd, BanglaWriting, CHIPS, BnGraphemizer | Academic, Bangladesh-heavy | VLM trained on Bengali document text; a Bengali OCR benchmark; pre-1950 WB letterpress | **High.** Method template exists (QARI-OCR). |
| Dialect text | Vashantor, BanglaDial, ONUBAD, FeniVerse, BIDWESH | Bangladesh universities | **All five West Bengal dialect groups. Nothing exists.** | **High.** Local access is the entire barrier. |
| Dialect speech | Ben-10 (78h), RegSpeech12 (~100h) | Bangladesh | All West Bengal dialect zones | **Medium.** Labour-intensive, cheap locally. |
| ASR / TTS | IndicConformer, IndicF5, Whisper, IndicVoices, Shrutilipi | AI4Bharat, OpenAI | Dialect-aware WB Bengali; spontaneous speech; prosody | **Medium.** Fine-tune, do not rebuild. |
| LLM | Sarvam-1/30B/105B, TituLLM, TigerLLM, Param-1 | Sarvam, Hishab, BharatGen | Bengali-first model with literary register and dialect competence | **Medium.** Gated on corpus. |
| Benchmarks | IndicGenBench, IndicXTREME, TituLLM suite, BNLI, MILU, IndQA | Google, AI4Bharat, Hishab | OCR benchmark; dialect benchmark; cultural-register benchmark | **High.** Zero compute. Pure domain expertise. |

### 5.2 The dialect coverage matrix

Legend: ● exists ◐ partial ○ **none**

| Dialect group | Zone | Parallel text | Speech corpus | Dialect ID | MT to Standard | Country |
|---|---|:---:|:---:|:---:|:---:|---|
| Bangali (Chittagong) | BD | ● | ● | ● | ● | Bangladesh |
| Bangali (Sylhet) | BD | ● | ● | ● | ● | Bangladesh |
| Bangali (Noakhali) | BD | ● | ◐ | ● | ● | Bangladesh |
| Bangali (Barishal) | BD | ● | ● | ● | ● | Bangladesh |
| Bangali (Mymensingh) | BD | ● | ◐ | ● | ● | Bangladesh |
| Rangpuri (BD side) | BD | ◐ | ● | ◐ | ◐ | Bangladesh |
| **Rāṛhī** | Hooghly, Nadia, Kolkata, Howrah, Burdwan, Birbhum | ○ | ○ | ○ | ○ | **India** |
| **Manbhumi / Jhaṛkhaṇḍī** | Purulia, Bankura, Jhargram, Medinipur div., N. Burdwan div.; into Santhal Pargana, Kolhan, N. Chotanagpur, Ranchi; adjoining Odisha | ○ | ○ | ○ | ○ | **India** |
| **Varendrī** | Malda division; adjoining Bihar/Jharkhand villages | ○ | ○ | ○ | ○ | **India** |
| **Sundarbani** | Presidency Division (and Khulna Division, BD) | ○ | ○ | ○ | ○ | **India / BD** |
| **Kāmrūpī / Rangpuri (Indian side)** | Cooch Behar, Jalpaiguri, Alipurduar | ○ | ○ | ○ | ○ | **India** |

**Every published Bengali dialect resource covers Bangali subvarieties inside Bangladesh.** The relevant literature notes that "systematic research on the computational processing of Bengali dialects remains limited." It understates the case. On the West Bengal side there is none.

### 5.3 The defensible position

- Everyone else is building **Bangla**: Bangladesh-centric, Standard, Dhaka-register. Sarvam gives Bengali 8 percent of its Indic mix. AI4Bharat gives it one slot in 22. Hishab and BUET serve Bangladesh.
- Nobody is building **Bengali as a whole language**, including its western half, its literary corpus, and its five undocumented dialect groups.
- The principal investigator is located **inside the Rarhi dialect zone**. Local access to speakers, to Bengali-native annotators at Indian labour cost, and to West Bengal archives and libraries, is a structural advantage that no lab in Bengaluru, Dhaka, or California can replicate at any budget.

---

## 6. Research Questions and Hypotheses

Each hypothesis is stated so that it can be falsified, with the falsifying observation named.

### RQ1 — Tokenization

**H1.** A Bengali-only BPE vocabulary induced with a grapheme-cluster-aware pre-tokenizer and a literary-weighted induction corpus achieves fertility ≤ 1.8 and STRR strictly greater than Sarvam-1 and IndicSuperTokenizer on held-out Bengali literary text.

- **Falsified if:** IndicSuperTokenizer or Gemma 3 achieves within 15 percent of the Bornomala tokenizer on both metrics.
- **Consequence of falsification:** Adopt IndicSuperTokenizer. Reallocate Track A effort to Track C. Publish the negative result; it is informative.

### RQ2 — The grapheme-cluster hypothesis

**H2.** Reporting GCER rather than codepoint CER materially changes the ranking of existing Bengali OCR systems.

- **Falsified if:** Rank correlation between GCER and codepoint CER across ≥ 8 systems exceeds Spearman ρ = 0.95.
- **Consequence of falsification:** The metric argument weakens. GCER remains correct but ceases to be a contribution.

### RQ3 — The corpus hypothesis (central)

**H3.** Continued pretraining on an OCR-recovered Bengali literary corpus (1850–1950) improves a base model's performance on Bengali literary-register, idiom, and cultural-knowledge tasks by a margin exceeding that obtained from an equivalent token budget of Bengali web text.

- **Falsified if:** With token budget held constant, literary-corpus continued pretraining does not outperform web-corpus continued pretraining on the cultural-register benchmark.
- **Consequence of falsification:** **This falsifies the central premise of Project Bornomala.** Publish it. It is a real and publishable negative result about Bengali, and it would redirect the field.

### RQ4 — The letterpress gap

**H4.** The performance gap between frontier VLMs and a Bengali-specialised model is small on modern print and large on pre-1950 letterpress.

- **Falsified if:** Frontier VLMs achieve GCER < 5 percent on historic Bengali letterpress in the month-12 pilot.
- **Consequence of falsification:** Track B narrows to a benchmark plus a cheap on-device tier. The corpus recovery still proceeds, using frontier APIs.

### RQ5 — Dialect transfer

**H5.** ASR fine-tuned on West Bengal dialect speech transfers positively to Bangladesh dialects, and vice versa, indicating a shared dialect-invariant acoustic representation.

- **Falsified if:** Cross-training degrades WER on either side relative to in-domain training.
- **Consequence of falsification:** The pan-Bengali resource argument weakens. Partnership pitch to Hishab and BUET must be reframed around data complementarity rather than model transfer.

### RQ6 — Small-model domain dominance

**H6.** A 2 to 4B Bengali-specialised model outperforms frontier models on the cultural-register benchmark.

- **Falsified if:** Frontier models win, at any parameter count.
- **Consequence of falsification:** The on-device product argument survives (cost, latency, offline). The capability argument does not.

---

## 7. Build-From-Scratch versus Adaptation: Formal Decision

### 7.1 Decision rule

> Train from random initialisation **iff** (a) the pretrained artifact encodes an assumption actively wrong for Bengali, **and** (b) retraining cost is small relative to the programme budget.
>
> Otherwise, adapt.

Applied:

| Component | Decision | (a) Wrong assumption? | (b) Cheap to retrain? | Base if adapting |
|---|---|---|---|---|
| **Tokenizer** | **FROM SCRATCH** | Yes. Every existing vocabulary is English-first or pan-Indic-average. | Yes. CPU, hours. | — |
| **OCR fast tier** (line recogniser) | **FROM SCRATCH** | Yes. No pretrained artifact encodes Bengali grapheme structure. | Yes. One 4090, ~30h, < USD 50. | — |
| **OCR layout detector** | ADAPT | No. YOLO-class detectors transfer across scripts. | — | YOLOv8/v11, DocLayout-YOLO |
| **OCR accurate tier** (VLM) | ADAPT | No. Vision encoders transfer. | No. Requires billions of image-text pairs. | Qwen3-VL 2B, dots.ocr 1.7B, Surya 2 650M |
| **ASR** | ADAPT | No. The gap is data (dialect, spontaneity), not architecture. | No. | IndicConformer, Whisper-large-v3 |
| **TTS** | ADAPT | No. Same reasoning. | No. | IndicF5, F5-TTS |
| **LLM base** | ADAPT (vocab surgery + continued pretraining) | Partially. Mitigated by vocabulary surgery. | No. See §7.2. | Sarvam-1 2B, BharatGen Param-1 2.9B, Gemma 3 4B |
| **Instruction tuning** | ADAPT (LoRA) | — | — | PEFT |
| **Preference alignment** | ADAPT (DPO, not RLHF) | — | — | TRL |

### 7.2 Why from-scratch pretraining is *deleted*, not deprioritised

Four arguments, in ascending order of force.

1. **The representational argument is already satisfied.** The case for from-scratch training was "Bengali should not be a translated afterthought." Sarvam-1 was built ground-up for Indic rather than adapted from an English model, achieving Indic fertility of 1.4–2.1. The representational property being sought already exists in an open-weight artifact.

2. **Vocabulary surgery *is* representational change.** Replacing the embedding matrix `E ∈ ℝ^(V×d)` and unembedding `U ∈ ℝ^(d×V)` with a Bengali-native vocabulary, then continuing pretraining on Bengali data, alters the model's internal representation of Bengali. It is not fine-tuning. The distinction between "from scratch" and "vocabulary surgery + continued pretraining" is one of degree, and the degree is bought at 1 percent of the cost.

3. **The scale comparison is a category error.** Sarvam-30B was pretrained on approximately **16 trillion tokens**. Matching that is not a funding gap.

4. **The differentiating asset is data, not weights.** Every hour spent on from-scratch pretraining is an hour not spent on the corpus that constitutes the actual moat. See §5.3.

---

## 8. Compute Model

### 8.1 FLOPs

For causal language modelling, forward + backward compute approximates to

```
C ≈ 6 · N · D          FLOPs
```

where `N` is non-embedding parameter count and `D` is training tokens. Wall time:

```
t = C / (P_peak · MFU · n_gpu)
```

where `P_peak` is per-GPU peak throughput at the training precision and MFU is model FLOPs utilisation. For bf16 on A100-80GB, `P_peak ≈ 312 TFLOP/s`; realistic MFU with FlashAttention and a competent dataloader is 0.35 to 0.50.

### 8.2 Applied

Assuming `MFU = 0.40`, effective `1.25 × 10^14` FLOP/s per A100, spot price USD 1.80/A100-hour.

| Run | N | D | C (FLOPs) | A100-hours | Cost |
|---|---|---|---|---|---|
| Continued pretrain, small | 2B | 10B | 1.2 × 10²⁰ | 100–150 | **USD 180–270** |
| Continued pretrain, mid | 4B | 20B | 4.8 × 10²⁰ | 400–600 | **USD 720–1,080** |
| Continued pretrain, large | 7B | 20B | 8.4 × 10²⁰ | 700–1,050 | **USD 1,260–1,890** |
| OCR VLM LoRA fine-tune | 2B | 5–10M samples | — | 400–800 | **USD 720–1,440** |
| OCR CTC recogniser | 650M | 20M synthetic lines | — | 1× RTX 4090, ~30h | **< USD 50** |
| ASR fine-tune | 1.5B | 500–1,000 h audio | — | 100–200 | **USD 180–360** |
| DPO | 2–4B | 2,000 pairs | — | 8–16 | **< USD 30** |

Multiply by **2×** for failed runs, restarts, checkpoint storage, and hyperparameter search.

### 8.3 Programme total

```
Total GPU compute:  USD 5,000 – 10,000  over 33 months
```

> **Any compute figure quoted above USD 50,000 for this programme is wrong**, and a technically literate reviewer will identify it as wrong. The budget is people, fieldwork, annotation, and archive access. Compute is under 10 percent of it. State this plainly. It is a credibility signal.

---

## 9. Track A: Tokenization

**Sub-project name:** `[TBD]`

### 9.1 Objective

A Bengali-only subword vocabulary that (i) respects grapheme cluster boundaries, (ii) preserves conjunct integrity, and (iii) allocates vocabulary across the tatsama and tadbhava strata proportionally to their use in *literary and formal* Bengali rather than in web text.

### 9.2 Method

#### Step 1 — Induction corpus

Assemble 5 to 20 GB of Bengali text, **deliberately over-weighted toward literary and formal register** relative to the natural web distribution.

| Source | Target share | Rationale |
|---|---|---|
| Public-domain Bengali literature | 30% | Tatsama-dense; conjunct-dense |
| Sangraha Bengali (Verified split) | 35% | Highest-quality available |
| Bengali Wikipedia | 10% | Encyclopaedic register |
| Government / administrative Bengali | 10% | Formal register, Perso-Arabic stratum |
| Contemporary news | 10% | Modern usage, named entities |
| Code-mixed Bengali-English | 5% | Realistic contemporary usage |

#### Step 2 — Normalisation and pre-tokenization

```
raw → NFC normalise → ZWJ/ZWNJ policy → grapheme cluster segmentation → pre-token boundaries
```

Requirements:

- **NFC normalise** before any processing.
- Adopt an explicit, documented ZWJ/ZWNJ policy. Do not silently strip them; they carry ligature-formation intent.
- **Never split a conjunct, a reph, a phala, or a matra from its base.** Adopt or extend the trie-based **BnGraphemizer** approach, which was designed for exactly this failure mode in Bengali HTR.
- Emit pre-token boundaries only at grapheme cluster boundaries, whitespace, and script transitions (Bengali ↔ Latin ↔ digit).

#### Step 3 — Vocabulary induction

Ablate:

| Axis | Values |
|---|---|
| Algorithm | BPE, Unigram (SentencePiece) |
| Vocabulary size | 16k, 32k, 48k, 64k |
| Superword stage | off, on (two-stage subword→superword, per SuperBPE / IndicSuperTokenizer) |
| Induction corpus | web-natural distribution vs literary-weighted |

The last axis is itself an experiment: it directly tests whether corpus composition, not algorithm, drives tatsama coverage.

#### Step 4 — Evaluation

Report **fertility** and **STRR** on a **held-out corpus not seen during induction**, disaggregated by register:

- Modern news
- Literary prose (Tagore, Bankim, Bibhutibhushan, Sarat Chandra)
- Formal administrative Bengali
- Code-mixed Bengali-English
- Each of the five dialect groups, once Track C data exists

Compare against: **Llama 3.1, Llama 4, GPT-4o, Gemma 3, Sarvam-1, IndicSuperTokenizer, BrahmicTokenizer-131K, Tesseract-adjacent baselines, BengaliBPE.**

Also report:
- **Bytes per token** (script-independent compression)
- **Conjunct fragmentation rate**: fraction of grapheme clusters split across token boundaries. Target: **0**.

### 9.3 Hard requirement: shaping validation

> **Requirement A-1.** Validate glyph shaping by rendering with HarfBuzz and reading the grapheme clusters back. Assert `𝒢(render_and_read(s)) == 𝒢(NFC(s))` for a large sample.

Naive rendering produces incorrect conjuncts. An incorrectly shaped induction corpus silently poisons the vocabulary and every downstream model. This is the **most common failure mode in Indic tokenizer and synthetic-data pipelines** and it is Gate G1 in §16.

### 9.4 Deliverables

1. Open-weight tokenizer (`[TBD]` name), HuggingFace `tokenizers` format.
2. **The first Bengali entry in a cross-tokenizer fertility and STRR comparison table.** Currently no such entry exists in any paper.
3. Grapheme-cluster pre-tokenizer, released as a standalone library.
4. Workshop paper.

### 9.5 Decision gate

If measured fertility for Gemma 3 or IndicSuperTokenizer on Bengali is already within ~15 percent of the Bornomala tokenizer, **H1 is falsified**. Adopt IndicSuperTokenizer and reallocate. Determine this on a CPU, in weeks, before spending any money.

---

## 10. Track B: Bengali Document Recognition

**Sub-project name:** `[TBD]`

### 10.1 Objective

Open state of the art on Bengali document OCR, with explicit coverage of pre-1950 West Bengal letterpress, and with **hallucination measured rather than ignored**.

### 10.2 Stage 1 — Synthetic data engine (CPU-only, months 2–6)

This stage runs entirely on the local Ryzen 3 machine and is the highest-leverage stage in the programme.

**Fonts.** Collect ≥ 200 Bengali fonts:
- Unicode: Kalpurush, SolaimanLipi, Nikosh, Hind Siliguri, Noto Serif Bengali, Noto Sans Bengali, Mukti, Lohit Bengali, Akaash, Siyam Rupali
- Historic typefaces where obtainable (digitised from letterpress specimens)

**Rendering.** For each text line `s` in the corpus, render across fonts, sizes, weights, using **HarfBuzz for correct shaping**. Round-trip validate per §9.3.

**Degradation stack.** Apply randomly sampled compositions of:

| Class | Operations |
|---|---|
| Compression | JPEG artifacts at varying quality |
| Blur | Gaussian, motion, defocus |
| Geometry | Skew (±5°), perspective warp, elastic distortion, curvature |
| Print physics | Ink spread, ink bloom, bleed-through from verso, show-through, uneven inking |
| Paper | Texture, yellowing, foxing, stains, fold shadows |
| Capture | Scanner noise, moiré from phone capture, uneven illumination, shadow gradients |
| Binarisation | Otsu artifacts, over-thresholding, speckle |
| Resolution | Downsample to 150, 100, 72 DPI equivalents |

**Volume.** 20 to 50 million synthetic line images. **Storage, not compute, is the binding constraint.** Budget 4 TB minimum, 8 TB preferred.

**Tooling.** Pillow + Raqm/HarfBuzz for rendering; `synthtiger` or `TextRecognitionDataGenerator` as scaffold; Albumentations for degradation. Write to WebDataset shards, not loose files.

### 10.3 Stage 2 — Real annotated data (months 4–12)

Synthetic alone plateaus. The residual error is real-scan distribution shift.

**Target.** 5,000 to 15,000 real page images with line-level ground truth, stratified:

| Category | Share | Sources |
|---|---|---|
| Modern printed books | 20% | Contemporary Bengali publishing |
| **1880–1950 letterpress** | **30%** | Bichitra scans; public-domain Tagore, Bankim, Sarat Chandra, Bibhutibhushan, Jibanananda; Bengali periodicals |
| Newspapers, multi-column | 15% | Archival newsprint |
| Forms, tables, records | 10% | Land and revenue records, municipal documents |
| Phone-camera captures | 15% | Field-collected |
| Handwriting | 10% | Reuse and extend BN-HTRd, BanglaWriting |

**Annotation protocol.**

1. Pre-label with a frontier VLM. Correction is roughly 5× faster than transcription.
2. **Verify the provider's terms of service** regarding use of outputs to train competing models *before* building a training set from them. This is a real constraint, not a formality.
3. All annotators Bengali-native.
4. Double-annotate a 10 percent sample. **Report inter-annotator agreement** at grapheme-cluster level.
5. Ground truth stored NFC-normalised.

### 10.4 Stage 3 — Models (rented GPU, months 9–14)

Two tiers. Ship both.

#### Tier 1: Fast

```
page image
  → layout / line detector  (YOLOv8/v11 or DocLayout-YOLO, fine-tuned)
  → line crops
  → line recogniser         (ViT or CRNN encoder + CTC head)
  → grapheme cluster sequence
  → NFC-normalised Unicode
```

- Parameters: ~650M total
- **The CTC head emits grapheme clusters, not codepoints.** This is the BnGraphemizer insight and it is the difference between a working Bengali recogniser and a broken one.
- Vocabulary of the CTC head = the set of grapheme clusters observed above a frequency threshold in the corpus, plus an UNK-decomposition fallback that spells rare clusters from their constituent codepoints.
- Precedent: BN-DRISHTI reports F 99.97% line and 98% word segmentation on Bengali using YOLO + Hough + affine skew correction.
- Deployment: CPU, on-device, batch digitisation at volume.

#### Tier 2: Accurate

- Fine-tune an open VLM (1.7 to 2B) on the full synthetic + real mix.
- Candidates: Qwen3-VL 2B, dots.ocr 1.7B, Surya 2 650M. Evaluate all three before committing.
- LoRA first; full fine-tune only if LoRA plateaus.
- **This is the QARI-OCR method**, which reached open-source SOTA for Arabic at WER 0.160 / CER 0.061.
- Handles: degraded scans, historic letterpress, layout, tables, handwriting.
- Deployment: single GPU or API.

#### Hallucination control

For corpus construction (Track E input), Tier 2 output is admitted only if:

1. Per-page hallucination score below threshold `τ` (see §10.6).
2. **Two-model agreement filtering**: Tier 1 and Tier 2 outputs agree above a GCER threshold on the same page, OR a second independent VLM agrees.

Pages failing admission are routed to human review or excluded.

### 10.5 Stage 4 — Bengali OCR benchmark (months 10–14)

**Sub-project name:** `[TBD]`. Modelled on KITAB-Bench and olmOCR-Bench.

**Categories:**

1. Modern printed books
2. Modern newspapers, multi-column
3. **Historic letterpress, 1880–1950, West Bengal presses**
4. Handwriting
5. Forms, tables, structured documents
6. Phone-camera capture, uneven lighting
7. Low-resolution and degraded scans
8. Code-mixed Bengali-English
9. Reading-order validation (multi-column)

**Baselines to run and publish, win or lose:**
Tesseract Bengali, Surya 2, PaddleOCR-VL, dots.ocr, Qwen3-VL, Gemini 3 Flash, Claude Opus, GPT-5.2, Mistral OCR.

### 10.6 Metric protocol (non-negotiable)

1. **NFC-normalise** hypothesis and reference before any comparison.
2. **GCER** is the headline metric. Codepoint CER may appear only as a labelled secondary figure.
3. **WER** at whitespace-word granularity.
4. **Hallucination rate:**

```
HR(H, R) = |{n-grams g ∈ H : g ∉ R}| / |n-grams(H)|,   n = 4
```

over grapheme clusters. **No Bengali OCR paper currently reports this.** It is the metric that matters most for corpus construction, because a hallucinated Bengali sentence is grammatical and passes every automatic filter.

5. **Reading-order accuracy** on multi-column pages.
6. **Report per-category. Never as a single aggregate number.** Aggregation hides exactly the failure this programme exists to fix.

### 10.7 The flywheel

```
scanned Bengali books, periodicals, records
        ↓  Track B
   clean, high-register Bengali corpus   ← the moat
        ↓
   Track A tokenizer (vocabulary induced on it)
        ↓
   Track E continued pretraining
        ↓
   better Bengali model → better OCR post-correction → better corpus
```

Track B is not a product built on the LLM. It is the instrument that manufactures the LLM's training data.

---

## 11. Track C: West Bengal Dialect Documentation

**Sub-project name:** `[TBD]`

### 11.1 Objective

The first computational resource for the five Bengali dialect groups spoken in West Bengal and adjoining Indian states. Text and speech. This is simultaneously the programme's data moat and its cultural-preservation deliverable.

### 11.2 Coverage plan

| Dialect group | Districts / zones | Text target (parallel pairs) | Speech target (hours) | Priority rationale |
|---|---|---|---|---|
| **Rāṛhī** | Hooghly, Nadia, Howrah, Kolkata, Burdwan, Birbhum | 12,000 | 150 | Prestige dialect. Basis of the literary standard. Locally accessible. **Start here.** |
| **Manbhumi / Jhaṛkhaṇḍī** | Purulia, Bankura, Jhargram, Medinipur div., N. Burdwan div.; Jharkhand and Odisha border villages | 10,000 | 120 | Most divergent from Standard. Highest linguistic information content. |
| **Varendrī** | Malda division; adjoining Bihar/Jharkhand villages | 6,000 | 80 | Cross-border continuum with Bangladesh Rajshahi. Enables RQ5. |
| **Sundarbani** | Presidency Division, Sundarbans belt | 6,000 | 80 | Shares features with both Bangali and Rarhi. Transitional. |
| **Kāmrūpī / Rangpuri (Indian side)** | Cooch Behar, Jalpaiguri, Alipurduar | 6,000 | 70 | Contested classification. **Document, do not adjudicate.** |
| **Total** | | **40,000** | **500** | |

### 11.3 Method

#### Text: parallel elicitation

Follow the **Vashantor** methodology (32,500 sentences, five Bangladesh regions), so the resulting corpus is:
- **parallel** to Standard Bengali, hence directly usable for dialect→standard MT and for dialect identification;
- **comparable** to Vashantor, BanglaDial, and ONUBAD, enabling the first pan-Bengali dialect resource.

Protocol:
1. Fix a stimulus set of Standard Bengali sentences covering the semantic and syntactic space (following Vashantor's construction).
2. Elicit dialect renderings from native speakers of each dialect.
3. **Cross-check every entry with a second native speaker of the same dialect** (FeniVerse protocol).
4. Record phonological, lexical, and syntactic divergence annotations.

#### Speech: spontaneous, topic-prompted

Replicate the **Ben-10** protocol exactly. It is published, validated, and it makes the two corpora directly comparable:

| Ben-10 parameter | Value | Bornomala target |
|---|---|---|
| Regions | 10 | 5 dialect groups, multi-site |
| Speakers | 394 | ≥ 400 |
| Topic stimuli | 155 (family, religion, sports, politics, etc.) | ≥ 155, adapted for WB context |
| Segmentation | Silero VAD, 30-second clips | Same |
| Total | 78+ hours, 16,690 clips | **500 hours** |
| Avg clip | 16.60 s | Same |
| Speech rate | 131.38 wpm | Report |
| Annotators | 34, over 14 months | ≥ 30 |
| Validation | Linguists | University linguistics partner |

Metadata per speaker: age band, gender, district, education level, first language, self-reported dialect, urban/rural. **Never the name.**

#### Transcription convention

Bengali script, with a **documented and published convention** for representing dialect phonology (e.g. how to represent Manbhumi vowel realisations absent from Standard orthography). Produce this jointly with a linguistics department. The convention is itself a deliverable.

### 11.4 Deliverables

1. Parallel dialect→Standard text corpus: five groups, ≥ 40,000 pairs. `[TBD]` name.
2. Transcribed spontaneous dialect speech: 500 hours. `[TBD]` name.
3. Dialect identification baseline; dialect→Standard MT baseline. Reported against BanglaDial and Vashantor conventions.
4. Published transcription convention for West Bengal Bengali dialect phonology.
5. Dialect evaluation benchmark. `[TBD]` name.

### 11.5 Why this is the moat

| Asset | Reproducible by a better-funded competitor? |
|---|---|
| A tokenizer | Yes. Weeks. |
| An OCR model | Yes. Anyone with the paper and a GPU. |
| A benchmark | Yes, once published. |
| **500 hours of consented, transcribed, dialect-labelled spontaneous speech from Purulia, Malda, and the Sundarbans** | **No.** Requires physical presence, local trust, native competence, and years. |

It is the only asset in this programme that money alone cannot buy.

---

## 12. Track D: Speech

**Sub-project name:** `[TBD]`

### 12.1 ASR

**Base.** Evaluate both IndicConformer (AI4Bharat) and Whisper-large-v3. Do not assume; measure.

**Two-stage fine-tuning.**

```
Stage 1:  base → fine-tune on existing Bengali corpora
          (IndicVoices, Shrutilipi, Kathbath, Common Voice Bengali, OpenSLR SLR53)

Stage 2:  → fine-tune on Track C dialect speech
          (+ Ben-10 and RegSpeech12 if licence permits, for RQ5)
```

**Reporting.** WER disaggregated by:
- Dialect group
- Spontaneous vs read speech (**separately** — aggregating them hides the entire problem)
- Code-mixing rate bucket
- Speaker demographic band

**Target (O4).** ≥ 30 percent relative WER reduction over the unmodified base on held-out West Bengal dialect audio.

**Architectural precedent.** BanglaTalk (arXiv:2510.06188) implements dialect-aware ASR feeding an LLM for real-time Bengali regional dialect speech assistance. Study its pipeline before building.

### 12.2 TTS

**Base.** IndicF5 or F5-TTS. Flow-matching TTS fine-tunes on 20 to 40 hours of studio audio.

**Registers to record:** neutral, narrative, interrogative, emotional. Dialect voices as a stretch goal once Track C speech exists.

**Evaluation.** MOS from native speakers, not automatic metrics alone. Report separately on:
- **Sandhi realisation** (Bengali sandhi is systematically mishandled by current TTS)
- **Interrogative intonation** (Bengali yes/no questions carry no segmental marker; intonation is the sole cue)
- Naturalness, intelligibility

### 12.3 On-device target

```
streaming ASR (~300M)  +  LLM (2–4B, Q4 GGUF)  +  compact TTS
        → end-to-end on a mid-range Android device
        → sub-second first-token latency
        → fully offline
```

**Frame the product as on-device and offline-capable, not as benchmark-topping.** In a region with uneven connectivity and inexpensive handsets, offline operation is the differentiating property. A leaderboard position is not.

---

## 13. Track E: Foundation Model

**Sub-project name:** `[TBD]`

### 13.1 Base selection

| Base | Params | For | Against |
|---|---|---|---|
| **Sarvam-1** | 2B | Ground-up Indic; fertility 1.4–2.1; cheapest to continue-pretrain; open weights | Bengali 8 percent of Indic mix; capacity ceiling |
| **BharatGen Param-1** | 2.9B | From-scratch, 25 percent Indic; government provenance useful for public grants | Less benchmarked; smaller ecosystem |
| **Gemma 3 4B** | 4B | 262k vocabulary handles Bengali reasonably; strong general capability | Not Indic-native; larger vocabulary surgery required |
| **Sarvam-30B** | 30B MoE | Strong reasoning; open weights since March 2026 | Competes with Sarvam; deployment cost defeats the on-device objective |

**Recommendation.** Sarvam-1 2B as primary, Gemma 3 4B as control. Run both. **The comparison is itself a result** (Indic-native small base vs general strong base, under identical Bengali continued pretraining).

### 13.2 Vocabulary surgery

Let the base have vocabulary `V_old`, embedding `E_old ∈ ℝ^(|V_old|×d)`. Let the Track A vocabulary be `V_new`.

For each new token `t ∈ V_new`, decompose it under the *old* tokenizer: `T_old(t) = [u₁, …, u_k]`. Initialise

```
E_new[t]  =  (1/k) · Σᵢ E_old[uᵢ]
U_new[t]  =  (1/k) · Σᵢ U_old[uᵢ]
```

Then:
1. Freeze the transformer body. Train `E_new`, `U_new` only, for a warm-up budget (typically 0.5 to 1 percent of total tokens) until embedding loss plateaus.
2. Unfreeze. Continue pretraining end-to-end.

This is standard practice and is not novel. The novelty is the vocabulary being installed and the corpus being trained on.

### 13.3 Continued pretraining

**Token budget.** 10 to 20B tokens.

**Corpus mix.**

| Source | Share | Rationale |
|---|---|---|
| **OCR-recovered literary and periodical Bengali, 1850–1950** (Track B output) | **20–30%** | The register no other model has. The moat. |
| Sangraha Bengali, Verified split | 30–40% | Highest-quality available web + PDF Bengali |
| Bengali Wikipedia, government, formal administrative text | 10% | Formal register, factual grounding |
| Contemporary Bengali news and journalism (licensed) | 10–15% | Contemporary usage, named entities |
| Track C dialect text | 3–5% | Dialect competence |
| English + Hindi replay | 5–10% | **Prevent catastrophic forgetting**; preserve reasoning transfer |
| Code and mathematics | 3–5% | Preserve reasoning capability |

**The literary share is the experiment.** RQ3 is tested by running an ablation with the literary share replaced by an equal token budget of Bengali web text.

**Hyperparameters (starting point, to be tuned).**
- bf16, FlashAttention-2
- Cosine schedule with warmup, peak LR ~10× lower than original pretraining
- Sequence length 4,096, extended later
- Gradient checkpointing; FSDP or DeepSpeed ZeRO-3
- Checkpoint every 500 steps to object storage (spot instances are preempted)
- Track MFU. **Below 0.30 MFU, the bottleneck is the dataloader, not the GPU.**

### 13.4 Instruction tuning and alignment

- **SFT:** LoRA on Bengali instruction data, including dialect comprehension tasks from Track C and literary-register tasks from Track B.
- **Alignment:** DPO on 500 to 2,000 native-speaker preference pairs. Runs overnight on one A100. **Full RLHF is not warranted at this scale.**
- **Quantisation:** Q4 GGUF. Measure on-device throughput on a mid-range Android device. Report tokens/second and first-token latency.

### 13.5 Cultural-register benchmark

**Sub-project name:** `[TBD]`. Categories:

1. Bengali idiom and proverb comprehension
2. Literary register discrimination (sadhu vs chalit)
3. West Bengal administrative and civic knowledge
4. Bengali-native reasoning tasks (**authored in Bengali, not translated from English**)
5. Dialect comprehension (from Track C)
6. Historical and literary knowledge (Bengal Renaissance, Bengali literature, regional history)

Model on MILU, IndQA, and the TituLLM benchmark suite for format. **Author natively.** Translated benchmarks measure translation, not competence.

**Named candidate for an off-the-shelf comparison point ahead of the natively-authored benchmark above: IndicGenBench** (Google Research, `google-research-datasets/indic-gen-bench`). 29 Indic languages including Bengali, four generation tasks (CrossSum-IN summarisation, Flores-IN translation, XQuAD-IN reading comprehension, XorQA-IN cross-lingual QA), 2.9k-14.5k examples per task, canary-stringed against accidental training-set contamination. It is translated/parallel-constructed, not natively authored, so it does not substitute for the six categories above (idiom, sadhu/chalit discrimination, dialect, native reasoning) - but it is real, immediately usable generation-quality signal for Gate G6's "beat an equal-token web-text control" comparison, available the day a continued-pretrained checkpoint exists, with no authoring lead time. Mixed licensing per task (CC BY-SA 4.0, MIT, CC BY-NC-SA 4.0 - check per task before any redistribution).

---

## 14. Evaluation Protocol

A programme that builds its own benchmarks must be unusually disciplined, or the benchmarks become self-serving and worthless.

### 14.1 Standing rules

| # | Rule |
|---|---|
| **E1** | Every benchmark is **released publicly, with a held-out test split and a submission protocol, before** the programme's own models are evaluated on it. |
| **E2** | Every benchmark reports baselines for competing systems, **run by this programme**, even where the programme's own system loses. |
| **E3** | **Negative results are published.** If RQ3 is falsified, that is a finding about Bengali and it is publishable. |
| **E4** | **No fabricated numbers.** No estimated benchmark scores presented as measured. Where a figure is an estimate, it carries the word *estimate*. |
| **E5** | Inter-annotator agreement reported for every human-annotated dataset. |
| **E6** | All metrics computed after **NFC normalisation**. All Bengali error rates at **grapheme-cluster granularity**. |
| **E7** | All results reported **per category**, never as a single aggregate. |
| **E8** | Every dataset is versioned. **A corpus without a version is not reproducible and not publishable.** |

### 14.2 Benchmarks produced

| Benchmark | Track | Categories | Prior art to model on |
|---|---|---|---|
| Bengali OCR benchmark `[TBD]` | B | Modern print, historic letterpress, handwriting, forms/tables, phone capture, low-res, reading order, code-mixed | KITAB-Bench, olmOCR-Bench |
| Dialect benchmark `[TBD]` | C | Dialect ID, dialect→standard MT, dialect comprehension QA, dialect ASR WER | Vashantor, BanglaDial, Ben-10 |
| Cultural-register benchmark `[TBD]` | E | Idiom, literary register, WB civic knowledge, native Bengali reasoning | MILU, IndQA, TituLLM suite |

---

## 15. Infrastructure

### 15.1 Local hardware verdict

**Stated configuration:** Ryzen 3 CPU, 16 GB RAM, NVIDIA GeForce GT 710 (2 GB VRAM).

> **The GT 710 cannot be used for any part of this programme.**
>
> It is a Kepler-generation card, compute capability **sm_35**. Current PyTorch CUDA builds compile kernels for **sm_50 and above**. Kepler is unsupported by the shipped wheels. Even were it supported, 2 GB of VRAM does not hold a 2B model's weights at 4-bit quantisation, let alone optimiser state.
>
> **Treat the machine as CPU-only:** Ryzen 3, 16 GB RAM.

This is **not a limitation for the first twelve months.** Track A and Track B Stage 1 are entirely CPU-bound. They are also the two highest-leverage stages in the programme.

**Never buy training hardware for this.** Rent by the hour. A workstation GPU costs more than the entire programme's compute budget and depreciates.

### 15.2 What runs locally

| Workload | Feasible? | Notes |
|---|---|---|
| Corpus ingestion, language ID, dedup, quality filtering | ✅ | MinHash LSH (`datasketch`); fastText `lid`; KenLM perplexity. Storage-bound, not compute-bound. |
| Tokenizer induction | ✅ | Overnight on 5–20 GB with input sharding |
| Synthetic OCR line rendering + degradation | ✅ | Embarrassingly parallel. **4 TB drive required.** |
| Benchmark authoring and annotation tooling | ✅ | Zero compute |
| Toy from-scratch LM (10–30M params, nanoGPT class) | ✅ (days) | **Only** to validate the tokenizer and dataloader. Not a research result. |
| Inference of Sarvam-1 2B @ Q4 via llama.cpp | ✅ | Usable speed on 16 GB RAM. Use for eyeballing eval sets. |
| Any training above ~100M params | ❌ | Rent |
| VLM fine-tuning | ❌ | Rent 4×–8× A100/H100 |

### 15.3 Rented compute

- **Providers:** RunPod, Vast.ai, Lambda Labs. Spot/community at ~USD 1.50–2.00 per A100-hour.
- **Indian alternatives, worth pricing:** E2E Networks, Yotta, Jarvislabs. Relevant if applying for **IndiaAI Mission** compute subsidy, which explicitly subsidises GPU access for Indian AI work.
- **Always checkpoint to object storage.** Spot instances are preempted.
- **bf16 + FlashAttention.** Track MFU. Below 0.30, fix the dataloader before renting more GPUs.

### 15.4 Data engineering

- Corpus stored as **Parquet or WebDataset shards**, never loose files.
- **Adopt Setu** (AI4Bharat's Indic cleaning, filtering, deduplication pipeline) rather than reimplementing. `Setu-translate` and `Setu-transliterate` are also available.
- Deduplication: MinHash LSH, then exact substring dedup.
- Quality filtering: KenLM perplexity filtering (following the CCNet-derived approach Sangraha uses) plus Gopher-style heuristics adapted for Bengali.
- **Version every dataset.**

---

## 16. Roadmap and Phase Gates

### 16.1 Timeline (months from start)

```
Track A (Foundations)
  A1  Bengali tokenizer                       [ 0 ── 4 ]
  A2  Corpus pipeline, dedup, filtering       [ 1 ────── 9 ]

Track B (OCR)
  B1  Synthetic OCR data engine               [ 2 ───── 8 ]
  B2  Real OCR annotation (WB letterpress)    [ 4 ────────── 12 ]
  B3  OCR models (CTC tier + VLM tier)        [ 9 ────── 14 ]
  B4  Bengali OCR benchmark                   [ 10 ───── 14 ]
  B5  Mass digitisation → corpus feedback     [ 13 ──────────── 21 ]

Track C (Dialect)
  C1  Dialect text corpus                     [ 6 ─────────── 15 ]
  C2  Dialect speech corpus (500h)            [ 9 ─────────────── 20 ]

Track D (Speech)
  D1  Dialect-aware ASR fine-tune             [ 18 ───── 23 ]
  D2  Bengali TTS fine-tune                   [ 20 ──── 24 ]

Track E (LLM)
  E1  Continued pretraining (2–4B)            [ 20 ────── 26 ]
  E2  SFT + DPO                               [ 25 ──── 29 ]
  E3  On-device voice stack                   [ 27 ───── 32 ]
```

**Nothing in the first eight months requires a GPU.**

### 16.2 Phase gates

| Gate | Month | Question | If NO |
|---|---|---|---|
| **G1** | 3 | Does the synthetic rendering pipeline produce correctly shaped conjuncts, verified by reading grapheme clusters back? | **Halt.** Nothing downstream works. Fix shaping. |
| **G2** | 4 | Does a Bengali-only tokenizer beat IndicSuperTokenizer and Gemma 3 on Bengali fertility and STRR by > 15 percent? | H1 falsified. Adopt IndicSuperTokenizer. Reallocate Track A to Track C. Publish the negative result. |
| **G3** | 6 | After dedup and quality filtering, what fraction of raw Bengali web text survives? Is the surviving corpus ≥ 5B clean tokens? | If < 5–10 percent survival and < 5B tokens, **OCR becomes the sole corpus route**, not a supplement. Re-weight the whole programme toward Track B. |
| **G4** | 12 | On the 50-page pilot benchmark, is the frontier-VLM-versus-Tesseract gap large on modern print but *enormous* on pre-1950 letterpress? | H4 falsified. Track B narrows to a benchmark plus a cheap on-device tier. Corpus recovery proceeds via frontier APIs. |
| **G5** | 18 | Has the dialect corpus reached 20,000 parallel pairs and 200 hours of speech? | The moat is not forming. Reassess whether field collection is executable at this scale, or partner. |
| **G6** | 26 | Does continued pretraining on the recovered literary corpus beat an equal-token web-text control on the cultural-register benchmark? | **RQ3 falsified. This is the central premise.** Publish it. It is a real negative result about Bengali and it redirects the field. |

### 16.3 First 30 days

1. Render 10,000 Bengali lines through **HarfBuzz** across 20 fonts. Read the grapheme clusters back. Verify conjunct shaping. **If wrong, nothing downstream works.**
2. Download Bengali Wikipedia and a Sangraha Bengali shard. Run dedup + quality filtering. **Measure the survival ratio on real data.**
3. Train a 32k SentencePiece BPE on the surviving text with a grapheme-cluster pre-tokenizer.
4. Measure **fertility and STRR** against Llama 3.1, Llama 4, Gemma 3, GPT-4o, Sarvam-1, IndicSuperTokenizer, BrahmicTokenizer-131K. **Publish the table. It does not currently exist.**
5. Assemble 50 real Bengali page scans, half modern print, half pre-1950 letterpress. Run Tesseract, Surya 2, dots.ocr, Gemini 3 Flash, Claude. Score **GCER after NFC normalisation**. Count hallucinated 4-grams.
6. **Publish that second table.**

> Between them, those two tables are the entire pitch, made with numbers instead of ambition. Neither requires a GPU. Neither costs money. Both can be produced on a Ryzen 3 in under a month.

---

## 17. Data Ethics, Consent, and Licensing

A cultural preservation programme that acquires its data badly is not a preservation programme.

### 17.1 Text and archives

| Source class | Constraint |
|---|---|
| Public-domain Bengali literature | Indian copyright subsists **60 years after the author's death**. Tagore (d. 1941), Bankim (d. 1894), Sarat Chandra (d. 1938), Bibhutibhushan (d. 1950), Jibanananda (d. 1954) are clear. **Confirm edition-specific typographic copyright** where a modern edition is scanned. |
| News archives (Anandabazar, Bartaman, Ei Samay, others) | Require licensing. **Begin these conversations in month one.** They take months and are the long pole. The programme does not depend on them; treat as upside. |
| Government documents | Check **GODL-India** (Government Open Data Licence) terms per source. |
| Social media text | **Do not scrape without lawful basis.** Reputational cost outweighs marginal token gain. |
| Frontier model outputs (annotation bootstrapping) | **Read the provider's terms** on training competing models before building a training set from them. A real constraint. |

### 17.2 Speech and dialect collection

| # | Requirement |
|---|---|
| 1 | **Written informed consent** from every speaker, in Bengali, stating that recordings will be publicly released and used to train models. |
| 2 | **Compensation at or above prevailing local rates.** Field recordists and annotators are the largest line item in the budget and should be. |
| 3 | Speaker anonymisation in released metadata: district and demographic band, **never name**. |
| 4 | A **withdrawal mechanism.** A speaker who asks to be removed is removed from subsequent releases. |
| 5 | **Institutional ethics review** if partnering with a university. Which is another reason to partner with one. |

### 17.3 Release licensing

- Corpora: **CC BY 4.0** or **CC BY-SA 4.0**
- Models: **Apache 2.0** or permissive equivalent
- Code: **Apache 2.0**

A restrictive licence on a cultural corpus produced from community speech is difficult to defend and destroys the credibility that is this programme's principal non-financial asset.

---

## 18. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | Conjunct shaping is silently wrong in the synthetic pipeline; all downstream models learn corrupted glyph-to-text mappings | Medium | **Catastrophic** | **Gate G1.** Round-trip validation via HarfBuzz before any model training. **The single largest technical risk in the programme.** |
| **R2** | A frontier model releases strong Bengali OCR, closing the gap before the programme ships | Medium | High | Ship the **benchmark first**. A benchmark retains value regardless of who wins it. The pre-1950 letterpress category is unlikely to be addressed by any general-purpose vendor. |
| **R3** | Dialect field collection stalls: access, trust, logistics, annotator turnover | **High** | High | Start with Rarhi (local, accessible, highest value). **Prove the pipeline on 50 hours before committing to 500.** Partner with Jadavpur or ISI Kolkata for fieldwork methodology and ethics cover. |
| **R4** | News archive licensing fails | High | Medium | The programme does not depend on it. Literary + public-domain + Sangraha is sufficient. |
| **R5** | Sarvam or Hishab releases a Bengali-first model, removing the LLM differentiator | Medium | Medium | The differentiator is the **corpus and the dialect data**, not the weights. If they release, offer the corpus as a partnership. That is a win. |
| **R6** | Solo or two-person execution capacity exceeded | **High** | High | Sequence strictly. Track A and Track B Stage 1 are single-person CPU tasks. **Do not begin Track C fieldwork before G3. Do not begin Track E before G4.** |
| **R7** | Benchmark self-serving bias: the programme builds a benchmark its own model happens to win | Medium | High (reputational) | **Rule E1.** Release benchmark and held-out split before evaluating own models. Invite external submissions. Report competitor baselines even when they win. |
| **R8** | Compute cost overrun | Low | **Low** | Compute is < 10 percent of total budget. Even a 3× overrun is immaterial. **This risk is routinely overweighted and should not drive decisions.** |
| **R9** | Hallucinated OCR output silently poisons the pretraining corpus | Medium | High | Hallucination rate is a first-class metric (§10.6). Corpus admission threshold on per-page hallucination score. Two-model agreement filtering on admitted pages. |
| **R10** | Ethics or consent failure in dialect collection | Low | **Catastrophic (reputational)** | §17.2. University partnership provides institutional review. Written consent, non-negotiable. |

---

## 19. Publication and Partnership Strategy

### 19.1 Publication sequence

| Order | Output | Venue class | Why it comes when it does |
|---|---|---|---|
| 1 | Bengali tokenizer + fertility/STRR table | Workshop (WILDRE, ICON, *ACL workshop) | Cheapest to produce; establishes presence; weeks not months |
| 2 | Bengali OCR benchmark | Dataset track (LREC, ACL Findings) or *Data in Brief* | Benchmarks accrue citations independent of model results |
| 3 | Bengali document OCR model | ACL, EMNLP, or ICDAR | **Follows the benchmark, not the reverse** |
| 4 | Dialect corpus | LREC, *Data in Brief* | Direct precedent: Vashantor, BanglaDial, Ben-10, FeniVerse all published exactly here |
| 5 | Dialect-aware Bengali ASR | Interspeech, ICASSP | Requires the corpus |
| 6 | Foundation model + cultural benchmark | ACL, NeurIPS Datasets & Benchmarks | Last, and only if earlier results justify it |

### 19.2 Partnership targets, in order of approach

| Organisation | Why | What to offer | What to ask for |
|---|---|---|---|
| **AI4Bharat, IIT Madras** | Open tooling (Sangraha, Setu, IndicConformer, IndicF5); permissive licensing; history of external collaboration | **West Bengal dialect data, which they do not have and cannot easily collect** | Compute, Setu pipeline support, co-authorship, credibility |
| **Hishab** (Dhaka / Singapore) | TituLLM authors. Most serious commercial Bengali LLM effort. They publish and release. | Complementary data: their Bangladesh dialect coverage + WB coverage = **the first pan-Bengali resource** | Data exchange, benchmark alignment. **Not competition.** |
| **BUET CSE NLP** | BanglaBERT, Bangla2B+. Academic centre of gravity for Bengali NLP. | Co-authorship on a pan-Bengali dialect resource | Methodology, review, standing |
| **Jadavpur University; ISI Kolkata** | Bengali linguistics faculty. No ML pipeline. **Exactly the complementary half.** | The engineering pipeline | Dialect phonology expertise, fieldwork methodology, ethics review, student annotators |
| **IndiaAI Mission** | Explicit GPU compute subsidy; sovereign-AI policy priority | A culturally significant, publishable, open-source deliverable in a scheduled language | Subsidised compute, grant funding |
| **Bhashini** | Government DPI for Indian languages | Dialect ASR and OCR models | Distribution, institutional legitimacy |
| **WB state archives, Bichitra, university libraries** | Custodians of the scanned corpus | **Free digitisation and OCR of their holdings** | Scanning access |

### 19.3 Framing for a funder

> **Do not pitch "a Bengali LLM."** That pitch invites the immediate and correct rebuttal that Sarvam already ships one.
>
> **Pitch this:** "The Bengali literary corpus does not exist in machine-readable form, and the dialects of West Bengal have no computational resource at all. We are building the OCR to recover the first and the fieldwork to create the second. The language model is what those two assets make possible, not the other way around."
>
> **Total compute across the programme is under USD 10,000.** The budget is people, fieldwork, annotation, and archive access. Say this plainly. A funder who hears an honest, small compute number and a large data-labour number is being told the truth about how this class of work actually goes.

### 19.4 Note on the competitive framing

Bangladesh has more Bengali NLP activity than India does, by a wide margin, and it is concentrated on Standard Bangla and Bangladesh regional dialects. India has more Indic ML infrastructure, and it treats Bengali as one language among twenty-two.

Neither side covers West Bengal dialect. Neither side has digitised the Bengali literary corpus.

That is the entire opportunity, **and it is not a competitive opportunity. It is a complementary one.** The correct posture toward both Hishab and AI4Bharat is partnership, and the currency for that partnership is data neither of them can collect.

---

## Appendix A: Literature

### A.1 Tokenization

| Work | Identifier | Relevance |
|---|---|---|
| IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs | [arXiv:2511.03237](https://arxiv.org/abs/2511.03237) | **Current SOTA.** Subword + superword, language-specific pre-tokenization. 39.5% fertility gain over LLaMA4, 18% over Sutra, 44% throughput gain. **Read first, twice.** |
| Sarvam-1 technical write-up | [sarvam.ai/blogs/sarvam-1](https://www.sarvam.ai/blogs/sarvam-1) | 68,096 vocab (4,096 reserved). Fertility 1.4–2.1 across 10 Indic languages. Reference implementation. |
| Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation | ResearchGate 396459552 | **Single Token Retention Rate.** Required second metric. |
| Evaluating Subword Tokenization Techniques for Bengali: A Benchmark Study with BengaliBPE | ResearchGate 397441197 | The **only** Bengali-specific tokenizer benchmark in the literature. Direct baseline. |
| BrahmicTokenizer-131K: An Indic-Capable Drop-In Replacement for o200k_base | [arXiv:2605.29379](https://arxiv.org/abs/2605.29379) | Evaluated on 27.12M docs from the public AI4Bharat + Sarvam stack. **Bengali is the largest slice at 745M words.** |
| IndicGenBench | [arXiv:2404.16816](https://arxiv.org/abs/2404.16816) | Fertility 4.1 (Pashto) to 19.9 (Tibetan). Shows high fertility → fewer in-context examples fit → degraded performance. |
| SuperBPE (Liu et al., 2025) | See IndicSuperTokenizer refs | The superword induction mechanism IndicSuperTokenizer builds on |
| MYTE: Morphology-Driven Byte Encoding (Limisiewicz et al., 2024) | ACL 2024 | Most interesting non-BPE direction. Equitable multilingual byte encoding. |
| Ahia et al. 2023; Petrov et al. 2023 | — | Tokenizer cost disparity. The economic-inequity framing. Useful for grant narrative. |
| VerChol: Grammar-First Tokenization for Agglutinative Languages | [arXiv:2603.05883](https://arxiv.org/abs/2603.05883) | Source of the cross-tokenizer fertility table in §4.1 |
| Regional Tiny Stories: Small Models to Compare Language Learning and Tokenizer Performance | [arXiv:2504.07989](https://arxiv.org/abs/2504.07989) | Method for evaluating tokenizer effects with tiny models. **Directly runnable on the local CPU machine.** |

### A.2 Bengali and Indic language models

| Work | Identifier | Relevance |
|---|---|---|
| TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking | [arXiv:2502.11187](https://arxiv.org/abs/2502.11187) | Hishab + QCRI. 1b and 3b. ACL 2025 Findings. `github.com/hishab-nlp/titulm`. **Closest prior art.** |
| TigerLLM: A Family of Bangla Large Language Models | [arXiv:2503.10995](https://arxiv.org/abs/2503.10995) | Raihan & Zampieri. Reported to surpass open and proprietary models on Bengali benchmarks. |
| BanglaBERT | [arXiv:2101.00204](https://arxiv.org/abs/2101.00204) | BUET CSE. Bangla2B+, 27.5 GB from 110 sites. Foundational. |
| IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages | [arXiv:2403.06350](https://arxiv.org/abs/2403.06350) | Sangraha (251B tokens), IndicAlign (74.7M pairs), Setu pipeline. **Essential.** |
| Open-Sourcing Sarvam 30B and 105B | [sarvam.ai/blogs/sarvam-30b-105b](https://www.sarvam.ai/blogs/sarvam-30b-105b) | Released 6 March 2026. Open weights on HF and AI Kosh. 30B pretrained on ~16T tokens. |
| Evaluating LLMs' Multilingual Capabilities for Bengali: Benchmark Creation and Performance Analysis | [arXiv:2507.23248](https://arxiv.org/abs/2507.23248) | Source for Sangraha's Bengali share (~30B tokens vs ~2T English). Good survey of the Bengali LLM gap. |
| Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus | [arXiv:2410.14815](https://arxiv.org/abs/2410.14815) | Nemotron-Mini-Hindi. **Method template for continued pretraining.** Includes Sarvam-1 vs Gemma-2 vs OpenHathi comparison. |
| GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO | [arXiv:2601.06767](https://arxiv.org/abs/2601.06767) | Current survey of Bengali reasoning work: Shomikoron, PatiGonit, BMWP, SOMADHAN, BEnQA |
| BNLI: A Linguistically-Refined Bengali Dataset for Natural Language Inference | [arXiv:2511.08813](https://arxiv.org/abs/2511.08813) | Bengali NLI. Evaluation resource. |

### A.3 Dialect

| Work | Identifier | Relevance |
|---|---|---|
| Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects | [arXiv:2311.11142](https://arxiv.org/abs/2311.11142) | 32,500 sentences, 5 Bangladesh regions. **The methodology to replicate for West Bengal.** |
| BanglaDial: A merged and imbalanced text dataset for Bengali regional dialect analysis | [doi:10.1016/j.dib.2025.112200](https://doi.org/10.1016/j.dib.2025.112200) | 60,729 sentences, 11 dialects + Standard. Daffodil Intl. University. Published in *Data in Brief*, **the venue to target.** |
| RegSpeech12: A Regional Corpus of Bengali Spontaneous Speech Across Dialects | [arXiv:2510.24096](https://arxiv.org/abs/2510.24096) | ~100 hours, 12 Bangladesh dialects. Also the source for the five-group phonological classification. |
| Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages? (**Ben-10**) | [arXiv:2510.23252](https://arxiv.org/abs/2510.23252) | 78+ h, 394 speakers, 155 topic stimuli, Silero VAD, 34 annotators / 14 months. **The field protocol to copy exactly.** |
| BanglaTalk: Towards Real-Time Speech Assistance for Bengali Regional Dialects | [arXiv:2510.06188](https://arxiv.org/abs/2510.06188) | Dialect-aware ASR + LLM pipeline. **Direct architectural precedent for Track D.** |
| BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset | [arXiv:2507.16183](https://arxiv.org/abs/2507.16183) | 9,183 instances, 3 dialects, manual translation from BD-SHS |
| FeniVerse: A parallel corpus of Feni dialect, standard Bengali, and English | PMC12666053 | Single-dialect trilingual corpus. **Proof that a small, focused dialect corpus is publishable.** |
| Bengali dialects (classification reference) | [en.wikipedia.org/wiki/Bengali_dialects](https://en.wikipedia.org/wiki/Bengali_dialects) | Sukumar Sen's five-part scheme: Rāṛhī, Jhaṛkhaṇḍī, Varendrī, Baṅgālī, Kāmrūpī. **Cite Sen, not Wikipedia.** |
| Oral to Web: Digitizing 'Zero Resource' Languages of Bangladesh | [arXiv:2603.05272](https://arxiv.org/abs/2603.05272) | 85,792 entries, ~107 h, 42 language varieties, 9 districts, 77 speakers, 43 validators. Fieldwork protocol reference. |

### A.4 OCR

| Work | Identifier | Relevance |
|---|---|---|
| **QARI-OCR: High-Fidelity Arabic Text Recognition through Multimodal LLM Adaptation** | [arXiv:2506.02295](https://arxiv.org/abs/2506.02295) | **THE TEMPLATE.** WER 0.160, CER 0.061, BLEU 0.737. Open-source SOTA. Models and datasets released. **Copy the method.** |
| **KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding** | [arXiv:2502.14949](https://arxiv.org/abs/2502.14949) | 8,809 samples, 9 domains, 36 sub-domains, 21 chart types. VLMs beat traditional OCR by ~60% CER. Best model 65% on PDF→Markdown. **The benchmark template.** |
| Adapting multilingual vision language transformers for low-resource Urdu OCR (ViLanOCR) | PMC11065407 | 1.1% CER on Urdu handwriting. Nastaliq is harder than Bengali. **Proof of tractability.** |
| BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla HTR and Line Segmentation | [arXiv:2206.08977](https://arxiv.org/abs/2206.08977) | 788 pages, 150 writers, 108,147 words, 13,867 lines. Ground truth from BBC Bangla News corpus. |
| End-to-End Optical Character Recognition for Bengali Handwritten Words | [arXiv:2105.04020](https://arxiv.org/abs/2105.04020) | 0.091 CER, 0.273 WER (DenseNet121 + GRU) on BanglaWriting. **Honest current baseline. Beatable.** |
| BN-DRISHTI: Bangla Document Recognition through Instance-level Segmentation of Handwritten Text Images | Springer 10.1007/978-3-031-41501-2_14 | YOLO + Hough + affine skew correction. F 99.97% line, 98% word. **Adopt for layout.** |
| BnGraphemizer (within the PLATTER / CHIPS line of work) | See Boise State / PLATTER refs | **Trie-based grapheme tokenizer.** Character-based tokenizers fail on Bengali HTR because conjunct graphemes have visual forms unrelated to constituent characters. **Adopt directly for the CTC head.** |
| olmOCR-Bench | Poznanski et al. 2025; Taghadouini et al. 2026 | 1,403 pages, 6 categories. Unit-test-driven. Penalises hallucinated/repeated n-grams. **Adopt the hallucination methodology.** |
| The Definitive Guide to OCR in 2026: From Pipelines to VLMs | Practitioner survey, March 2026 | Current landscape: VLMs lead with 3–4× lower CER; dots.ocr (1.7B) and Qwen3-VL are deployable open baselines; **hallucination evades spell-checkers.** |

---

## Appendix B: Datasets and Access

### B.1 Text corpora

| Dataset | Access | Scale | Licence |
|---|---|---|---|
| **Sangraha** (AI4Bharat) | [github.com/AI4Bharat/IndicLLMSuite](https://github.com/AI4Bharat/IndicLLMSuite) | 251B tokens / 22 langs; **~30B Bengali**; Verified split 64.3B | Check repo; HF mirror |
| **IndicAlign** | same repo | 74.7M instruction pairs, 20 languages | Check repo |
| **Setu** (cleaning/dedup pipeline) | same repo | Pipeline + Setu-translate + Setu-transliterate | Open |
| **IndicCorp v2** | [indicnlp.ai4bharat.org](https://indicnlp.ai4bharat.org/) | Superseded by Sangraha; useful for ablation | Per-source |
| **Bangla2B+** (BanglaBERT) | [arXiv:2101.00204](https://arxiv.org/abs/2101.00204) | 27.5 GB, 110 Bengali sites | Research use |
| **Bengali Wikipedia** | [dumps.wikimedia.org](https://dumps.wikimedia.org/) | ~120M tokens (*estimate*) | CC BY-SA |
| **CulturaX / FineWeb2** Bengali subsets | Hugging Face | Web-derived; requires heavy filtering | Per-source |
| **Bichitra** (Tagore variorum) | bichitra.jdvu.ac.in (Jadavpur University) | Scanned Tagore corpus. **Images, not text. Track B input.** | Verify per work |

### B.2 Dialect datasets

| Dataset | Access | Scale |
|---|---|---|
| **Vashantor** | [data.mendeley.com/datasets/bj5jgk878b/2](https://data.mendeley.com/datasets/bj5jgk878b/2) | 32,500 sentences, 5 BD regions, parallel to Standard |
| **ONUBAD** | [data.mendeley.com/datasets/6ft99kf89b/2](https://data.mendeley.com/datasets/6ft99kf89b/2) | 4 dialect categories |
| **BanglaDial** | Mendeley, doi 10.17632/sx6ybcps2n.2 | 60,729 sentences, 11 dialects + Standard |
| **Bhashamul** | [kaggle.com/competitions/regipa/data](https://www.kaggle.com/competitions/regipa/data) | Competition data |
| **Bangla Dialect Dataset** | [data.mendeley.com/datasets/sm63ryv5dt/1](https://data.mendeley.com/datasets/sm63ryv5dt/1) | Dialect text |
| **Bangla Regional Dialects Speech** | [data.mendeley.com/datasets/777wsgjgtm/1](https://data.mendeley.com/datasets/777wsgjgtm/1) | 5 BD regions, 19 speech categories |
| **Ben-10** | [arXiv:2510.23252](https://arxiv.org/abs/2510.23252) | 78+ h, 16,690 clips, 394 speakers, 10 regions |
| **RegSpeech12** | [arXiv:2510.24096](https://arxiv.org/abs/2510.24096) | ~100 h, 12 BD dialects |

### B.3 OCR datasets

| Dataset | Access | Scale |
|---|---|---|
| **BN-HTRd** | [data.mendeley.com/datasets/743k6dm543](https://data.mendeley.com/datasets/743k6dm543/1) | 788 pages, 150 writers, 108,147 words, 13,867 lines, 23,115 unique words, 574,203 chars |
| **BanglaWriting** | Mendeley; see [arXiv:2105.04020](https://arxiv.org/abs/2105.04020) | ~21,234 handwritten words; scanner + smartphone |
| **Boise State Bangla Handwriting** | Boise State University | Isolated characters |
| **BanglaLekha-Isolated** | Mendeley (Biswas et al. 2017) | Isolated characters |
| **CHIPS** | Released with PLATTER framework (code + models) | Page-level, 10 Indic languages, detection + recognition |
| **BN-DRISHTI extended BN-HTRd** | Springer, see A.4 | 786 full pages, line + word segmentation annotation |

### B.4 Speech datasets

| Dataset | Source | Note |
|---|---|---|
| IndicVoices | AI4Bharat | Spontaneous Indic speech, Bengali included |
| Shrutilipi | AI4Bharat | Mined from broadcast; large |
| Kathbath | AI4Bharat | Read speech, 22 languages |
| Common Voice Bengali | Mozilla | **CC0.** Crowd-sourced read speech. |
| OpenSLR SLR53 | openslr.org | Bengali ASR corpus, read speech |
| IndicTTS / IndicF5 corpora | AI4Bharat | TTS training audio, Bengali voice |

---

## Appendix C: Tooling

| Purpose | Choice | Note |
|---|---|---|
| LLM base (primary) | Sarvam-1 2B | Indic-native, low fertility, open weights |
| LLM base (control) | Gemma 3 4B; BharatGen Param-1 2.9B | **Run both. The comparison is a result.** |
| OCR VLM base | Qwen3-VL 2B; dots.ocr 1.7B; Surya 2 650M | Evaluate all three before committing |
| OCR layout detection | YOLOv8 / v11; DocLayout-YOLO | BN-DRISHTI precedent |
| OCR fast tier | ViT or CRNN encoder + CTC head **over grapheme clusters** | Train from scratch |
| ASR base | IndicConformer; Whisper-large-v3 | Evaluate both |
| TTS base | IndicF5; F5-TTS | Flow-matching; fine-tunes on 20–40 h |
| Tokenizer training | SentencePiece; HuggingFace `tokenizers` | Plus custom grapheme-cluster pre-tokenizer |
| **Text shaping (critical)** | **HarfBuzz**, via `uharfbuzz` or Pillow + Raqm | **Round-trip validate.** Naive rendering produces wrong conjuncts. |
| Synthetic image generation | `synthtiger`; `TextRecognitionDataGenerator`; Albumentations | Degradation stack |
| Corpus cleaning | **Setu** (AI4Bharat) | Adopt, do not reimplement |
| Deduplication | `datasketch` MinHash LSH | CPU; runs locally |
| Language ID | fastText `lid` | CPU |
| Quality filtering | KenLM perplexity; Gopher rules adapted for Bengali | CPU |
| Training framework | PyTorch + DeepSpeed ZeRO-3 or FSDP + FlashAttention-2 | bf16 |
| Fine-tuning | PEFT (LoRA / QLoRA); TRL for DPO | Standard |
| Inference / on-device | llama.cpp; GGUF Q4 | Local testing on the Ryzen 3; final Android deployment |
| Compute | RunPod, Vast.ai, Lambda; E2E Networks or Yotta for IndiaAI subsidy | **Always spot. Always checkpoint.** |
| Experiment tracking | Weights & Biases or MLflow | Non-optional for a multi-year programme |
| Data format | Parquet or WebDataset shards | Never loose files |

---

## Appendix D: Organisational Landscape

### D.1 India

| Organisation | Type | Bengali-relevant output | Posture |
|---|---|---|---|
| **AI4Bharat, IIT Madras** | Academic | Sangraha, Setu, IndicConformer, IndicF5, IndicGenBench, IndicBERT, IndicTrans | Open tooling, permissive licences, history of external collaboration. **Approach first.** |
| **Sarvam AI, Bengaluru** | Commercial | Sarvam-1 (2B); Sarvam-30B and 105B (open weights, Mar 2026); Sarvam-M (24B on Mistral) | Open-weight strategy. Bengali at 8% of Indic mix. Potential adopter of the dialect corpus. |
| **BharatGen** | Government-backed | Param-1 (2.9B, from scratch, 25% Indic) | Public mission. Grant-adjacent. |
| **Bhashini** | Government DPI | APIs and models for Indian languages | Distribution channel, not a research partner |
| **IndiaAI Mission** | Government programme | GPU compute subsidy; sovereign AI policy | Funding and compute source |
| **Krutrim (Ola)** | Commercial | Pan-Indic models | Bengali shallow. Not a priority contact. |
| **Jadavpur University; ISI Kolkata** | Academic | Bengali linguistics; Bichitra project | **The complementary half.** No ML pipeline. Strong fit. |
| **IIIT Hyderabad; CDAC** | Academic / govt | Indic speech and NLP research | Secondary contacts |

### D.2 Bangladesh

| Organisation | Type | Bengali-relevant output | Posture |
|---|---|---|---|
| **BUET CSE** | Academic | BanglaBERT, Bangla2B+, BEnQA, DL Sprint competitions | The academic centre of gravity for Bengali NLP |
| **Hishab** (Dhaka / Singapore) | Commercial | TituLLM 1b and 3b + Bengali benchmarking datasets. ACL 2025 Findings. | **The most serious commercial Bengali LLM effort.** They publish and release. **Complementary partner, not competitor.** |
| **Giga Tech Limited** | Commercial | Bangla NLP programme | Secondary |
| **Daffodil International University** | Academic | BanglaDial | Dialect data authors |
| **BRAC University; Jahangirnagar University** | Academic | Dialect and zero-resource language work; Multilingual Cloud (multiling.cloud) | Fieldwork methodology precedent |
| **Bangladesh Computer Council (EBLICT)** | Government | Multilingual Cloud Corpus: 85,792 entries, ~107 h, 42 language varieties | National-scale digitisation precedent |

---

## Appendix E: Notation and Glossary

### E.1 Notation

| Symbol | Meaning |
|---|---|
| `Σ` | Set of Bengali Unicode codepoints |
| `𝒢(·)` | Grapheme cluster segmentation function |
| `T(·)` | Tokenizer applied to a string |
| `Lev(·,·)` | Levenshtein edit distance |
| `N` | Non-embedding parameter count |
| `D` | Training tokens |
| `C ≈ 6ND` | Training FLOPs |
| `MFU` | Model FLOPs utilisation |
| `E ∈ ℝ^(V×d)` | Embedding matrix |
| `U ∈ ℝ^(d×V)` | Unembedding matrix |
| `τ` | Hallucination admission threshold |

### E.2 Glossary

| Term | Definition |
|---|---|
| **Fertility** | Average subword tokens per whitespace word. Lower is better. Controls training cost, inference cost, effective context length. |
| **STRR** | Single Token Retention Rate. Proportion of words preserved as one token. Exposes vocabulary allocation where fertility cannot. |
| **Grapheme cluster** | The smallest user-perceived unit of written text. In Bengali: a base consonant plus its conjuncts, matras, reph, and phalas. **Not the same as a codepoint.** |
| **GCER** | Grapheme-Cluster Error Rate. Edit distance over grapheme clusters. **The correct headline OCR metric for Bengali.** |
| **Conjunct** (juktakkhor, যুক্তাক্ষর) | A ligature glyph formed from a consonant cluster. 300–500 in common use. Visual form often unrelated to constituents. |
| **Matra** (মাত্রা) | Two senses: (1) the horizontal headline joining letters within a word; (2) a vowel sign (kar) attached to a consonant. |
| **Reph** (র্) | Glyph for a preceding *ra* rendering as a mark above the following consonant, reordered from logical position. |
| **Phala** (ফলা) | Subjoined consonant form hanging below the base. Ya-phala (্য), ra-phala (্র). |
| **Virama / hasanta** (্, U+09CD) | Vowel-killer. Triggers conjunct formation. |
| **ZWJ / ZWNJ** | Zero-width joiner (U+200D) / non-joiner (U+200C). Force or suppress ligature formation. Used inconsistently in the wild. |
| **Tatsama / tadbhava** | Vocabulary strata. Tatsama borrowed unchanged from Sanskrit, conservative orthography. Tadbhava evolved through Prakrit into vernacular forms. |
| **Sadhu / chalit** | The two literary registers of Bengali. Sadhu is archaic and Sanskritised; chalit is the colloquial standard. Both derive from Rarhi. |
| **Ghoti / Bangal** | Informal sociolinguistic labels for West Bengal-origin and East Bengal-origin speakers. **Not a formal dialect classification.** Use Sen's scheme in publications. |
| **Continued pretraining** | Further self-supervised pretraining of an existing base model on a new corpus. Changes internal representation. **Distinct from fine-tuning.** |
| **Vocabulary surgery** | Replacing a model's embedding and unembedding matrices to use a new tokenizer, initialising new embeddings from old ones before continued pretraining. |
| **MFU** | Model FLOPs Utilisation. Fraction of theoretical peak FLOPs achieved. **Below 0.30 indicates a dataloader bottleneck.** |
| **DPO** | Direct Preference Optimization. Preference alignment from ranked output pairs without a separate reward model. Practical at small scale; full RLHF is not. |
| **CTC** | Connectionist Temporal Classification. Loss for sequence recognition without frame-level alignment. Standard for OCR line recognisers and streaming ASR. |
| **NFC** | Unicode Normalization Form C, canonical composition. **Mandatory before any Bengali string comparison.** |
| **Catastrophic forgetting** | Loss of previously learned capability during continued training on a narrow distribution. Mitigated by replay. |

---

## Naming Register

| Layer | Name | Status |
|---|---|---|
| Programme | **Project Bornomala** (বর্ণমালা) | **FIXED** |
| Track A output (tokenizer) | `[TBD]` | Deferred |
| Track B output (OCR models) | `[TBD]` | Deferred |
| Track B output (OCR benchmark) | `[TBD]` | Deferred |
| Track C output (dialect text corpus) | `[TBD]` | Deferred |
| Track C output (dialect speech corpus) | `[TBD]` | Deferred |
| Track C output (dialect benchmark) | `[TBD]` | Deferred |
| Track D output (ASR, TTS) | `[TBD]` | Deferred |
| Track E output (foundation model) | `[TBD]` | Deferred |
| Track E output (cultural benchmark) | `[TBD]` | Deferred |
| Organisation / repository handle | `[TBD]` | Deferred |

Romanisation of the programme name is fixed as **Bornomala**, never *Barnamala* or *Bornomala*'s variants. Bengali orthography fixed as **বর্ণমালা**. Do not let either drift across repositories, papers, or slides.

---

*End of document. Project Bornomala, Technical and Scientific Specification, Draft 1.0. Konko Maji, 10 July 2026.*
