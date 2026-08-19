# Bengali cross-tokenizer comparison

To our knowledge, the first fully reproducible benchmark comparing modern Bengali tokenizers across compression, word preservation, and conjunct fragmentation using a common evaluation pipeline. Every number is produced from the real tokenizer of each system on identical text. Nothing is estimated or fabricated. Both of this project's own tokenizers - **v1 (`bn-bpe-64k`, BPE over grapheme clusters)** and **BMBT (`bmbt-64k`, BPE over aksharas, plus a featural decomposition v1 lacks)** - are shown as separate rows everywhere in this document, not merged into one "ours" row: they are two real, independently trained artifacts, and where they diverge (FLORES+, see below) that divergence is reported, not hidden by averaging.

## Exactly how this was measured (full transparency)

- **Our tokenizers**: both BPE, 64,000 vocabulary, trained on the identical literary-weighted corpus of 1.5M lines (Wikisource public-domain text, Sangraha verified/ben - pdf-typed as a formal/literary-register proxy, OCR-noise-filtered, and web-typed for general register - the first 15,000 Bengali Wikipedia articles, and XL-Sum Bengali news; India-origin IndicCorp v2 added 2026-08). v1's atom is the grapheme cluster; BMBT's is the akshara (`bntok/akshara.py`'s finite-state parser), plus a morphology-aware variant (`bmbt-64k-morph`, benchmarked on this comparison's six registers below - see `docs/bmbt-morphology.md`). Full corpus composition, what was substituted and why: `docs/known-issues.md` point 6. An ablation across 32k/48k/64k vocab on this same corpus showed fertility recovering monotonically with vocab size (point 7); 64k is the smallest size that beats every external baseline tested.
- **Evaluation text**: six disjoint held-out sets, none touched during training.
  - Wikipedia: 828 lines from articles after the first 15,000.
  - Literary/formal, general web, news: reserved tails of Sangraha pdf-typed, Sangraha web-typed, and XL-Sum documents respectively, starting exactly where training's document budget ends (`bntok.corpus.build_register_held_out`).
  - Banglish: romanized-Bengali (Latin-script) text from CC-100's `bn_rom` config, filtered for genuine chat-style Banglish (`bntok.corpus.build_banglish_held_out`). Never used in any training config, so no disjointness bookkeeping needed. **This register measures our tokenizers directly on Latin-script text, deliberately** - see the Banglish section below for why that number is bad on purpose and what fixes it.
  - FLORES+: `openlanguagedata/flores_plus` (the maintained FLORES-200 successor), `ben_Beng` config, dev+devtest splits, 2,009 professionally translated sentences. Added specifically to measure on the exact corpus an external paper ("The Tokenizer Tax", Srivastava 2026) built its own published Bengali fertility numbers from, instead of relying on a cross-walk (`_personal/SESSION_LOG.md` session 6). Also never used in training.
- **Other tokenizers**: loaded from their official public releases and run with their own real tokenizers. Sarvam-1 (sarvamai/sarvam-1), SUTRA (TWO/sutra-mlt256-v2), BrahmicTokenizer-131K (theschoolofai/BrahmicTokenizer-131K), Krutrim (krutrim-ai-labs/Krutrim-2-instruct), Param2-17B (bharatgenai/Param2-17B-A2.4B-Thinking), IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only), BanglaBERT (csebuetnlp/banglabert), BanglaT5 (csebuetnlp/banglat5), mBERT (google-bert/bert-base-multilingual-cased), XLM-RoBERTa (FacebookAI/xlm-roberta-base), DeepSeek-V3, Llama-3.1, Gemma-2, Mistral-7B, Qwen2.5, plus GPT-4o/GPT-4 via tiktoken. Gemma-2 was gated behind Google's consent form until 2026-08-18, when access was granted to this project's own account - the row is real, not estimated, but reproducing this comparison from a different account without that grant will correctly show it as unavailable rather than fail. Two baselines named in the v2 design doc's roadmap still could not be added, honestly rather than faked: IndicSuperTokenizer (arXiv:2511.03237) has no public code/tokenizer release found; the only similarly-named BengaliBPE (arXiv:2511.05324) Hugging Face repo found fails to load and is not verifiably the paper's own artifact.
- **Metrics**: all text NFC-normalised first. Fertility = tokens / whitespace-words (lower is better). STRR = fraction of words kept as a single token. Bytes/token = UTF-8 bytes / tokens. Two fragmentation measures, reported together deliberately, not one instead of the other:
  - **Legacy fragmentation** (kept for continuity with every earlier-published number): fraction of grapheme clusters a token boundary splits, binary (any split counts the same), denominator includes clusters that could never be split.
  - **Destructive rate** (the corrected headline, `bntok/fragmentation.py`): denominator restricted to clusters that COULD be split, and only counts splits that actually sever something real (a stranded virama, a detached nukta) - not a consonant-cluster/vowel-sign seam, which is a harmless, literacy-recognised split, not damage. Destructive rate is always <= legacy fragmentation, structurally, and the tables below confirm that holds on every row.

  GPT-4o/GPT-4 give no character offsets, so neither fragmentation measure is computed for them (`n/a`, not zero). A small number of held-out lines that quote foreign-script text (Greek, Arabic, Japanese, genuinely present in Wikipedia and news text) fall outside this tokenizer's guaranteed coverage (Bengali block + ASCII, see `docs/known-issues.md` point 4) and are excluded from both fragmentation measures specifically, since that is a documented, separate scope boundary, not a conjunct-splitting question; they still count normally toward fertility/STRR/bytes.
- **Reproduce**:
  ```
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --skip 15000
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register literary_formal
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register general_web
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register news
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register banglish
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register flores
  ```
  Raw JSON for each run: `comparison-wikipedia.json`, `comparison-literary_formal.json`, `comparison-general_web.json`, `comparison-news.json`, `comparison-banglish.json`, `comparison-flores.json` (this directory).

## Results: Wikipedia held-out (828 lines, sorted by fertility, best first)

| Tokenizer | Fertility | STRR | Bytes/token | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| **Bornomala v1** | **1.524** | **0.722** | 11.38 | 0.0001 | 0.0004 |
| **Bornomala BMBT** | **1.524** | **0.722** | 11.38 | 0.0001 | 0.0004 |
| BanglaBERT (csebuetnlp) | 1.625 | 0.649 | 10.67 | 0.0300 | 0.0162 |
| IndicBERTv2 (AI4Bharat) | 1.652 | 0.612 | 10.50 | 0.0440 | 0.0191 |
| BanglaT5 (csebuetnlp) | 1.669 | 0.628 | 10.39 | 0.0221 | 0.0088 |
| SUTRA (TWO AI) | 2.218 | 0.419 | 7.82 | 0.1579 | 0.0499 |
| XLM-RoBERTa (Meta) | 2.464 | 0.363 | 7.04 | 0.1019 | 0.0627 |
| Param2-17B (BharatGen) | 2.517 | 0.131 | 6.89 | 0.1773 | 0.0420 |
| Sarvam-1 (Sarvam AI) | 2.593 | 0.415 | 6.69 | 0.1191 | 0.0364 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.111 | 6.65 | n/a | n/a |
| BrahmicTokenizer-131K (TSAI) | 2.620 | 0.154 | 6.62 | 0.2209 | 0.0820 |
| mBERT (Google) | 2.777 | 0.385 | 6.25 | 0.1800 | 0.1552 |
| DeepSeek-V3 | 2.994 | 0.089 | 5.79 | 0.2845 | 0.1031 |
| Krutrim (Krutrim AI) | 3.207 | 0.076 | 5.41 | 0.2859 | 0.0990 |
| Gemma-2 (Google) | 3.841 | 0.080 | 4.52 | 0.3289 | 0.1925 |
| Qwen2.5 (Alibaba) | 6.951 | 0.059 | 2.50 | 0.4570 | 0.4988 |
| Mistral-7B (Mistral AI) | 7.493 | 0.043 | 2.32 | 0.4570 | 0.4988 |
| Llama-3.1 (Meta) | 7.724 | 0.043 | 2.25 | 0.4570 | 0.4988 |
| GPT-4 (OpenAI cl100k) | 7.794 | 0.042 | 2.23 | n/a | n/a |

v2 akshara parser (pre-vocabulary, no merges, not a like-for-like row - see the step-4 section further down): fertility 4.527, STRR 0.045, destructive rate 0.0000.

## Results: Literary/formal held-out (19,053 lines)

| Tokenizer | Fertility | STRR | Bytes/token | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| **Bornomala v1** | **1.319** | **0.789** | 12.23 | 0.0001 | 0.0003 |
| **Bornomala BMBT** | **1.319** | **0.789** | 12.23 | 0.0001 | 0.0003 |
| BanglaBERT (csebuetnlp) | 1.488 | 0.680 | 10.83 | 0.0347 | 0.0169 |
| BanglaT5 (csebuetnlp) | 1.542 | 0.659 | 10.45 | 0.0296 | 0.0101 |
| IndicBERTv2 (AI4Bharat) | 1.615 | 0.605 | 9.98 | 0.0568 | 0.0211 |
| SUTRA (TWO AI) | 2.194 | 0.401 | 7.35 | 0.1811 | 0.0472 |
| XLM-RoBERTa (Meta) | 2.398 | 0.363 | 6.72 | 0.1144 | 0.0660 |
| Param2-17B (BharatGen) | 2.440 | 0.133 | 6.61 | 0.2164 | 0.0375 |
| BrahmicTokenizer-131K (TSAI) | 2.448 | 0.165 | 6.59 | 0.2388 | 0.0667 |
| GPT-4o (OpenAI o200k) | 2.455 | 0.119 | 6.57 | n/a | n/a |
| Sarvam-1 (Sarvam AI) | 2.617 | 0.378 | 6.16 | 0.1457 | 0.0471 |
| DeepSeek-V3 | 2.818 | 0.082 | 5.72 | 0.3117 | 0.0821 |
| mBERT (Google) | 2.846 | 0.312 | 5.67 | 0.2203 | 0.1492 |
| Krutrim (Krutrim AI) | 3.185 | 0.074 | 5.06 | 0.3426 | 0.0890 |
| Gemma-2 (Google) | 3.613 | 0.065 | 4.46 | 0.3621 | 0.1594 |
| Qwen2.5 (Alibaba) | 6.621 | 0.050 | 2.43 | 0.5164 | 0.4127 |
| Mistral-7B (Mistral AI) | 7.114 | 0.024 | 2.27 | 0.5164 | 0.4127 |
| Llama-3.1 (Meta) | 7.350 | 0.036 | 2.19 | 0.5164 | 0.4127 |
| GPT-4 (OpenAI cl100k) | 7.445 | 0.025 | 2.17 | n/a | n/a |

## Results: General web held-out (6,405 lines)

| Tokenizer | Fertility | STRR | Bytes/token | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| **Bornomala v1** | **1.195** | **0.863** | 14.17 | 0.0001 | 0.0002 |
| **Bornomala BMBT** | **1.195** | **0.863** | 14.17 | 0.0001 | 0.0002 |
| BanglaBERT (csebuetnlp) | 1.324 | 0.764 | 12.78 | 0.0152 | 0.0045 |
| BanglaT5 (csebuetnlp) | 1.346 | 0.753 | 12.57 | 0.0134 | 0.0030 |
| IndicBERTv2 (AI4Bharat) | 1.389 | 0.718 | 12.18 | 0.0273 | 0.0070 |
| XLM-RoBERTa (Meta) | 2.055 | 0.460 | 8.24 | 0.0810 | 0.0357 |
| SUTRA (TWO AI) | 2.060 | 0.435 | 8.22 | 0.1615 | 0.0326 |
| BrahmicTokenizer-131K (TSAI) | 2.259 | 0.136 | 7.49 | 0.2001 | 0.0446 |
| GPT-4o (OpenAI o200k) | 2.260 | 0.089 | 7.49 | n/a | n/a |
| Param2-17B (BharatGen) | 2.348 | 0.107 | 7.21 | 0.1900 | 0.0258 |
| Sarvam-1 (Sarvam AI) | 2.436 | 0.427 | 6.95 | 0.1209 | 0.0253 |
| mBERT (Google) | 2.712 | 0.348 | 6.24 | 0.1995 | 0.1171 |
| DeepSeek-V3 | 2.819 | 0.058 | 6.00 | 0.3067 | 0.0693 |
| Krutrim (Krutrim AI) | 3.129 | 0.047 | 5.41 | 0.3231 | 0.0675 |
| Gemma-2 (Google) | 3.684 | 0.040 | 4.59 | 0.3580 | 0.1440 |
| Qwen2.5 (Alibaba) | 6.906 | 0.025 | 2.45 | 0.5241 | 0.3824 |
| Mistral-7B (Mistral AI) | 7.391 | 0.010 | 2.29 | 0.5241 | 0.3824 |
| Llama-3.1 (Meta) | 7.647 | 0.011 | 2.21 | 0.5241 | 0.3823 |
| GPT-4 (OpenAI cl100k) | 7.724 | 0.010 | 2.19 | n/a | n/a |

## Results: News held-out (22,683 lines)

| Tokenizer | Fertility | STRR | Bytes/token | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| **Bornomala v1** | **1.142** | **0.893** | 14.91 | 0.0000 | 0.0001 |
| **Bornomala BMBT** | **1.142** | **0.893** | 14.91 | 0.0000 | 0.0001 |
| BanglaBERT (csebuetnlp) | 1.259 | 0.796 | 13.52 | 0.0110 | 0.0031 |
| BanglaT5 (csebuetnlp) | 1.277 | 0.783 | 13.33 | 0.0102 | 0.0019 |
| IndicBERTv2 (AI4Bharat) | 1.313 | 0.755 | 12.96 | 0.0208 | 0.0047 |
| XLM-RoBERTa (Meta) | 1.954 | 0.494 | 8.71 | 0.0733 | 0.0295 |
| SUTRA (TWO AI) | 1.958 | 0.467 | 8.69 | 0.1515 | 0.0286 |
| BrahmicTokenizer-131K (TSAI) | 2.187 | 0.134 | 7.78 | 0.1953 | 0.0395 |
| GPT-4o (OpenAI o200k) | 2.195 | 0.077 | 7.75 | n/a | n/a |
| Param2-17B (BharatGen) | 2.257 | 0.101 | 7.54 | 0.1893 | 0.0226 |
| Sarvam-1 (Sarvam AI) | 2.333 | 0.454 | 7.30 | 0.1138 | 0.0197 |
| mBERT (Google) | 2.639 | 0.362 | 6.45 | 0.1925 | 0.1123 |
| DeepSeek-V3 | 2.750 | 0.047 | 6.19 | 0.3066 | 0.0649 |
| Krutrim (Krutrim AI) | 3.063 | 0.035 | 5.56 | 0.3236 | 0.0649 |
| Gemma-2 (Google) | 3.651 | 0.026 | 4.66 | 0.3627 | 0.1371 |
| Qwen2.5 (Alibaba) | 6.897 | 0.013 | 2.47 | 0.5458 | 0.3672 |
| Mistral-7B (Mistral AI) | 7.402 | 0.005 | 2.30 | 0.5458 | 0.3672 |
| Llama-3.1 (Meta) | 7.673 | 0.005 | 2.22 | 0.5458 | 0.3671 |
| GPT-4 (OpenAI cl100k) | 7.737 | 0.005 | 2.20 | n/a | n/a |

## Results: FLORES+ held-out (2,009 lines - same corpus an external paper's own numbers come from)

This is the one register where v1 and BMBT are not bit-identical - reported exactly as measured, neither direction hidden:

| Tokenizer | Fertility | STRR | Bytes/token | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| **Bornomala BMBT** | **1.240** | **0.838** | 14.04 | 0.0000 | 0.0001 |
| **Bornomala v1** | **1.241** | **0.838** | 14.04 | 0.0000 | 0.0001 |
| BanglaBERT (csebuetnlp) | 1.341 | 0.755 | 12.99 | 0.0185 | 0.0066 |
| BanglaT5 (csebuetnlp) | 1.364 | 0.739 | 12.77 | 0.0155 | 0.0044 |
| IndicBERTv2 (AI4Bharat) | 1.393 | 0.715 | 12.51 | 0.0301 | 0.0091 |
| SUTRA (TWO AI) | 1.753 | 0.570 | 9.94 | 0.1152 | 0.0278 |
| XLM-RoBERTa (Meta) | 2.146 | 0.448 | 8.12 | 0.0938 | 0.0495 |
| Param2-17B (BharatGen) | 2.175 | 0.105 | 8.01 | 0.1657 | 0.0281 |
| Sarvam-1 (Sarvam AI) | 2.295 | 0.498 | 7.59 | 0.1028 | 0.0238 |
| BrahmicTokenizer-131K (TSAI) | 2.306 | 0.145 | 7.55 | 0.2097 | 0.0585 |
| GPT-4o (OpenAI o200k) | 2.309 | 0.066 | 7.54 | n/a | n/a |
| mBERT (Google) | 2.630 | 0.397 | 6.62 | 0.1880 | 0.1320 |
| DeepSeek-V3 | 2.779 | 0.044 | 6.27 | 0.2965 | 0.0855 |
| Krutrim (Krutrim AI) | 3.056 | 0.029 | 5.70 | 0.3027 | 0.0798 |
| Gemma-2 (Google) | 3.732 | 0.023 | 4.67 | 0.3606 | 0.1670 |
| Qwen2.5 (Alibaba) | 7.095 | 0.009 | 2.46 | 0.5406 | 0.4414 |
| Mistral-7B (Mistral AI) | 7.580 | 0.003 | 2.30 | 0.5406 | 0.4414 |
| Llama-3.1 (Meta) | 7.881 | 0.008 | 2.21 | 0.5406 | 0.4413 |
| GPT-4 (OpenAI cl100k) | 7.941 | 0.008 | 2.19 | n/a | n/a |

**Why BMBT edges v1 here and nowhere else, stated honestly, not overclaimed**: 0.001 fertility on a 2,009-sentence corpus is 1-2 tokens' difference in total, not a structural finding - it is reported because it is real and measured, not suppressed because it is small. On every other register the two are exactly tied. **This is also the first genuine, same-corpus confirmation of the session-6 cross-walk**: an external paper's own published Bengali "tokenizer tax" numbers, built from this exact FLORES-200/FLORES+ corpus, are reproduced almost exactly here for GPT-4/cl100k (paper's cross-walked ~7.8 implied fertility vs 7.941 measured directly) and GPT-4o/o200k (~2.6 implied vs 2.309-2.608 measured across our registers) - two independent measurements of the same tokenizers, on the same corpus this time, agreeing.

## Results: Banglish held-out (800 lines, romanized-Bengali chat text) - a deliberately bad number

Full ranking, not truncated - ours really is last:

| Tokenizer | Fertility | STRR | Bytes/token |
|---|--:|--:|--:|
| SUTRA (TWO AI) | 1.894 | 0.416 | 3.14 |
| XLM-RoBERTa (Meta) | 1.910 | 0.446 | 3.12 |
| GPT-4o (OpenAI o200k) | 1.952 | 0.356 | 3.05 |
| IndicBERTv2 (AI4Bharat) | 1.975 | 0.381 | 3.01 |
| Gemma-2 (Google) | 1.976 | 0.426 | 3.01 |
| BrahmicTokenizer-131K (TSAI) | 1.982 | 0.347 | 3.00 |
| DeepSeek-V3 | 2.104 | 0.318 | 2.83 |
| Krutrim (Krutrim AI) | 2.114 | 0.316 | 2.81 |
| Llama-3.1 (Meta) | 2.114 | 0.324 | 2.81 |
| GPT-4 (OpenAI cl100k) | 2.128 | 0.321 | 2.80 |
| mBERT (Google) | 2.158 | 0.354 | 2.76 |
| Qwen2.5 (Alibaba) | 2.175 | 0.318 | 2.74 |
| BanglaT5 (csebuetnlp) | 2.333 | 0.315 | 2.55 |
| Mistral-7B (Mistral AI) | 2.392 | 0.261 | 2.49 |
| Param2-17B (BharatGen) | 2.460 | 0.288 | 2.42 |
| BanglaBERT (csebuetnlp) | 2.473 | 0.306 | 2.41 |
| Sarvam-1 (Sarvam AI) | 2.679 | 0.179 | 2.22 |
| **Bornomala BMBT** | **2.905** | **0.110** | 2.05 |
| **Bornomala v1** | **2.906** | **0.110** | 2.05 |

**Ours is LAST here, on purpose to show why, not despite the rest of this document.** Both tokenizers are built entirely around Bengali-script structure (grapheme clusters / aksharas) - fed raw Latin-script text, that structure buys nothing, and generic multilingual BPE tokenizers that spread their vocabulary across scripts do better on Latin text almost by construction. This is the real gap that motivated building the Banglish transliteration pipeline (tiers 0-3: lookup table, character n-gram classifier, and a trained seq2seq model, `bntok/banglish.py` and `bntok/banglish_tier3.py`): **transliterate Latin-script Banglish to real Bengali script FIRST, then hand off to the tokenizer that already wins by a wide margin there.** End-to-end, with the pipeline in front of the tokenizer (a different measurement from the raw register above, not directly comparable), fertility drops from 2.827 (pre-pipeline baseline on a related held-out set) to 1.740 - beating every tokenizer on this table, including this register's own current leader. Full writeup: `docs/known-issues.md`'s Banglish section, including the tier-3 model's real trained accuracy (53.9% exact-match, 13.5% CER, beam search).

## What the four core registers show

- Our tokenizers need the fewest tokens per Bengali word, keep the most whole words, and split the fewest destructive conjuncts, **on every one of the four core registers and FLORES+**, not just the Wikipedia set most comparisons stop at.
- **The closest rival is now BanglaBERT (csebuetnlp), not IndicBERTv2** - a real change from earlier versions of this table, which only measured IndicBERTv2, SUTRA, Krutrim and a handful of others. Adding BanglaBERT and BanglaT5 (both Bengali-monolingual, added 2026-08-17) surfaced a harder bar: BanglaBERT beats IndicBERTv2 on every core register (e.g. Wikipedia 1.625 vs 1.652), though both remain well behind ours.
- Destructive rate stays near-zero across the board for ours (0.0001-0.0004), while even the closest rivals sit an order of magnitude higher (BanglaBERT: 0.0031-0.0169 depending on register).

## A real measurement bug caught mid-analysis (kept here, not quietly fixed away)

An early version of this comparison's own fragmentation counter for our tokenizer had two bugs, found via a hard assertion added specifically to catch this class of mistake: (1) an off-by-one from the Metaspace word-boundary marker's leading space, which manufactured thousands of false fragmentation hits; (2) `encode_tokens()`'s debug view returning the literal string `<unk>` in place of missing text for out-of-coverage foreign-script codepoints, rather than an empty placeholder, which could silently desynchronise offsets on such lines. Both are now fixed in `scripts/compare.py`, and the fix asserts the reconstruction is exact on every line it measures, so a similar bug cannot silently corrupt a comparison again. Full account: `docs/known-issues.md` point 8.

A second, later fix: the legacy fragmentation counter was always binary and its denominator always included unsplittable clusters - both defects documented in `bntok/fragmentation.py`. The graded `destructive_rate`/`any_split_rate` measures fix both, and are reported alongside the legacy field, not instead of it, so every earlier-published number stays comparable.

## v2 roadmap step 4: the akshara finite-state parser, measured (not a like-for-like row)

The v2 design's own roadmap (`docs/design/reading-bengali-on-its-own-terms.md`) calls for benchmarking the akshara parser (`bntok/akshara.py`) before building anything on top of it. Measured via `scripts/compare.py`'s `measure_akshara()`, on the exact same held-out sets as the tokenizer results above:

| Register | Fertility | STRR | Bytes/token | Destructive rate |
|---|--:|--:|--:|--:|
| Wikipedia | 4.527 | 0.045 | 3.83 | **0.0000** |
| Literary / formal | 4.043 | 0.079 | 3.99 | **0.0000** |
| General web | 4.185 | 0.041 | 4.04 | **0.0000** |
| News | 4.113 | 0.035 | 4.14 | **0.0000** |
| Banglish | 5.933 | 0.049 | 1.00 | **0.0000** |
| FLORES+ | 4.177 | 0.035 | 4.17 | **0.0000** |

**This is not a like-for-like comparison and is reported separately on purpose.** The akshara parser has no vocabulary and no merges (v2 roadmap step 5's statistical layer is what BMBT is): its "fertility" is the number of un-merged akshara/other chunks per word, the pre-compression granularity, not a trained tokenizer's post-BPE-merge token count. Needing roughly 3-4x more units per word than the trained BPE tokenizers is expected, not a regression - exactly the number the roadmap's own step 4 asked to be measured and reported honestly before proceeding to step 5.

**Destructive rate is the one column above that IS a fair, like-for-like number**, since it does not depend on compression: **exactly 0.0000 on every register measured**, by grammar construction, not by an atom-frequency threshold the way a trained BPE model's near-zero-but-not-quite number can carry.

**Three real bugs were found and fixed by running this measurement against real Wikipedia text first, not synthetic tests**, all now covered by `tests/test_akshara.py` and documented in `docs/known-issues.md` points 11-13: an independent vowel followed by a virama does not chain into a further consonant the way a consonant does; a Modifier (not a Matra or Nukta) blocks conjunct-chain continuation; and ZWJ/ZWNJ are not tied to a fixed position relative to the virama the way an earlier pass assumed.

**Step 4 is complete**: measured against both tokenizers' own held-out sets across all six registers, and against every real external baseline with a usable public release. IndicSuperTokenizer and BengaliBPE remain unavailable, reported honestly rather than faked.

## v2 roadmap step 5: BMBT, measured (a genuine like-for-like row)

BMBT (Bornomala's Bengali Tokenizer, `bntok/bmbt.py`) is grammar (the akshara parser above) plus a featural decomposition (`featurize()`) plus a statistical BPE layer over akshara atoms - the same architecture as v1, with the atomic unit swapped from grapheme cluster to akshara. Morphology is now built as well, opt-in, and aligns token boundaries to Bengali's suffix structure: `bmbt-64k-morph` is trained and its own results cross-validate an advance prediction almost exactly (`docs/bmbt-morphology.md`), now benchmarked on this table's own six registers below. Full design: `docs/bmbt-architecture.md`.

Unlike the raw akshara-parser row above, a trained BMBT has a real vocabulary and real merges the same way `bn-bpe-64k` does, so it is a genuine like-for-like fertility comparison, trained on the identical corpus for a controlled comparison. **Reported exactly as measured, not the outcome anyone assumed going in**: tied on five of six registers down to the fourth decimal, and on the sixth (FLORES+) BMBT edges v1 by 0.001 fertility - real, small, not overclaimed (see the FLORES+ section above).

**Why a tie on almost every register, not a loss**, given `docs/design/FORMAL_SPEC.md`'s own proof that a constrained BPE cannot beat an unconstrained one on raw token count: akshara-grammar boundaries are already nearly identical to `\X`'s grapheme-cluster boundaries on well-formed Bengali (the akshara-parser measurement above), so constraining BPE to akshara boundaries instead of grapheme-cluster boundaries barely constrains anything further in practice - the two atom schemes are close to isomorphic on real text.

**What BMBT adds, independent of the fertility tie**: a provable, Unicode-library-independent grammar instead of delegated trust in `regex`'s own `\X`, `featurize()` - a real, tested structural decomposition (onset/vowel/modifier per akshara) v1 never had, at zero fertility cost - and now morphology, which trades a measured, deliberate fertility cost (below) for token boundaries that fall where Bengali's morphemes fall.

### Morphology (`bmbt-64k-morph`), all six registers

The cost stated qualitatively above, quantified. `bmbt-64k-morph` measured on the same six held-out registers as v1/BMBT (full detail and the honest per-register discussion: `docs/bmbt-morphology.md`):

| Register | v1 (`bn-bpe-64k`) fertility | `bmbt-64k-morph` fertility | Cost | Legacy frag. | Destructive rate |
|---|--:|--:|--:|--:|--:|
| Wikipedia | 1.524 | 1.855 | +21.7% | 3.81% | 0.03% |
| Literary / formal | 1.319 | 1.697 | +28.7% | 4.39% | 0.03% |
| General web | 1.195 | 1.644 | +37.6% | 4.93% | 0.02% |
| News | 1.142 | 1.623 | +42.1% | 5.19% | 0.01% |
| FLORES+ | 1.241 | 1.691 | +36.3% | 4.74% | 0.01% |
| Banglish | 2.906 | 2.780 | -4.3% | 0.00% | 0.00% |

The cost tracks how much real Bengali morphology each register carries: news and general web (the densest everyday registers) pay the most, Wikipedia less. Banglish is the one register where `bmbt-64k-morph` reports *lower* fertility than v1, but 0.00% legacy fragmentation there shows the morphology layer found nothing to split, romanized text carries almost no native morpheme seams. That gap is ordinary vocabulary noise between two independently trained tables on largely non-Bengali content, not a morphology effect, stated plainly rather than smoothed into "worse everywhere." Destructive rate (real conjuncts severed) stays close to zero on every register, never the literal zero the rule layer alone guarantees, at most 0.03% of splittable clusters.

Reproduce:
```
python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --morphology --out artifacts/bmbt-64k-morph
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --skip 15000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --register literary_formal
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --register general_web
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --register news
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --register banglish
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k-morph --register flores
```

### CC-100 ablation

Trained both architectures again with CC-100 Bengali added to the corpus (`configs/bpe-64k-cc100.json`, same weights plus `cc100_general_web`; see `docs/known-issues.md` point 15 for a real bug found and fixed in `stream_cc100` while running this):

| Register | Fertility, no CC-100 | Fertility, +CC-100 | Change |
|---|--:|--:|--:|
| Wikipedia | 1.524 | 1.531 | +0.007 (slightly worse) |
| General web | 1.201 | 1.199 | -0.002 (slightly better) |

Both directions make sense: the same 64,000-token vocabulary budget now spans five sources instead of four, diluting Wikipedia-specific coverage slightly, while general web (the register CC-100 actually targets) gets a small real benefit. Both effects round to the third decimal place - a wash, not a case for or against adopting CC-100 in the default weights. `bn-bpe-64k` and `bmbt-64k` (without CC-100) remain the recommended artifacts; `bn-bpe-64k-cc100`/`bmbt-64k-cc100` are kept as the ablation record, not shipped as a recommendation. v1 and BMBT tie exactly on this ablation too (identical fertility/STRR/bytes/fragmentation on both registers tested). This ablation predates the destructive-rate metric and the FLORES+/Banglish registers, and has not been rerun with either.

Reproduce:
```
python -m bntok bmbt-train --corpus-config configs/bpe-64k-cc100.json --out artifacts/bmbt-64k-cc100
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k-cc100 --bmbt-tokenizer artifacts/bmbt-64k-cc100 --skip 15000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k-cc100 --bmbt-tokenizer artifacts/bmbt-64k-cc100 --register general_web
```

## Hard words: conjuncts and Bengali place names

The registers above are a corpus average; it can hide how a tokenizer treats specific, culturally load-bearing words. A fixed list of 13 - deity names, a national poet, well-known West Bengal places, all conjunct-dense - measured on every tokenizer this project tracks (`scripts/hard_words.py`, imported model list shared with `scripts/compare.py` so it can never drift out of sync). Full per-word table: `benchmarks/hard-words.md`.

**Three tokenizers tie at exactly 1.00 average tokens/word on this list: ours (v1 and BMBT) and, as of the 2026-08-17 BanglaBERT/BanglaT5 addition, both of those too.** This corrects an earlier, now-outdated claim on this page that ours was the *only* tokenizer to hold conjunct integrity on every one of these 13 words - it no longer is, and the earlier claim is retracted here rather than left standing. What is still true, and is the actual differentiator: **ours is the only one of the three that guarantees this by construction** (grammar-first parsing, provably cannot split a grapheme cluster/akshara) rather than by whatever their own vocabulary induction happened to cover on these specific words - a guarantee holds on words not on this list too; empirical coverage of one fixed list does not.

| Tokenizer | Avg tokens/word (13 words) |
|---|--:|
| **Bornomala v1** | **1.00** |
| **Bornomala BMBT** | **1.00** |
| BanglaBERT (csebuetnlp) | 1.00 |
| BanglaT5 (csebuetnlp) | 1.00 |
| IndicBERTv2 (AI4Bharat) | 1.31 |
| SUTRA (TWO AI) | 3.31 |
| Sarvam-1 (Sarvam AI) | 3.46 |
| Param2-17B (BharatGen) | 3.62 |
| XLM-RoBERTa (Meta) | 3.69 |
| BrahmicTokenizer-131K (TSAI) | 4.38 |
| mBERT (Google) | 4.38 |
| GPT-4o (o200k) | 4.46 |
| DeepSeek-V3 | 4.62 |
| Krutrim | 4.77 |
| Gemma-2 (Google) | 5.69 |
| Qwen2.5 (Alibaba) | 9.15 |
| GPT-4 (cl100k) | 10.85 |
| Llama-3.1 (Meta) | 10.85 |
| Mistral-7B | 11.08 |

IndicBERTv2 is the closest tokenizer NOT tied at 1.00 (1.31 avg) but still fragments 3 of 13 (ঋত্বিক, বিষ্ণুপুর, শান্তিনিকেতন). A striking, unverified-but-notable pattern: every single value in Llama-3.1's row is byte-for-byte identical to GPT-4's cl100k row - its tokenizer is tiktoken-derived and does not appear to extend Bengali coverage at all versus cl100k. Gemma-2's row is real now (5.69), not "unavailable" as an earlier version of this page reported before access was granted 2026-08-18.

Reproduce: `python scripts/hard_words.py` (from `bengali-tokenizer/`), writes `benchmarks/hard-words.md`.

## Prior result (v0.1, superseded)

The first released version of this tokenizer (BPE, 32,000 vocabulary, trained on 12,000 Wikipedia articles only) measured fertility 1.39 / STRR 0.766 / conjunct fragmentation 0.0006 on its own held-out Wikipedia set (a different held-out slice than the one above, and measured before the bug fix described above, so not directly comparable). It is kept here for the historical record; it is no longer the shipped artifact. Full writeup of why vocabulary size mattered more than corpus mix alone: `docs/known-issues.md` point 7.
