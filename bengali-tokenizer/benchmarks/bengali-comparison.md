# Bengali cross-tokenizer comparison

To our knowledge, the first fully reproducible benchmark comparing modern Bengali tokenizers across compression, word preservation, and conjunct fragmentation using a common evaluation pipeline. Every number is produced from the real tokenizer of each system on identical text. Nothing is estimated or fabricated.

## Exactly how this was measured (full transparency)

- **Our tokenizer**: Bornomala Bengali tokenizer, BPE, 64,000 vocabulary, trained on a literary-weighted corpus of 1.5M lines mixing Wikisource public-domain text, Sangraha verified/ben (pdf-typed as a formal/literary-register proxy, OCR-noise-filtered, and web-typed for general register), the first 15,000 Bengali Wikipedia articles, and XL-Sum Bengali news. Full corpus composition, what was substituted and why: `docs/known-issues.md` point 6. An ablation across 32k/48k/64k vocab on this same corpus showed fertility recovering monotonically with vocab size (point 7); 64k is the smallest size that beats IndicBERTv2 on every register tested.
- **Evaluation text**: four disjoint held-out sets, none touched during training. Wikipedia: 828 lines from articles after the first 15,000. Literary/formal, general web, news: reserved tails of Sangraha pdf-typed, Sangraha web-typed, and XL-Sum documents respectively, starting exactly where training's document budget ends (`bntok.corpus.build_register_held_out`).
- **Other tokenizers**: loaded from their official public releases and run with their own real tokenizers. Sarvam-1 (sarvamai/sarvam-1), IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only), XLM-RoBERTa (FacebookAI/xlm-roberta-base), mBERT (google-bert/bert-base-multilingual-cased), DeepSeek-V3 (deepseek-ai/DeepSeek-V3), GPT-4o (OpenAI o200k via tiktoken); on the Wikipedia held-out set only, also SUTRA (TWO/sutra-mlt256-v2) and Krutrim (krutrim-ai-labs/Krutrim-2-instruct). BrahmicTokenizer-131K (theschoolofai/BrahmicTokenizer-131K, Apache-2.0, arXiv:2605.29379) was added on 2026-08-16: it is the third baseline named in the whitepaper's own Gate G2 list, it does have a real public release, and it had simply never been run here before that date. That was a coverage gap in this benchmark, not an availability problem on their side, and it is recorded as such. Two baselines named in the v2 design doc's roadmap still could not be added, honestly rather than faked: IndicSuperTokenizer (arXiv:2511.03237) has no public code/tokenizer release found; the only similarly-named BengaliBPE (arXiv:2511.05324) Hugging Face repo found fails to load and is not verifiably the paper's own artifact.
- **Metrics**: all text NFC-normalised first. Fertility = tokens / whitespace-words (lower is better). STRR = fraction of words kept as a single token. Bytes/token = UTF-8 bytes / tokens. Conjunct fragmentation = fraction of Bengali grapheme clusters that a token boundary splits, computed from each tokenizer's own character offsets (GPT-4o gives no offsets, so its fragmentation is not measured). A small number of held-out lines that quote foreign-script text (Greek, Arabic, Japanese — genuinely present in Wikipedia and news text) fall outside this tokenizer's guaranteed coverage (Bengali block + ASCII, see `docs/known-issues.md` point 4) and are excluded from the fragmentation count specifically, since that is a documented, separate scope boundary, not a conjunct-splitting question; they still count normally toward fertility/STRR/bytes. At most 11 of 828-28,461 lines per register were excluded this way.
- **Reproduce**:
  ```
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --skip 15000 --limit 800
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register literary_formal --limit 1000
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register general_web --limit 1000
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register news --limit 1000
  ```
  Raw JSON for each run: `comparison-wikipedia.json`, `comparison-literary_formal.json`, `comparison-general_web.json`, `comparison-news.json` (this directory).

## Results: Wikipedia held-out (sorted by fertility, best first)

| Tokenizer | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| Bornomala Track A (bpe 64000) **(ours)** | 1.524 | 0.722 | 11.38 | 0.0001 |
| IndicBERTv2 (AI4Bharat) | 1.652 | 0.612 | 10.50 | 0.0440 |
| SUTRA (TWO AI) | 2.218 | 0.419 | 7.82 | 0.1579 |
| XLM-RoBERTa (Meta) | 2.464 | 0.363 | 7.04 | 0.1019 |
| Sarvam-1 (Sarvam AI) | 2.593 | 0.415 | 6.69 | 0.1191 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.111 | 6.65 | n/a |
| BrahmicTokenizer-131K (TSAI) | 2.620 | 0.154 | 6.62 | 0.2209 |
| mBERT (Google) | 2.777 | 0.385 | 6.25 | 0.1800 |
| DeepSeek-V3 | 2.994 | 0.089 | 5.79 | 0.2845 |
| Krutrim (Krutrim AI) | 3.207 | 0.076 | 5.41 | 0.2859 |

BrahmicTokenizer-131K across all four registers, alongside the two closest rows for scale:

| Register | Ours | BrahmicTokenizer-131K | GPT-4o (o200k) |
|---|--:|--:|--:|
| Wikipedia | **1.524** | 2.620 | 2.608 |
| Literary/formal | **1.320** | 2.449 | 2.456 |
| General web | **1.201** | 2.267 | 2.266 |
| News | **1.140** | 2.184 | 2.192 |

Its conjunct fragmentation is 0.2209 / 0.2378 / 0.2004 / 0.1949 on the four registers respectively, against our 0.0000-0.0001.

**Being aimed at Indic scripts is not, by itself, enough, and this is the row that shows it.** BrahmicTokenizer-131K is built specifically as an Indic-capable drop-in replacement for OpenAI's o200k, and it carries 131,072 tokens against our 64,000. On Bengali it lands within 0.01 of GPT-4o on every one of the four registers, and it splits roughly a fifth of all conjuncts, worse than script-blind mBERT on three registers of four. Targeting Indic scripts in the training mix is a different thing from constraining the merge space to the script's own units.

**The vocabulary comparison cuts both ways and is stated in both directions.** BrahmicTokenizer has just over twice our raw budget, which favours it; it also spreads that budget across 12 Brahmic-script languages, roughly 11k per language, where ours is spent entirely on Bengali, which favours us. The same caveat applies to this whole table: every external baseline in it is multilingual, so a monolingual Bengali tokenizer beating them on Bengali is a consequence of that design choice rather than a surprising discovery. The comparison is still the right one to run, because these are the tokenizers Bengali text is actually encoded by in practice.

SUTRA and Krutrim are now measured on all four registers (added after the initial Wikipedia-only pass):

| Register | SUTRA fertility / STRR / frag | Krutrim fertility / STRR / frag |
|---|--:|--:|
| Wikipedia | 2.218 / 0.419 / 0.1579 | 3.207 / 0.076 / 0.2859 |
| Literary/formal | 2.192 / 0.402 / 0.1800 | 3.181 / 0.074 / 0.3403 |
| General web | 2.066 / 0.432 / 0.1620 | 3.135 / 0.049 / 0.3234 |
| News | 1.954 / 0.468 / 0.1509 | 3.061 / 0.035 / 0.3232 |

Neither changes the ranking: our tokenizer (and BMBT, see below) outperforms every other tokenizer on every register; SUTRA is consistently 3rd; Krutrim is consistently last or near-last.

## Results across every register (ours vs. AI4Bharat IndicBERTv2, the closest rival)

| Register | Fertility (ours / IndicBERTv2) | STRR (ours / IndicBERTv2) | Conjunct fragmentation (ours / IndicBERTv2) |
|---|--:|--:|--:|
| Wikipedia | **1.524** / 1.652 | **0.722** / 0.612 | **0.0001** / 0.0440 |
| Literary / formal | **1.320** / 1.612 | **0.789** / 0.607 | **0.0001** / 0.0562 |
| General web | **1.201** / 1.395 | **0.861** / 0.715 | **0.0001** / 0.0277 |
| News | **1.140** / 1.312 | **0.893** / 0.755 | **0.0000** / 0.0206 |

## What it shows

- Our tokenizer needs the fewest tokens per Bengali word, keeps the most whole words, and splits the fewest conjuncts — **on every one of the four registers tested**, not just the Wikipedia set most comparisons stop at.
- Conjunct fragmentation is essentially zero across the board (0.0000-0.0001), 200x to 560x lower than IndicBERTv2's, which itself is the best of the non-Bornomala tokenizers.
- Every other tokenizer splits Bengali conjuncts far more often on every register: IndicBERTv2 between roughly 2% and 6%, the rest between 7% and 31%. Splitting a conjunct corrupts the written unit a reader recognises; this is the property no general Indic tokenizer controls for.

## A real measurement bug caught mid-analysis (kept here, not quietly fixed away)

An early version of this comparison's own fragmentation counter for our tokenizer had two bugs, found via a hard assertion added specifically to catch this class of mistake: (1) an off-by-one from the Metaspace word-boundary marker's leading space, which manufactured thousands of false fragmentation hits; (2) `encode_tokens()`'s debug view returning the literal string `<unk>` in place of missing text for out-of-coverage foreign-script codepoints, rather than an empty placeholder, which could silently desynchronise offsets on such lines. Both are now fixed in `scripts/compare.py`, and the fix asserts the reconstruction is exact on every line it measures, so a similar bug cannot silently corrupt a comparison again. The numbers above are from the corrected measurement. Full account: `docs/known-issues.md` point 8.

## v2 roadmap step 4: the akshara finite-state parser, measured (not a like-for-like row)

The v2 design's own roadmap (`docs/design/reading-bengali-on-its-own-terms.md`) calls for benchmarking the akshara parser (`bntok/akshara.py`) before building anything on top of it. Measured via `scripts/compare.py`'s `measure_akshara()`, on the exact same four held-out sets as v1's own results above (828 Wikipedia lines; the three register held-outs, same document budgets):

| Register | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| Wikipedia | 4.527 | 0.045 | 3.83 | **0.0000** |
| Literary / formal | 4.055 | 0.078 | 3.99 | 0.000005 (13/2,868,557) |
| General web | 4.189 | 0.041 | 4.03 | **0.0000** |
| News | 4.111 | 0.035 | 4.14 | **0.0000** |
| Bornomala Track A bpe 64000 (v1, Wikipedia, for reference) | 1.524 | 0.722 | 11.38 | 0.0001 |

**This is not a like-for-like comparison and is reported separately on purpose.** The akshara parser has no vocabulary and no merges yet (v2 roadmap step 5 is not built): its "fertility" is the number of un-merged akshara/other chunks per word, the pre-compression granularity, not a trained tokenizer's post-BPE-merge token count. It needing roughly 3x more units per word than v1's compressed BPE tokenizer is expected, not a regression, exactly the number the roadmap's own step 4 asks to be measured and reported honestly before proceeding to step 5.

**Conjunct fragmentation is the one column above that IS a fair, like-for-like number**, since it does not depend on compression: **exactly 0.0000 on three of the four registers** (verified directly, not assumed) — better than v1's own 0.0001 on Wikipedia, which still carries a small residual from its atom-frequency threshold (`docs/known-issues.md` point 1). The akshara parser's guarantee has no such threshold: it is 0 by grammar construction, on every cluster the corpus happens to contain or not.

**Literary/formal measured 0.000005 (13 out of 2,868,557 clusters), not exactly 0 — investigated, not left unexplained.** All 13 are a Bengali consonant directly followed by a Devanagari combining mark (U+093E/U+093F/U+0902), almost certainly OCR/font-mapping noise in the scanned 19th/20th-century literature this register draws from (which includes a Mahabharata translation, `docs/known-issues.md` point 6). `\X` clusters cross-script combining marks with any preceding base regardless of script; this parser's Matra/Modifier sets are deliberately Bengali-block only, so it correctly does not absorb a foreign-script mark into a Bengali consonant's chunk. This is a documented scope boundary, not a bug — see `docs/known-issues.md` point 14 for the full account and why it is not "fixed" by widening the grammar.

**Three real bugs were found and fixed by running this measurement against real Wikipedia text first, not synthetic tests**, all now covered by `tests/test_akshara.py` and documented in `docs/known-issues.md` points 11-13: an independent vowel followed by a virama does not chain into a further consonant the way a consonant does; a Modifier (not a Matra or Nukta) blocks conjunct-chain continuation; and ZWJ/ZWNJ are not tied to a fixed position relative to the virama the way an earlier pass assumed. The first fragmentation measurement on the Wikipedia set, before these fixes, was 0.0012 — worse than v1's 0.0001 — and is kept in the commit history rather than quietly replaced, the same way the bug in point 8 above is.

Reproduce:
```
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --skip 15000 --limit 800
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register literary_formal --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register general_web --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register news --limit 1000
```
The akshara row is printed alongside the tokenizer comparison in each case; raw numbers also in each `comparison-*.json`'s `akshara_v2` key.

**Step 4 is now complete**: measured against v1's own held-out sets across all four registers, and against every real external baseline that has a usable public release (Sarvam-1, SUTRA, Krutrim, IndicBERTv2, XLM-RoBERTa, mBERT, DeepSeek-V3, GPT-4o), on all four registers. IndicSuperTokenizer and BengaliBPE remain unavailable, reported honestly rather than faked (see the "Other tokenizers" bullet above).

## v2 roadmap step 5 (partial): BMBT, measured (a genuine like-for-like row)

BMBT (Bornomala's Bengali Tokenizer, `bntok/bmbt.py`) is grammar (the akshara parser above) plus a featural decomposition (`featurize()`) plus a statistical BPE layer over akshara atoms - the same architecture as v1, with the atomic unit swapped from grapheme cluster to akshara. **Morphology is explicitly not built yet.** Full design: `docs/bmbt-architecture.md`.

Unlike the raw akshara-parser row above, a trained BMBT has a real vocabulary and real merges the same way `bn-bpe-64k` does, so it is a genuine like-for-like fertility comparison, trained on the identical corpus (`configs/bpe-64k.json`, same vocab size 64000) for a controlled comparison:

| Register | Fertility (v1 / BMBT) | STRR (v1 / BMBT) | Bytes/token (v1 / BMBT) | Conjunct frag. (v1 / BMBT) |
|---|--:|--:|--:|--:|
| Wikipedia | 1.524 / 1.524 | 0.722 / 0.722 | 11.38 / 11.38 | 0.000075 / 0.000075 |
| Literary/formal | 1.320 / 1.320 | 0.789 / 0.789 | 12.25 / 12.25 | 0.000104 / 0.000112 |
| General web | 1.201 / 1.201 | 0.861 / 0.861 | 14.07 / 14.07 | 0.000055 / 0.000057 |
| News | 1.140 / 1.140 | 0.893 / 0.894 | 14.93 / 14.93 | 0.000025 / 0.000025 |

**Reported exactly as measured, not the outcome anyone assumed going in.** On Wikipedia the two are not just close but identical down to the raw integer counts (17,245 tokens, 11,316 words, 3 fragmented clusters, both tokenizers) despite the two artifacts having genuinely different vocabularies (12,233 atoms for v1, 12,199 for BMBT - verified directly, not a coincidence of rounding). On the three larger registers, tiny real, non-identical differences appear in *both* directions: BMBT needs marginally fewer tokens (25-60 fewer out of 370,000-1,230,000, roughly 0.005-0.02%) but has marginally more fragmented clusters (2-23 more, still within the same 0.00005-0.0001 near-zero band as v1). Neither direction is large enough to call a win. This is an honest tie.

**Why a tie, not a loss**, given `docs/design/FORMAL_SPEC.md`'s own proof that a constrained BPE cannot beat an unconstrained one on raw token count: akshara-grammar boundaries are already nearly identical to `\X`'s grapheme-cluster boundaries on well-formed Bengali (the akshara-parser measurement above, points 11-14 in `docs/known-issues.md`), so constraining BPE to akshara boundaries instead of grapheme-cluster boundaries barely constrains anything further in practice - the two atom schemes are close to isomorphic on real text.

**What BMBT adds, independent of the fertility tie**: a provable, Unicode-library-independent grammar instead of delegated trust in `regex`'s own `\X`, and `featurize()` - a real, tested structural decomposition (onset/vowel/modifier per akshara) v1 never had, at zero fertility cost.

### CC-100 ablation

Trained both architectures again with CC-100 Bengali added to the corpus (`configs/bpe-64k-cc100.json`, same weights plus `cc100_general_web`; see `docs/known-issues.md` point 15 for a real bug found and fixed in `stream_cc100` while running this):

| Register | Fertility, no CC-100 | Fertility, +CC-100 | Change |
|---|--:|--:|--:|
| Wikipedia | 1.524 | 1.531 | +0.007 (slightly worse) |
| General web | 1.201 | 1.199 | -0.002 (slightly better) |

Both directions make sense: the same 64,000-token vocabulary budget now spans five sources instead of four, diluting Wikipedia-specific coverage slightly, while general web (the register CC-100 actually targets) gets a small real benefit. Both effects round to the third decimal place - a wash, not a case for or against adopting CC-100 in the default weights. `bn-bpe-64k` and `bmbt-64k` (without CC-100) remain the recommended artifacts; `bn-bpe-64k-cc100`/`bmbt-64k-cc100` are kept as the ablation record, not shipped as a recommendation. v1 and BMBT tie exactly on this ablation too (identical fertility/STRR/bytes/fragmentation on both registers tested).

Reproduce:
```
python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --out artifacts/bmbt-64k
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --skip 15000 --limit 800
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register literary_formal --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register general_web --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --register news --limit 1000
```

**Still not done**: morphology. BMBT's featural output has no morphological layer yet, so it cannot yet claim the "quality-per-token" advantage the design doc's own risk section frames as the actual bet worth making - that claim needs a downstream task evaluation this project does not have at all yet, and remains completely unmeasured.

## Hard words: conjuncts and Bengali place names

The registers above are a corpus average; it can hide how a tokenizer treats
specific, culturally load-bearing words. A fixed list of 13 - deity names, a
national poet, well-known West Bengal places, all conjunct-dense - measured
on every tokenizer this project tracks (`scripts/hard_words.py`, imported
model list shared with `scripts/compare.py` so it can never drift out of
sync). Full per-word table and every tokenizer: `benchmarks/hard-words.md`.

**Ours (v1 and BMBT) tokenizes every one of the 13 words as exactly one
token. No exception, including the triple-conjunct আকাঙ্ক্ষা and the
multi-akshara রবীন্দ্রনাথ.**

| Word | Meaning | Ours (v1/BMBT) | IndicBERTv2 (best rival) | GPT-4o |
|---|---|--:|--:|--:|
| স্ত্রী | wife/woman | 1 | 1 | 2 |
| আকাঙ্ক্ষা | aspiration | 1 | 1 | 6 |
| রবীন্দ্রনাথ | Rabindranath (Tagore) | 1 | 1 | 7 |
| পশ্চিমবঙ্গ | West Bengal | 1 | 1 | 5 |
| বিষ্ণুপুর | Bishnupur | 1 | 2 | 5 |
| শান্তিনিকেতন | Santiniketan | 1 | 3 | 5 |

Average tokens/word over all 13 words, every tokenizer measured:

| Tokenizer | Avg tokens/word |
|---|--:|
| **Bornomala v1 / BMBT (ours)** | **1.00** |
| IndicBERTv2 (AI4Bharat) | 1.31 |
| SUTRA (TWO AI) | 3.31 |
| Sarvam-1 | 3.46 |
| Param2-17B (BharatGen) | 3.62 |
| XLM-RoBERTa (Meta) | 3.69 |
| mBERT (Google) | 4.38 |
| GPT-4o (o200k) | 4.46 |
| DeepSeek-V3 | 4.62 |
| Krutrim | 4.77 |
| Qwen2.5 (Alibaba) | 9.15 |
| GPT-4 (cl100k) | 10.85 |
| Llama-3.1 (Meta) | 10.85 |
| Mistral-7B | 11.08 |
| Gemma-2 (Google) | unavailable, gated repo |

IndicBERTv2 is the only real rival (1.31 avg) but still fragments 3 of 13
(ঋত্বিক, বিষ্ণুপুর, শান্তিনিকেতন) - even India's best-funded Indic
tokenizer cannot hold conjunct integrity on every word here. Ours does, by
construction, not luck. A striking, unverified-but-notable pattern: every
single value in Llama-3.1's row is byte-for-byte identical to GPT-4's
cl100k row - its tokenizer is tiktoken-derived and does not appear to
extend Bengali coverage at all versus cl100k.

Reproduce: `python scripts/hard_words.py` (from `bengali-tokenizer/`),
writes `benchmarks/hard-words.md`.

## Prior result (v0.1, superseded)

The first released version of this tokenizer (BPE, 32,000 vocabulary, trained on 12,000 Wikipedia articles only) measured fertility 1.39 / STRR 0.766 / conjunct fragmentation 0.0006 on its own held-out Wikipedia set (a different held-out slice than the one above, and measured before the bug fix described above, so not directly comparable). It is kept here for the historical record; it is no longer the shipped artifact. Full writeup of why vocabulary size mattered more than corpus mix alone: `docs/known-issues.md` point 7.
