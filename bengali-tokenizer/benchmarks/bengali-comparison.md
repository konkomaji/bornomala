# Bengali cross-tokenizer comparison

The first measured, reproducible comparison of how efficiently mainstream tokenizers encode Bengali. Every number is produced from the real tokenizer of each system on identical text. Nothing is estimated or fabricated.

## Exactly how this was measured (full transparency)

- **Our tokenizer**: Bornomala Bengali tokenizer, BPE, 64,000 vocabulary, trained on a literary-weighted corpus of 1.5M lines mixing Wikisource public-domain text, Sangraha verified/ben (pdf-typed as a formal/literary-register proxy, OCR-noise-filtered, and web-typed for general register), the first 15,000 Bengali Wikipedia articles, and XL-Sum Bengali news. Full corpus composition, what was substituted and why: `docs/known-issues.md` point 6. An ablation across 32k/48k/64k vocab on this same corpus showed fertility recovering monotonically with vocab size (point 7); 64k is the smallest size that beats IndicBERTv2 on every register tested.
- **Evaluation text**: four disjoint held-out sets, none touched during training. Wikipedia: 828 lines from articles after the first 15,000. Literary/formal, general web, news: reserved tails of Sangraha pdf-typed, Sangraha web-typed, and XL-Sum documents respectively, starting exactly where training's document budget ends (`bntok.corpus.build_register_held_out`).
- **Other tokenizers**: loaded from their official public releases and run with their own real tokenizers. Sarvam-1 (sarvamai/sarvam-1), IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only), XLM-RoBERTa (FacebookAI/xlm-roberta-base), mBERT (google-bert/bert-base-multilingual-cased), DeepSeek-V3 (deepseek-ai/DeepSeek-V3), GPT-4o (OpenAI o200k via tiktoken).
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
| XLM-RoBERTa (Meta) | 2.464 | 0.363 | 7.04 | 0.1019 |
| Sarvam-1 (Sarvam AI) | 2.593 | 0.415 | 6.69 | 0.1191 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.111 | 6.65 | n/a |
| mBERT (Google) | 2.777 | 0.385 | 6.25 | 0.1800 |
| DeepSeek-V3 | 2.994 | 0.089 | 5.79 | 0.2845 |

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

Step 4 is not complete: measured against v1's own held-out sets across all four registers now, but not yet checked against the published external baselines (Sarvam-1, SUTRA, IndicSuperTokenizer, BengaliBPE) the roadmap also names. Step 5 (featural encoding, morphology, the statistical fallback layer that would make fertility genuinely comparable) has not started.

## Prior result (v0.1, superseded)

The first released version of this tokenizer (BPE, 32,000 vocabulary, trained on 12,000 Wikipedia articles only) measured fertility 1.39 / STRR 0.766 / conjunct fragmentation 0.0006 on its own held-out Wikipedia set (a different held-out slice than the one above, and measured before the bug fix described above, so not directly comparable). It is kept here for the historical record; it is no longer the shipped artifact. Full writeup of why vocabulary size mattered more than corpus mix alone: `docs/known-issues.md` point 7.
