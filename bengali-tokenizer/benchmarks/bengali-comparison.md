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

## Prior result (v0.1, superseded)

The first released version of this tokenizer (BPE, 32,000 vocabulary, trained on 12,000 Wikipedia articles only) measured fertility 1.39 / STRR 0.766 / conjunct fragmentation 0.0006 on its own held-out Wikipedia set (a different held-out slice than the one above, and measured before the bug fix described above, so not directly comparable). It is kept here for the historical record; it is no longer the shipped artifact. Full writeup of why vocabulary size mattered more than corpus mix alone: `docs/known-issues.md` point 7.
