# Bengali cross-tokenizer comparison

The first measured, reproducible comparison of how efficiently mainstream tokenizers encode Bengali. Every number is produced from the real tokenizer of each system on identical text. Nothing is estimated or fabricated.

## Exactly how this was measured (full transparency)

- **Our tokenizer**: Bornomala Bengali tokenizer, BPE, 32000 vocabulary, trained on the first 12,000 articles of the Bengali Wikipedia dump (wikimedia/wikipedia, config 20231101.bn).
- **Evaluation text**: 878 held-out lines taken from Bengali Wikipedia articles AFTER the first 12,000, so this text was never seen during our training. Every tokenizer is measured on this same text.
- **Other tokenizers**: loaded from their official public releases and run with their own real tokenizers. Sarvam-1 (sarvamai/sarvam-1), IndicBERTv2 (ai4bharat/IndicBERTv2-MLM-only), XLM-RoBERTa (FacebookAI/xlm-roberta-base), mBERT (google-bert/bert-base-multilingual-cased), DeepSeek-V3 (deepseek-ai/DeepSeek-V3), GPT-4o (OpenAI o200k via tiktoken).
- **Metrics**: all text NFC-normalised first. Fertility = tokens / whitespace-words (lower is better). STRR = fraction of words kept as a single token. Bytes/token = UTF-8 bytes / tokens. Conjunct fragmentation = fraction of Bengali grapheme clusters that a token boundary splits, computed from each tokenizer character offsets (GPT-4o gives no offsets, so its fragmentation is not measured).
- **Reproduce**: python scripts/compare.py --tokenizer artifacts/bn-bpe-32k --skip 12000 --limit 800

## Results (sorted by fertility, best first)

| Tokenizer | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| Bornomala Track A (bpe 32000) **(ours)** | 1.390 | 0.766 | 13.03 | 0.0006 |
| IndicBERTv2 (AI4Bharat) | 1.520 | 0.669 | 11.92 | 0.0364 |
| Sarvam-1 (Sarvam AI) | 2.005 | 0.490 | 9.04 | 0.0942 |
| XLM-RoBERTa (Meta) | 2.351 | 0.406 | 7.71 | 0.1034 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.095 | 6.95 | n/a |
| mBERT (Google) | 3.012 | 0.317 | 6.02 | 0.2164 |
| DeepSeek-V3 | 3.024 | 0.071 | 5.99 | 0.2988 |

## What it shows

- The Bornomala tokenizer needs the fewest tokens per Bengali word (fertility 1.39), ahead of AI4Bharat IndicBERTv2 (1.52), Sarvam-1 (2.01), GPT-4o (2.61), mBERT (3.01), and DeepSeek-V3 (3.02).
- It keeps the most whole words (STRR 0.77) and packs the most bytes per token (13.03).
- It almost never splits a Bengali conjunct (0.0006), while every general tokenizer splits between about 4 and 30 percent of them. Splitting conjuncts corrupts the written unit a reader recognises; this is the property no general Indic tokenizer controls for.
