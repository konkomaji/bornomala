# Hard words: conjuncts and Bengali place names

Not a held-out corpus average - a fixed, small, qualitative word list (deity names, a national poet, well-known West Bengal places), chosen for cultural resonance and conjunct density. Regenerate with `python scripts/hard_words.py` from `bengali-tokenizer/`.

| Word | Note | Bornomala v1 | Bornomala BMBT | GPT-4o (OpenAI o200k) | GPT-4 (OpenAI cl100k) | Sarvam-1 (Sarvam AI) | SUTRA (TWO AI) | BrahmicTokenizer-131K (TSAI) | Krutrim (Krutrim AI) | Param2-17B (BharatGen) | IndicBERTv2 (AI4Bharat) | mBERT (Google) | XLM-RoBERTa (Meta) | DeepSeek-V3 | Llama-3.1 (Meta) | Gemma-2 (Google) | Mistral-7B (Mistral AI) | Qwen2.5 (Alibaba) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| স্ত্রী | wife/woman - classic hard conjunct | 1 | 1 | 2 | 8 | 1 | 3 | 2 | 3 | 2 | 1 | 2 | 1 | 3 | unavailable | unavailable | 7 | 7 |
| আকাঙ্ক্ষা | aspiration - triple conjunct ঙ্ক্ষ | 1 | 1 | 6 | 11 | 4 | 5 | 6 | 6 | 4 | 1 | 7 | 5 | 6 | unavailable | unavailable | 12 | 9 |
| ঋত্বিক | name - ত্ব conjunct | 1 | 1 | 4 | 8 | 6 | 3 | 3 | 4 | 4 | 2 | 3 | 4 | 5 | unavailable | unavailable | 9 | 7 |
| বিজ্ঞান | science - জ্ঞ conjunct | 1 | 1 | 3 | 8 | 1 | 1 | 3 | 3 | 1 | 1 | 3 | 1 | 3 | unavailable | unavailable | 10 | 7 |
| স্বাধীনতা | independence - স্ব conjunct | 1 | 1 | 3 | 12 | 1 | 1 | 3 | 4 | 3 | 1 | 1 | 1 | 3 | unavailable | unavailable | 12 | 10 |
| কৃষ্ণ | Krishna - ষ্ণ conjunct | 1 | 1 | 3 | 8 | 2 | 1 | 3 | 4 | 3 | 1 | 4 | 4 | 3 | unavailable | unavailable | 10 | 6 |
| রবীন্দ্রনাথ | Rabindranath (Tagore) - ন্দ্র conjunct | 1 | 1 | 7 | 13 | 6 | 7 | 7 | 7 | 5 | 1 | 6 | 5 | 5 | unavailable | unavailable | 14 | 12 |
| পশ্চিমবঙ্গ | West Bengal - শ্চ conjunct | 1 | 1 | 5 | 14 | 3 | 3 | 5 | 4 | 6 | 1 | 3 | 4 | 6 | unavailable | unavailable | 13 | 10 |
| বর্ধমান | Bardhaman - র্ধ reph conjunct | 1 | 1 | 3 | 9 | 3 | 3 | 3 | 4 | 1 | 1 | 4 | 4 | 4 | unavailable | unavailable | 10 | 7 |
| মুর্শিদাবাদ | Murshidabad - র্শ reph conjunct | 1 | 1 | 7 | 13 | 5 | 4 | 7 | 6 | 5 | 1 | 6 | 5 | 6 | unavailable | unavailable | 12 | 12 |
| বিষ্ণুপুর | Bishnupur - ষ্ণ conjunct | 1 | 1 | 5 | 13 | 4 | 4 | 5 | 6 | 4 | 2 | 6 | 5 | 5 | unavailable | unavailable | 12 | 11 |
| শান্তিনিকেতন | Santiniketan - ন্ত conjunct | 1 | 1 | 5 | 14 | 5 | 4 | 5 | 6 | 5 | 3 | 7 | 4 | 6 | unavailable | unavailable | 13 | 12 |
| দার্জিলিং | Darjeeling - র্জ reph conjunct | 1 | 1 | 5 | 10 | 4 | 4 | 5 | 5 | 4 | 1 | 5 | 5 | 5 | unavailable | unavailable | 10 | 9 |

## Averages (available models only)

| Tokenizer | Avg tokens/word |
|---|--:|
| Bornomala v1 | 1.00 |
| Bornomala BMBT | 1.00 |
| GPT-4o (OpenAI o200k) | 4.46 |
| GPT-4 (OpenAI cl100k) | 10.85 |
| Sarvam-1 (Sarvam AI) | 3.46 |
| SUTRA (TWO AI) | 3.31 |
| BrahmicTokenizer-131K (TSAI) | 4.38 |
| Krutrim (Krutrim AI) | 4.77 |
| Param2-17B (BharatGen) | 3.62 |
| IndicBERTv2 (AI4Bharat) | 1.31 |
| mBERT (Google) | 4.38 |
| XLM-RoBERTa (Meta) | 3.69 |
| DeepSeek-V3 | 4.62 |
| Llama-3.1 (Meta) | unavailable |
| Gemma-2 (Google) | unavailable |
| Mistral-7B (Mistral AI) | 11.08 |
| Qwen2.5 (Alibaba) | 9.15 |
