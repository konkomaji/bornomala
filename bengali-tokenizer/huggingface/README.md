---
license: apache-2.0
language:
- bn
- en
library_name: tokenizers
pipeline_tag: token-classification
tags:
- bengali
- bangla
- tokenizer
- tokenization
- grapheme
- indic
- unicode
- bornomala
datasets:
- wikimedia/wikipedia
- wikimedia/wikisource
- ai4bharat/sangraha
- csebuetnlp/xlsum
metrics:
- name: Fertility (tokens per word, held-out Bengali Wikipedia)
  type: fertility
  value: 1.524
- name: Destructive fragmentation rate (corrected measure)
  type: fragmentation
  value: 0.0004
---

# Bornomala Bengali Tokenizer

A Bengali-first, grapheme-cluster-aware tokenizer that **never splits a conjunct**.
It is the first component of [Project Bornomala](https://github.com/konkomaji/bornomala),
a non-commercial research effort from West Bengal to build a Bengali-first,
dialect-aware language model and to preserve the Bengali language and its dialects.

> **Version 0.2.** Trained on a literary-weighted corpus (Wikisource,
> AI4Bharat Sangraha, Wikipedia, XL-Sum news), 64k vocabulary. Supersedes the
> v0.1 Wikipedia-only model.

## Key features

- **Conjunct integrity by design.** Text is normalised to NFC, segmented into
  Unicode UAX #29 grapheme clusters, and each cluster is remapped to an atomic
  symbol before subword training. Every learned token is a whole number of
  grapheme clusters, so a Bengali conjunct is never split across a boundary.
- **Efficient.** Lowest tokens per word, on every register tested (Wikipedia,
  literary/formal, general web, news), with a 64k vocabulary.
- **Robust.** The whole Bengali Unicode block, Bengali's shared sentence
  punctuation (danda/double danda), and ASCII are guaranteed, so any Bengali or
  code-mixed English text round-trips exactly.
- **Open and CPU-only.** Apache 2.0. No GPU needed to train or use.

## Performance

Measured on 828 held-out Bengali Wikipedia lines (unseen during training); three
further disjoint held-out registers (literary/formal, general web, news) confirm
the same ranking beyond Wikipedia, see the repository's
`benchmarks/bengali-comparison.md`. Every other tokenizer is its real public
tokenizer on the same NFC-normalised text. To our knowledge, this is the first
fully reproducible benchmark comparing modern Bengali tokenizers across
compression, word preservation, and conjunct fragmentation using a common
evaluation pipeline.

| Tokenizer | Fertility | STRR | Bytes/token | Destructive rate |
|---|--:|--:|--:|--:|
| **Bornomala (this model)** | **1.524** | **0.722** | **11.38** | **0.0004** |
| BanglaBERT (csebuetnlp) | 1.625 | 0.649 | 10.67 | 0.0162 |
| IndicBERTv2 (AI4Bharat) | 1.652 | 0.612 | 10.50 | 0.0191 |
| BanglaT5 (csebuetnlp) | 1.669 | 0.628 | 10.39 | 0.0088 |
| XLM-RoBERTa (Meta) | 2.464 | 0.363 | 7.04 | 0.0627 |
| Sarvam-1 (Sarvam AI) | 2.593 | 0.415 | 6.69 | 0.0364 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.111 | 6.65 | n/a |
| BrahmicTokenizer-131K (TSAI) | 2.620 | 0.154 | 6.62 | 0.0820 |
| mBERT (Google) | 2.777 | 0.385 | 6.25 | 0.1552 |
| DeepSeek-V3 | 2.994 | 0.089 | 5.79 | 0.1031 |

Lower fertility and lower destructive rate are better; higher STRR and bytes
per token are better. Fewer tokens per word means lower cost and more usable
context. Destructive rate counts only splits that sever something real (a
stranded virama, a detached nukta), not a harmless consonant-cluster/vowel-sign
seam - the corrected replacement for a cruder binary fragmentation count.
Every general tokenizer breaks between 0.9% and 15.5% of Bengali conjuncts
destructively on this held-out set; this one breaks 0.04%. Also measured on
literary/formal, general web, news, romanized Banglish, and FLORES+ (the
exact corpus an external tokenizer-fertility paper's own numbers come from):
see `benchmarks/bengali-comparison.md` in the repository for all six
registers and the BMBT sibling tokenizer's numbers alongside this one.

## Hard words: conjuncts and Bengali place names

A register average can hide how a tokenizer treats specific, culturally
load-bearing words. A fixed list of 13 - deity names, the national poet
Rabindranath Tagore, well-known West Bengal places, all conjunct-dense -
measured on every tokenizer this project tracks (16 total).

**This model tokenizes every one of the 13 words as exactly one token**,
including the triple-conjunct আকাঙ্ক্ষা and the multi-akshara রবীন্দ্রনাথ.
**Correction to an earlier version of this card**: this used to say "no
exception" as if this model were the only one - BanglaBERT and BanglaT5
(csebuetnlp, added 2026-08-17) also score a perfect 1.00 average here. What
still sets this model apart: it guarantees the result by construction
(grammar cannot split a grapheme cluster), not by whatever a vocabulary
happened to cover on these 13 specific words.

| Word | Meaning | Ours | BanglaBERT/BanglaT5 | IndicBERTv2 | GPT-4o |
|---|---|--:|--:|--:|--:|
| স্ত্রী | wife/woman | **1** | 1 / 1 | 1 | 2 |
| আকাঙ্ক্ষা | aspiration | **1** | 1 / 1 | 1 | 6 |
| রবীন্দ্রনাথ | Rabindranath (Tagore) | **1** | 1 / 1 | 1 | 7 |
| পশ্চিমবঙ্গ | West Bengal | **1** | 1 / 1 | 1 | 5 |
| বিষ্ণুপুর | Bishnupur | **1** | 1 / 1 | 2 | 5 |
| শান্তিনিকেতন | Santiniketan | **1** | 1 / 1 | 3 | 5 |

Average tokens/word over all 13 words, all 19 tokenizers measured: **ours,
BanglaBERT, and BanglaT5 all tie at 1.00**; IndicBERTv2, the closest
tokenizer not tied, averages 1.31 and still fragments 3 of the 13 words; the
rest (SUTRA, Sarvam-1, Param2-17B, BrahmicTokenizer-131K, XLM-RoBERTa, mBERT,
GPT-4o, DeepSeek-V3, Krutrim, Gemma-2, Qwen2.5, GPT-4 cl100k, Llama-3.1,
Mistral-7B) run 3.31-11.08 (Gemma-2 5.69 - real now, access was gated until
2026-08-18). Full per-word table and the reproduce command:
[`benchmarks/hard-words.md`](https://github.com/konkomaji/bornomala/blob/main/bengali-tokenizer/benchmarks/hard-words.md).

## Usage

The tokenizer uses a grapheme-atom scheme, so encode and decode go through the
`bntok` helper, which handles NFC normalisation and the cluster-to-atom remap.

```bash
pip install "bntok @ git+https://github.com/konkomaji/bornomala#subdirectory=bengali-tokenizer"
# or clone the repo and: pip install -e bengali-tokenizer
```

```python
from huggingface_hub import snapshot_download
from bntok import BengaliTokenizer

path = snapshot_download("konko/bornomala-bengali-tokenizer")
# the config file is stored as bornomala_config.json; rename to config.json in the folder,
# or copy the three files (tokenizer.json, atoms.json, config.json) into one directory.
tok = BengaliTokenizer.load(path)

ids = tok.encode("আমি বাংলায় ক্ষুদ্র গান গাই")
assert tok.decode(ids) == "আমি বাংলায় ক্ষুদ্র গান গাই"   # exact round-trip
print(len(ids), "tokens")
```

To simply count tokens with the raw atom-space model (advanced), load
`tokenizer.json` with the `tokenizers` library, but note it expects atom-remapped
input; the `bntok` wrapper is the supported path.

## Training details

- Algorithm: BPE, vocabulary 64,000.
- Corpus: literary-weighted, 1.5M lines: Wikisource Bengali public-domain text,
  AI4Bharat Sangraha verified/ben (pdf-typed as a formal/literary proxy,
  OCR-noise-filtered, and web-typed for general register), the first 15,000
  articles of `wikimedia/wikipedia` config `20231101.bn`, and XL-Sum Bengali
  news. Full mix and what was substituted: repository `docs/known-issues.md`
  point 6.
- Normalisation: NFC plus explicit re-composition of the three Bengali letters
  NFC leaves decomposed (ড়/ঢ়/য়), and a preserve-by-default ZWJ/ZWNJ policy.
- Hardware: CPU only.

## Limitations

Two of the corpus's six configured sources (government/administrative text,
code-mixed Bengali-English) have no clean public dataset and are omitted. The
literary/formal register is a proxy (Sangraha pdf-typed documents: genuinely
old-orthography and OCR-noisy, but not confirmed pre-1950 public domain).
Fragmentation is near zero, not exactly zero, because rare sub-threshold
clusters decompose. See the repository's `docs/known-issues.md` for the full,
honest list, including two real bugs found and fixed in the comparison script
itself.

## BMBT: a v2 tokenizer now exists (not published here yet)

The GitHub repository now also has **BMBT (Bornomala's Bengali Tokenizer)**,
a v2 tokenizer that parses Bengali's akshara grammar directly with a
finite-state machine instead of discovering structure statistically, and
adds a real featural decomposition (`featurize()`) as an output of the
tokenizer itself. Measured against this model on identical held-out text,
BMBT ties it rather than beating it - reported honestly, matching the
design's own formal proof that a grammar-constrained subword model cannot
beat an unconstrained one on raw token count. BMBT also carries a morphology
layer that aligns its token boundaries to Bengali's suffix structure, and a
vectorized segmenter that runs at more than twice the throughput of the C
regex this model delegates to.
This Hugging Face listing still serves the model above (v1, unchanged);
BMBT has not been published as a separate Hugging Face model yet. See
[the GitHub repository](https://github.com/konkomaji/bornomala/tree/main/bengali-tokenizer)
for BMBT's code, architecture doc, and full measured comparison.

## Citation

```bibtex
@software{maji_bornomala_tokenizer_2026,
  author  = {Maji, Konko},
  title   = {A Bengali-First, Grapheme-Cluster-Aware Tokenizer with Zero Conjunct Fragmentation},
  year    = {2026},
  note    = {Project Bornomala. Version 0.2},
  url     = {https://github.com/konkomaji/bornomala}
}
```

## License

Apache 2.0. A Project Bornomala release. Founder: Konko Maji
(work.konkomaji@gmail.com).
