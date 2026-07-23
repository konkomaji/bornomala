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
metrics:
- name: Fertility (tokens per word, held-out Bengali)
  type: fertility
  value: 1.39
- name: Conjunct fragmentation
  type: fragmentation
  value: 0.0006
---

# Bornomala Bengali Tokenizer

A Bengali-first, grapheme-cluster-aware tokenizer that **never splits a conjunct**.
It is the first component of [Project Bornomala](https://github.com/konkomaji/bornomala),
a non-commercial research effort from West Bengal to build a Bengali-first,
dialect-aware language model and to preserve the Bengali language and its dialects.

> **Version 0.1 (preliminary).** Trained on Bengali Wikipedia. It will be updated
> as larger literary-weighted and dialect datasets are added.

## Key features

- **Conjunct integrity by design.** Text is normalised to NFC, segmented into
  Unicode UAX #29 grapheme clusters, and each cluster is remapped to an atomic
  symbol before subword training. Every learned token is a whole number of
  grapheme clusters, so a Bengali conjunct is never split across a boundary.
- **Efficient.** Lowest tokens per word among the systems tested, with a 32k
  vocabulary.
- **Robust.** The whole Bengali Unicode block and ASCII are guaranteed, so any
  Bengali or code-mixed English text round-trips exactly.
- **Open and CPU-only.** Apache 2.0. No GPU needed to train or use.

## Performance

Measured on 878 held-out Bengali Wikipedia lines (unseen during training). Every
other tokenizer is its real public tokenizer on the same NFC-normalised text.

| Tokenizer | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| **Bornomala (this model)** | **1.390** | **0.766** | **13.03** | **0.0006** |
| IndicBERTv2 (AI4Bharat) | 1.520 | 0.669 | 11.92 | 0.0364 |
| Sarvam-1 (Sarvam AI) | 2.005 | 0.490 | 9.04 | 0.0942 |
| XLM-RoBERTa (Meta) | 2.351 | 0.406 | 7.71 | 0.1034 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.095 | 6.95 | n/a |
| mBERT (Google) | 3.012 | 0.317 | 6.02 | 0.2164 |
| DeepSeek-V3 | 3.024 | 0.071 | 5.99 | 0.2988 |

Lower fertility and lower fragmentation are better; higher STRR and bytes per
token are better. Fewer tokens per word means lower cost and more usable context.
Every general tokenizer breaks between 3.6% and 30% of Bengali conjuncts; this one
breaks 0.06%.

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

- Algorithm: BPE, vocabulary 32,000.
- Corpus: first 12,000 articles of `wikimedia/wikipedia`, config `20231101.bn`.
- Normalisation: NFC, with a preserve-by-default ZWJ/ZWNJ policy.
- Hardware: CPU only.

## Limitations

Preliminary results on Wikipedia, which is not weighted toward literary register.
Fragmentation is near zero, not exactly zero, because rare sub-threshold clusters
decompose. See the repository's `docs/known-issues.md` for the full, honest list.

## Citation

```bibtex
@software{maji_bornomala_tokenizer_2026,
  author  = {Maji, Konko},
  title   = {A Bengali-First, Grapheme-Cluster-Aware Tokenizer with Zero Conjunct Fragmentation},
  year    = {2026},
  note    = {Project Bornomala. Version 0.1},
  url     = {https://github.com/konkomaji/bornomala}
}
```

## License

Apache 2.0. A Project Bornomala release. Founder: Konko Maji
(work.konkomaji@gmail.com).
