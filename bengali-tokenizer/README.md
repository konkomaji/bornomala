<p align="center">
  <img src="../assets/logo.svg" width="120" height="120" alt="Project Bornomala"/>
</p>

<h1 align="center">BMBT &nbsp;<sub>Bornomala's Bengali Tokenizer &middot; Project Bornomala, Track A</sub></h1>

<p align="center">
  <b>A Bengali tokenizer that parses the script's own grammar instead of discovering it statistically.</b><br/>
  Finite-state akshara parser, featural decomposition, conjunct fragmentation near-zero by construction.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3A2E8C"/>
  <img alt="track" src="https://img.shields.io/badge/Bornomala-Track%20A-5B45C7"/>
  <img alt="fragmentation" src="https://img.shields.io/badge/conjunct%20fragmentation-~0-00A9A5"/>
  <img alt="compute" src="https://img.shields.io/badge/runs%20on-CPU%20only-F4A400"/>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-E4572E"/>
</p>

---

## Why this exists

Bengali script is an abugida. A written unit (a base consonant with its
conjuncts, reph, phalas, matra, and signs) spans several codepoints but reads as
one symbol. Most tokenizers, including this project's own first version,
*discover* that structure statistically (BPE merges over grapheme clusters).
**BMBT parses it directly**, from Bengali's own generative grammar (the virama
rule, a finite-state machine, not a statistical guess), and only then trains a
statistical layer on top for what the grammar can't explain (loanwords,
code-mixing, noise).

> **Two things BMBT is honest about.** Its measured fertility *ties* the
> previous version, it does not beat it - `docs/design/FORMAL_SPEC.md` proves a
> grammar-constrained BPE cannot beat an unconstrained one on raw token count,
> and that held up in practice: see the measured comparison below. What it adds
> regardless is `featurize()` - a real, tested structural decomposition (onset
> consonants, vowel, modifiers) as an actual output of the tokenizer, not an
> embedding-layer afterthought. **Morphology is not built yet** - this is v2
> roadmap step 5, *partial*.

## Install and use

```bash
pip install -r requirements.txt        # core (CPU only)
# optional: pip install ".[shaping,corpus]"   # HarfBuzz gate + Wikipedia streaming

# Train on your corpus, or reuse the literary-weighted corpus config
python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --out out/bmbt   # recommended
python -m bntok bmbt-train --input data/*.txt --algo bpe --vocab-size 64000 --out out/bmbt

# Encode, evaluate, and inspect the featural decomposition
python -m bntok bmbt-encode --tokenizer out/bmbt --text "আমি বাংলায় গান গাই"
python -m bntok bmbt-evaluate --tokenizer out/bmbt --input held_out.txt
python -m bntok bmbt-featurize --text "স্ত্রী ক্ষ্ম আকাঙ্ক্ষা"
```

```python
from bntok import BMBT, featurize

tok = BMBT.train(corpus, algo="bpe", vocab_size=64000)
ids = tok.encode("আমি বাংলায় ক্ষুদ্র গান গাই")
assert tok.decode(ids) == "আমি বাংলায় ক্ষুদ্র গান গাই"   # exact round-trip
tok.save("out/bmbt")

for f in featurize("স্ত্রী"):        # no training needed - pure grammar
    print(f.onset, f.vowel, f.modifiers)   # ['স', 'ত', 'র'] ী []
```

## What it guarantees

| Property | How |
|---|---|
| Conjunct integrity | Subword model trains over akshara atoms (whole conjunct chains, parsed by grammar), never codepoints. Fragmentation near-zero, same order as v1. |
| Round-trip fidelity | Full Bengali block and ASCII are guaranteed atoms and forced into the vocabulary. Any Bengali or code-mixed text decodes back exactly. |
| Correct normalisation | NFC before anything; documented ZWJ / ZWNJ policy (reused unchanged from v1). |
| Featural output | `featurize()`: onset/vowel/modifier per akshara, lossless (reconstructs the original text exactly), needs no trained model. |
| No silent failure | Typed error hierarchy; every entry point validates inputs. |
| Isolation from v1 | `bmbt.py` imports nothing from `atoms.py`/`tokenizer.py` - a change to either can never silently affect the other. |

## Measured result: ties v1, adds featural structure

<!-- METRICS:START -->
Trained on the identical literary-weighted corpus as v1 (`configs/bpe-64k.json`, same 64,000 vocabulary), measured on the same four held-out registers:

| Register | Fertility (v1 / BMBT) | STRR (v1 / BMBT) | Conjunct frag. (v1 / BMBT) |
|---|--:|--:|--:|
| Wikipedia | 1.524 / 1.524 | 0.722 / 0.722 | 0.000075 / 0.000075 |
| Literary/formal | 1.320 / 1.320 | 0.789 / 0.789 | 0.000104 / 0.000112 |
| General web | 1.201 / 1.201 | 0.861 / 0.861 | 0.000055 / 0.000057 |
| News | 1.140 / 1.140 | 0.893 / 0.894 | 0.000025 / 0.000025 |

On Wikipedia the two are identical down to the raw integer counts, despite genuinely different vocabularies (12,233 atoms for v1, 12,199 for BMBT). On the larger registers, tiny real differences appear in both directions - BMBT needs marginally fewer tokens, has marginally more fragmented clusters - neither large enough to call a win. **This is an honest tie**, reported exactly as measured. Both still lead every external baseline tested (IndicBERTv2, SUTRA, Sarvam-1, XLM-RoBERTa, GPT-4o, mBERT, DeepSeek-V3, Krutrim) by a wide margin on every register.

Full account, the CC-100 ablation, and why the tie makes sense given `FORMAL_SPEC.md`'s own proof: [`docs/known-issues.md`](docs/known-issues.md) ("Roadmap: a proposed v2") and [`benchmarks/bengali-comparison.md`](benchmarks/bengali-comparison.md).

Reproduce:
```bash
python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --out artifacts/bmbt-64k
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --bmbt-tokenizer artifacts/bmbt-64k --skip 15000 --limit 800
```
<!-- METRICS:END -->

## Hard words: conjuncts and Bengali place names

A register average can hide how a tokenizer treats specific, culturally
load-bearing words. A fixed list of 13 - deity names, a national poet,
well-known West Bengal places, all conjunct-dense - measured on every
tokenizer this repository tracks:

**Ours (v1 and BMBT) tokenizes every one of the 13 words as exactly one
token.** No exception, including the triple-conjunct আকাঙ্ক্ষা and the
multi-akshara রবীন্দ্রনাথ.

| Word | Meaning | Ours (v1/BMBT) | IndicBERTv2 (best rival) | GPT-4o |
|---|---|--:|--:|--:|
| স্ত্রী | wife/woman | 1 | 1 | 2 |
| আকাঙ্ক্ষা | aspiration | 1 | 1 | 6 |
| রবীন্দ্রনাথ | Rabindranath (Tagore) | 1 | 1 | 7 |
| পশ্চিমবঙ্গ | West Bengal | 1 | 1 | 5 |
| বিষ্ণুপুর | Bishnupur | 1 | 2 | 5 |
| শান্তিনিকেতন | Santiniketan | 1 | 3 | 5 |

Average tokens/word over all 13 words, all 16 tokenizers measured (ours,
IndicBERTv2, SUTRA, Sarvam-1, Param2-17B, XLM-RoBERTa, mBERT, GPT-4o,
DeepSeek-V3, Krutrim, Qwen2.5, GPT-4 cl100k, Llama-3.1, Mistral-7B; Gemma-2
gated, reports unavailable): **ours 1.00**, IndicBERTv2 (the only real
rival) 1.31, the rest 3.31-11.08. IndicBERTv2 still fragments 3 of the 13
words; ours never does, by construction. Full per-word, per-tokenizer
table and reproduce command: [`benchmarks/hard-words.md`](benchmarks/hard-words.md).

## `bn-bpe-64k` (v1, previous, stable, unchanged)

The project's first tokenizer remains fully available and untouched: grapheme-cluster-aware BPE/Unigram, conjunct fragmentation 0 by construction, the artifact behind the published Hugging Face model (`konko/bornomala-bengali-tokenizer`).

```bash
python -m bntok train --corpus-config configs/bpe-64k.json --out out/tok
python -m bntok encode --tokenizer out/tok --text "আমি বাংলায় গান গাই"
```

```python
from bntok import BengaliTokenizer
tok = BengaliTokenizer.train(corpus, algo="bpe", vocab_size=64000)
```

Full v1 documentation: [`docs/architecture.md`](docs/architecture.md).

## Layout

```
bengali-tokenizer/
├── bntok/
│   ├── normalize.py    NFC + ZWJ/ZWNJ policy (shared by v1 and BMBT)
│   ├── substrate.py    Bengali Unicode inventory (consonants/vowels/matras/modifiers)
│   ├── akshara.py      finite-state akshara grammar parser (BMBT's segmentation)
│   ├── bmbt.py         BMBT: akshara atoms, train/encode/decode, featurize()
│   ├── graphemes.py    UAX #29 clusters + Bengali structure (v1's segmentation)
│   ├── atoms.py        cluster <-> atom map (v1's integrity mechanism)
│   ├── tokenizer.py    BengaliTokenizer: v1's train/encode/decode/save/load
│   ├── evaluate.py     fertility, STRR, fragmentation, round-trip (shared)
│   ├── shaping.py      HarfBuzz Gate G1
│   ├── corpus.py       robust corpus loading, Wikipedia/Sangraha/CC-100 streaming
│   ├── errors.py       typed error hierarchy
│   └── cli.py          python -m bntok
├── configs/            training + ablation configs
├── docs/bmbt-architecture.md, architecture.md
└── tests/
```

## Documentation

- **[BMBT architecture](docs/bmbt-architecture.md)**: pipeline, featurize(), the isolation-from-v1 design.
- **[v1 architecture](docs/architecture.md)**: pipeline, diagrams, the integrity proof.
- **[Known issues and limitations](docs/known-issues.md)**: honest caveats, comparison notes, and the bugs found and fixed during development.
- **[Benchmark method and results](benchmarks/bengali-comparison.md)**, and the [hard-words showcase](benchmarks/hard-words.md) (conjuncts and Bengali place names, every tokenizer tracked).
- **[Paper](paper/)**: the arXiv preprint source (LaTeX) and submission guide.
- **v2 design docs:** [Reading Bengali on Its Own Terms](docs/design/reading-bengali-on-its-own-terms.md) (the position paper BMBT implements) and its [formal specification](docs/design/FORMAL_SPEC.md) (losslessness, totality, linear time, constrained optimality as proofs and a fuzzer contract).
- **[Hugging Face release](huggingface/)**: upload-ready model card and tokenizer files (v1; BMBT not yet published there).
- **[Changelog](CHANGELOG.md)**.
- Parent programme: **[Project Bornomala](../README.md)** (the Bengali-first, dialect-aware LLM).

## Grounding

Unicode [UAX #29](https://unicode.org/reports/tr29/) (grapheme clusters),
[UAX #15](https://unicode.org/reports/tr15/) (NFC), the Unicode Bengali block's
virama/Indic_Conjunct_Break rules (BMBT's grammar). Method precedent:
BnGraphemizer (grapheme tokenization for Bengali) and the IndicSuperTokenizer
line of work, adapted to a Bengali-only, integrity-first design.

---

<p align="center"><sub>Project Bornomala, Track A · Apache-2.0 · Konko Maji</sub></p>
