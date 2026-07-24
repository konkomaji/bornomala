<p align="center">
  <img src="../assets/logo.svg" width="120" height="120" alt="Project Bornomala"/>
</p>

<h1 align="center">Bengali Tokenizer &nbsp;<sub>Project Bornomala, Track A</sub></h1>

<p align="center">
  <b>A Bengali-first tokenizer that never splits a conjunct.</b><br/>
  Grapheme-cluster aware, NFC-normalised, modern subword training, conjunct fragmentation rate 0 by construction.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3A2E8C"/>
  <img alt="track" src="https://img.shields.io/badge/Bornomala-Track%20A-5B45C7"/>
  <img alt="fragmentation" src="https://img.shields.io/badge/conjunct%20fragmentation-0-00A9A5"/>
  <img alt="compute" src="https://img.shields.io/badge/runs%20on-CPU%20only-F4A400"/>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-E4572E"/>
</p>

---

## Why this exists

Bengali script is an abugida. A written unit (a base consonant with its
conjuncts, reph, phalas, matra, and signs) spans several codepoints but reads as
one symbol. Character-level BPE, the default in nearly every LLM tokenizer, can
place a token boundary inside such a unit, producing tokens that correspond to
nothing a reader recognises. This tokenizer makes that impossible.

> **The guarantee.** Every learned token is a whole number of grapheme clusters.
> The conjunct fragmentation rate is **0 by construction**, not by tuning.

It does this the modern, robust way: normalise to NFC, segment into UAX #29
grapheme clusters, remap each cluster to an atomic symbol, then train BPE or
Unigram over atoms. Merges combine whole clusters, so no token can split one.

## Install and use

```bash
pip install -r requirements.txt        # core (CPU only)
# optional: pip install ".[shaping,corpus]"   # HarfBuzz gate + Wikipedia streaming

# Gate G1: verify Bengali shapes correctly before trusting the pipeline
python -m bntok gate-g1

# Train on your corpus, or stream Bengali Wikipedia
python -m bntok train --input data/*.txt --algo bpe --vocab-size 64000 --out out/tok
python -m bntok train --wikipedia bn --limit 5000 --vocab-size 64000 --out out/tok
python -m bntok train --corpus-config configs/bpe-64k.json --out out/tok   # recommended: literary-weighted corpus

# Encode and evaluate
python -m bntok encode --tokenizer out/tok --text "আমি বাংলায় গান গাই"
python -m bntok evaluate --tokenizer out/tok --input held_out.txt
```

```python
from bntok import BengaliTokenizer, evaluate
tok = BengaliTokenizer.train(corpus, algo="bpe", vocab_size=64000)
ids = tok.encode("আমি বাংলায় ক্ষুদ্র গান গাই")
assert tok.decode(ids) == "আমি বাংলায় ক্ষুদ্র গান গাই"   # exact round-trip
tok.save("out/tok")
```

## What it guarantees

| Property | How |
|---|---|
| Conjunct integrity | Subword model trains over grapheme-cluster atoms, never codepoints. Fragmentation rate 0. |
| Round-trip fidelity | Full Bengali block and ASCII are guaranteed atoms and forced into the vocabulary. Any Bengali or code-mixed text decodes back exactly. |
| Correct normalisation | NFC before anything (requirement B-1); documented ZWJ / ZWNJ policy. |
| Correct shaping | HarfBuzz round-trip validation (Requirement A-1, Gate G1). |
| No silent failure | Typed error hierarchy; every entry point validates inputs (mitigates risk R1). |

## Metrics

Reported per the whitepaper (spec section 9.2 step 4): **fertility**,
**STRR**, **bytes/token**, **grapheme-clusters/token**, and the headline
**conjunct fragmentation rate** (target 0). An ablation across 16k/32k/48k/64k
vocab sizes on the literary-weighted corpus showed fertility recovering
monotonically with vocab size; 64k is the current recommendation and the
shipped artifact (`configs/bpe-64k.json`).

<!-- METRICS:START -->
### Measured result: it leads a cross-tokenizer Bengali comparison

On held-out Bengali, the Bornomala tokenizer needs the fewest tokens per word and
almost never breaks a conjunct, where every other system breaks many. This holds
across every register tested (Wikipedia, literary/formal, general web, news),
not only the Wikipedia table shown below.

| Tokenizer | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| **Bornomala (bpe 64k)** | **1.524** | **0.722** | **11.38** | **0.0001** |
| IndicBERTv2 (AI4Bharat) | 1.652 | 0.612 | 10.50 | 0.0440 |
| SUTRA (TWO AI) | 2.218 | 0.419 | 7.82 | 0.1579 |
| XLM-RoBERTa (Meta) | 2.464 | 0.363 | 7.04 | 0.1019 |
| Sarvam-1 (Sarvam AI) | 2.593 | 0.415 | 6.69 | 0.1191 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.111 | 6.65 | n/a |
| mBERT (Google) | 2.777 | 0.385 | 6.25 | 0.1800 |
| DeepSeek-V3 | 2.994 | 0.089 | 5.79 | 0.2845 |
| Krutrim (Krutrim AI) | 3.207 | 0.076 | 5.41 | 0.2859 |

**How this was measured (full transparency).** Trained on a literary-weighted
corpus (Wikisource, Sangraha verified/ben pdf- and web-typed, the first 15,000
Bengali Wikipedia articles, XL-Sum Bengali news; full mix in
[`docs/known-issues.md`](docs/known-issues.md) point 6); evaluated on 828
held-out lines from Wikipedia articles after those 15,000, unseen in training.
Every other tokenizer is its real public tokenizer on the same NFC-normalised
text. Conjunct fragmentation is computed from each tokenizer's own character
offsets; GPT-4o exposes none (n/a). The trained tokenizer is checked in at
`artifacts/bn-bpe-64k/`, the full numbers at `benchmarks/comparison.json`, and
the method plus every-register breakdown in
[`benchmarks/bengali-comparison.md`](benchmarks/bengali-comparison.md).

Reproduce:

```bash
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --skip 15000 --limit 800
```
<!-- METRICS:END -->

## Layout

```
bengali-tokenizer/
├── bntok/
│   ├── normalize.py    NFC + ZWJ/ZWNJ policy
│   ├── graphemes.py    UAX #29 clusters + Bengali structure
│   ├── atoms.py        cluster <-> atom map (the integrity mechanism)
│   ├── tokenizer.py    BengaliTokenizer: train/encode/decode/save/load
│   ├── evaluate.py     fertility, STRR, fragmentation, round-trip
│   ├── shaping.py      HarfBuzz Gate G1
│   ├── corpus.py       robust corpus loading + Wikipedia streaming
│   ├── errors.py       typed error hierarchy
│   └── cli.py          python -m bntok
├── configs/            training + ablation configs
├── docs/architecture.md
└── tests/
```

## Documentation

- **[Architecture](docs/architecture.md)**: pipeline, diagrams, the integrity proof.
- **[Known issues and limitations](docs/known-issues.md)**: honest caveats, comparison notes, and the bugs found and fixed during development.
- **[Benchmark method and results](benchmarks/bengali-comparison.md)**.
- **[Paper](paper/)**: the arXiv preprint source (LaTeX) and submission guide.
- **v2 design (not yet built):** [Reading Bengali on Its Own Terms](docs/design/reading-bengali-on-its-own-terms.md), a grammar-first, featural akshara tokenizer proposal that replaces BPE-as-primary with a finite-state akshara parser and demotes statistics to a fallback, plus its [formal specification](docs/design/FORMAL_SPEC.md) (losslessness, totality, linear time, constrained optimality as proofs and a fuzzer contract). The shipped `bn-bpe-64k` tokenizer above is v1; this is the position paper for v2.
- **[Hugging Face release](huggingface/)**: upload-ready model card and tokenizer files.
- **[Changelog](CHANGELOG.md)**.
- Parent programme: **[Project Bornomala](../README.md)** (the Bengali-first, dialect-aware LLM).

## Grounding

Unicode [UAX #29](https://unicode.org/reports/tr29/) (grapheme clusters),
[UAX #15](https://unicode.org/reports/tr15/) (NFC). Method precedent:
BnGraphemizer (grapheme tokenization for Bengali) and the IndicSuperTokenizer
line of work, adapted to a Bengali-only, integrity-first design.

---

<p align="center"><sub>Project Bornomala, Track A · Apache-2.0 · Konko Maji</sub></p>
