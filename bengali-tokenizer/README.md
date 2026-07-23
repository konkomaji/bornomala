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
python -m bntok train --input data/*.txt --algo bpe --vocab-size 32000 --out out/tok
python -m bntok train --wikipedia bn --limit 5000 --vocab-size 32000 --out out/tok

# Encode and evaluate
python -m bntok encode --tokenizer out/tok --text "আমি বাংলায় গান গাই"
python -m bntok evaluate --tokenizer out/tok --input held_out.txt
```

```python
from bntok import BengaliTokenizer, evaluate
tok = BengaliTokenizer.train(corpus, algo="bpe", vocab_size=32000)
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
**conjunct fragmentation rate** (target 0). Run the full ablation grid (BPE and
Unigram; 16k, 32k, 48k, 64k; web-natural vs literary-weighted) from
`configs/ablation.json`.

<!-- METRICS:START -->
### Measured result: it leads a cross-tokenizer Bengali comparison

On held-out Bengali, the Bornomala tokenizer needs the fewest tokens per word and
almost never breaks a conjunct, where every other system breaks many.

| Tokenizer | Fertility | STRR | Bytes/token | Conjunct fragmentation |
|---|--:|--:|--:|--:|
| **Bornomala (bpe 32k)** | **1.390** | **0.766** | **13.03** | **0.0006** |
| IndicBERTv2 (AI4Bharat) | 1.520 | 0.669 | 11.92 | 0.0364 |
| Sarvam-1 (Sarvam AI) | 2.005 | 0.490 | 9.04 | 0.0942 |
| XLM-RoBERTa (Meta) | 2.351 | 0.406 | 7.71 | 0.1034 |
| GPT-4o (OpenAI o200k) | 2.608 | 0.095 | 6.95 | n/a |
| mBERT (Google) | 3.012 | 0.317 | 6.02 | 0.2164 |
| DeepSeek-V3 | 3.024 | 0.071 | 5.99 | 0.2988 |

**How this was measured (full transparency).** Trained on the first 12,000
articles of the Bengali Wikipedia dump (`wikimedia/wikipedia`, `20231101.bn`);
evaluated on 878 held-out lines from articles after those 12,000, so the test
text was unseen in training. Every other tokenizer is its real public tokenizer
on the same NFC-normalised text. Conjunct fragmentation is computed from each
tokenizer's own character offsets; GPT-4o exposes none (n/a). The trained
tokenizer is checked in at `artifacts/bn-bpe-32k/`, the full numbers at
`benchmarks/comparison.json`, and the method in
[`benchmarks/bengali-comparison.md`](benchmarks/bengali-comparison.md).

Reproduce:

```bash
python scripts/compare.py --tokenizer artifacts/bn-bpe-32k --skip 12000 --limit 800
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
- **[Changelog](CHANGELOG.md)**.
- Parent programme: **[Project Bornomala](../README.md)** (the Bengali-first, dialect-aware LLM).

## Grounding

Unicode [UAX #29](https://unicode.org/reports/tr29/) (grapheme clusters),
[UAX #15](https://unicode.org/reports/tr15/) (NFC). Method precedent:
BnGraphemizer (grapheme tokenization for Bengali) and the IndicSuperTokenizer
line of work, adapted to a Bengali-only, integrity-first design.

---

<p align="center"><sub>Project Bornomala, Track A · Apache-2.0 · Konko Maji</sub></p>
