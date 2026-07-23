# Changelog: Track A Bengali tokenizer (`bntok`)

All notable changes to the Track A tokenizer are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-23

First working version of the Project Bornomala Track A tokenizer. CPU only.

### Added
- **Grapheme-cluster-aware core** that never splits a conjunct across a token
  boundary (conjunct fragmentation rate = 0 by construction):
  - `normalize.py`: NFC normalisation (UAX #15) and a documented, preserve-by-
    default ZWJ / ZWNJ policy, with canonical khanda-ta handling.
  - `graphemes.py`: UAX #29 grapheme-cluster segmentation and Bengali structural
    predicates (conjunct, reph, ya-phala, ra-phala, nukta), plus guaranteed
    coverage sets (Bengali block and ASCII).
  - `atoms.py`: a reversible grapheme-cluster to Private-Use-Area atom map with a
    codepoint decomposition fallback for rare or unseen clusters.
  - `tokenizer.py`: `BengaliTokenizer`, training BPE or Unigram over atoms with a
    Metaspace word-boundary marker; encode, decode, save, load, and a round-trip
    self-check. Every atom is forced into the base vocabulary so covered text
    always round-trips.
- **Evaluation** (`evaluate.py`): fertility, STRR, bytes per token, grapheme
  clusters per token, conjunct fragmentation rate, and round-trip fidelity.
- **Shaping validation** (`shaping.py`): HarfBuzz coverage and cluster-
  correspondence check (Requirement A-1, Gate G1), with system-font auto-detect.
- **Corpus loading** (`corpus.py`): local files and directories, optional Bengali
  Wikipedia streaming, and a literary-weighted source combiner.
- **CLI** (`python -m bntok`): `gate-g1`, `train`, `evaluate`, `encode`.
- **Robustness**: a typed error hierarchy (`errors.py`); every entry point
  validates inputs and fails with a clear message, never a silent corruption or a
  raw traceback (mitigates risk R1).
- **Configs** for the whitepaper ablation grid (BPE and Unigram; 16k, 32k, 48k,
  64k; web-natural vs literary-weighted).
- **Tests**: grapheme integrity, zero fragmentation, round-trip (Bengali and
  code-mixed), save/load, and the error paths.
- **Benchmark**: a measured cross-tokenizer Bengali comparison
  (`scripts/compare.py`, `benchmarks/`) against AI4Bharat IndicBERTv2, Sarvam-1,
  XLM-RoBERTa, GPT-4o, mBERT, and DeepSeek-V3 on held-out Bengali. The Bornomala
  tokenizer leads on fertility (1.39), single-token retention, bytes/token, and
  conjunct fragmentation (0.0006 vs 4 to 30 percent for the others).
- **Artifact**: a trained BPE 32k tokenizer checked in at `artifacts/bn-bpe-32k/`.
- **Docs**: architecture reference with diagrams (`docs/architecture.md`).

### Notes
- Induction is CPU-bound and runs anywhere. Large corpus assembly and the full
  ablation grid run on the training machine.
- This is the tokenizer itself (Track A). It is distinct from the
  MotherTongueIndex subproject, which benchmarks tokenizers.

[0.1.0]: https://github.com/konkomaji/bornomala/tree/main/bengali-tokenizer
