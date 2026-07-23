# Changelog: Project Bornomala (main repository)

All notable changes to the Project Bornomala repository are documented here. This
is the repository-level changelog. Each subproject also keeps its own changelog
(for example `mothertongueindex/CHANGELOG.md`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Bornomala identity**: a distinct project logo built around the Bengali
  letter ব (Ba), the first letter of বর্ণমালা, with the matra headline and five
  dialect marks for the West Bengal dialect groups. Plus a matching banner.
  Separate from the MotherTongueIndex subproject logo.
- **Root CHANGELOG** (this file) and the standing rule to document technical
  architecture for every deliverable.
- **Track A: Bengali tokenizer** (`bengali-tokenizer/`): a grapheme-cluster-aware
  Bengali tokenizer that never splits a conjunct (fragmentation rate 0 for
  clusters given an atom). NFC + ZWJ/ZWNJ policy, grapheme-atom BPE/Unigram,
  HarfBuzz shaping validation (Gate G1), full metrics and a typed error
  hierarchy for robustness, CLI, tests, and architecture docs. Includes a
  measured cross-tokenizer Bengali comparison against Sarvam-1, AI4Bharat
  IndicBERT, mBERT, XLM-R, DeepSeek, and GPT-4o. See its own changelog.

- **Publication preparation**: an arXiv preprint (LaTeX source and submission
  guide) and an upload-ready Hugging Face release (model card and tokenizer
  files) for the Bengali tokenizer, version 0.1 (preliminary, to be updated as
  larger datasets are added).

### Changed
- Root README rewritten to separate the two concerns clearly: Project Bornomala
  is the Bengali-first, dialect-aware LLM programme (the whitepaper, tracks A to
  E); MotherTongueIndex is a distinct subproject tool. (PR #2)
- **Bengali tokenizer scaled up**: real literary-weighted training corpus
  (Wikisource, AI4Bharat Sangraha, Wikipedia, XL-Sum news) replaces the
  Wikipedia-only v0.1 demonstrator; new 64k-vocabulary artifact beats
  IndicBERTv2 on fertility, single-token retention, and conjunct fragmentation
  across every register tested (Wikipedia, literary/formal, general web,
  news), not only Wikipedia. See `bengali-tokenizer/CHANGELOG.md` and
  `bengali-tokenizer/docs/known-issues.md` for the full account, including two
  real bugs found and fixed along the way.

## History

- **MotherTongueIndex subproject** added under `mothertongueindex/`: a
  multilingual tokenizer efficiency analyzer, with engine, CLI, website, research
  paper, and full repository hygiene. (PR #1)
- Project Bornomala technical specification (Draft 1.0) published as
  `PROJECT_BORNOMALA_Technical_Specification.md`.

[Unreleased]: https://github.com/konkomaji/bornomala
