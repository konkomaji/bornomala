# Changelog: Project Bornomala (main repository)

All notable changes to the Project Bornomala repository are documented here. This
is the repository-level changelog. The `bengali-tokenizer/` subproject also
keeps its own changelog (`bengali-tokenizer/CHANGELOG.md`).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **BMBT (Bornomala's Bengali Tokenizer)**: the v2 tokenizer, now built and
  the recommended/primary tokenizer of the project. Parses Bengali's akshara
  grammar directly (a finite-state machine) instead of discovering structure
  statistically, and adds a real featural decomposition (`featurize()`) as
  an actual tokenizer output, plus a morphology layer that aligns token
  boundaries to Bengali's suffix structure. Measured against
  the original tokenizer (`bn-bpe-64k`) on identical held-out text: an
  honest tie, not a win - reported exactly as measured, matching the design's
  own formal proof that a grammar-constrained BPE cannot beat an
  unconstrained one on raw token count. `bn-bpe-64k` remains fully available
  and unchanged. Full detail: `bengali-tokenizer/CHANGELOG.md`,
  `bengali-tokenizer/docs/known-issues.md`,
  `bengali-tokenizer/benchmarks/bengali-comparison.md`.
- **Bornomala identity**: a distinct project logo built around the Bengali
  letter ব (Ba), the first letter of বর্ণমালা, with the matra headline and five
  dialect marks for the West Bengal dialect groups. Plus a matching banner.
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

- **Track A v2 design record**: a literature-grounded position paper and
  companion formal specification proposing a grammar-first, featural akshara
  tokenizer to eventually replace BPE-as-primary in the Bengali tokenizer.
  Not yet built; see `bengali-tokenizer/docs/design/`. Root README and project
  website updated with a "what's next" section pointing to it.

### Removed
- **Subproject tool extracted to its own repository.** It grew out of this
  programme but was never the tokenizer itself, and is now maintained
  separately. Its directory removed from this repo; CI workflow, README,
  project website, CONTRIBUTING, and SECURITY updated accordingly. History up
  to the extraction remains in this repo's git log.

### Changed
- Root README rewritten to focus solely on Project Bornomala: the
  Bengali-first, dialect-aware LLM programme (the whitepaper, tracks A to E). (PR #2)
- **Bengali tokenizer scaled up**: real literary-weighted training corpus
  (Wikisource, AI4Bharat Sangraha, Wikipedia, XL-Sum news) replaces the
  Wikipedia-only v0.1 demonstrator; new 64k-vocabulary artifact beats
  IndicBERTv2 on fertility, single-token retention, and conjunct fragmentation
  across every register tested (Wikipedia, literary/formal, general web,
  news), not only Wikipedia. See `bengali-tokenizer/CHANGELOG.md` and
  `bengali-tokenizer/docs/known-issues.md` for the full account, including two
  real bugs found and fixed along the way.

## History

- Project Bornomala technical specification (Draft 1.0) published as
  `PROJECT_BORNOMALA_Technical_Specification.md`.

[Unreleased]: https://github.com/konkomaji/bornomala
