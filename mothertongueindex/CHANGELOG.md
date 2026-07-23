# Changelog

All notable changes to MotherTongueIndex are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Website: light Material 3 Expressive site with interactive analyzer, docs,
  and SEO / AEO / GEO metadata. (in progress)
- Technical research paper (`docs/PAPER.md`) and architecture reference
  (`docs/architecture.md`). (in progress)
- GitHub community files: license, contributing guide, code of conduct,
  security policy, citation metadata, issue and PR templates, CI.

## [0.1.0] - 2026-07-23

First working version. Core engine and CLI, running fully on CPU.

### Added
- **Core engine** (`mti/`):
  - `segment.py`: UAX #29 grapheme-cluster segmentation and Unicode script
    detection.
  - `backends.py`: exact tokenization via `tiktoken` (OpenAI) and Hugging Face
    `tokenizers` (open and gated models); labelled heuristic estimate for models
    with no public tokenizer.
  - `registry.py`: catalogue of 28 model tokenizers across OpenAI, open-weight
    frontier, Western gated, multilingual encoders, and Indian models, with
    honest availability tiers (ungated / gated / estimate) and preset groups.
  - `metrics.py`: fertility, single-token retention rate (STRR), bytes per
    token, grapheme clusters per token.
  - `baseline.py`: English anchoring in two modes, a fixed UDHR Article 1
    reference and an exact parallel mode, yielding the `xEN` ratio.
  - `capability.py`: derived reasoning-capability impact from effective-context
    loss, with risk bands.
  - `languages.py`: metadata for 35 languages across scripts and families.
  - `analyze.py`: high-level analysis API.
  - `cli.py`: command line with table, `--why`, `--capability`, `--show`,
    `--json`, `--list`, `--group`.
- **Data** (`data/`): parallel UDHR Article 1 samples in 15 languages and
  `build_tables.py` to generate cross-language by cross-tokenizer tables.
- **Eval** (`eval/reasoning_probe.py`): opt-in measured cross-language reasoning
  probe, designed to run on a separate GPU or API machine.
- **Packaging**: `pyproject.toml` with extras, `mti` console script, clean
  subproject layout.
- **Branding**: symbolic logo (a ring of world scripts around an index dial) and
  banner.

### Notes
- All error rates and counts distinguish exact from estimated values.
- No fabricated numbers (Project Bornomala rule E4).
- Runs on CPU only. Model training is out of scope for this subproject.

[Unreleased]: https://github.com/konkomaji/bornomala/tree/main/mothertongueindex
[0.1.0]: https://github.com/konkomaji/bornomala/tree/main/mothertongueindex
