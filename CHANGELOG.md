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
- **Track A: Bengali tokenizer** (`bengali-tokenizer/`): the state-of-the-art,
  grapheme-cluster-aware Bengali tokenizer training system. (in progress)

### Changed
- Root README rewritten to separate the two concerns clearly: Project Bornomala
  is the Bengali-first, dialect-aware LLM programme (the whitepaper, tracks A to
  E); MotherTongueIndex is a distinct subproject tool. (PR #2)

## History

- **MotherTongueIndex subproject** added under `mothertongueindex/`: a
  multilingual tokenizer efficiency analyzer, with engine, CLI, website, research
  paper, and full repository hygiene. (PR #1)
- Project Bornomala technical specification (Draft 1.0) published as
  `PROJECT_BORNOMALA_Technical_Specification.md`.

[Unreleased]: https://github.com/konkomaji/bornomala
