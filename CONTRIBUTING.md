# Contributing to Project Bornomala

Thank you for considering a contribution. Bornomala is a Bengali-first language
technology research programme. The tokenizer in **[`bengali-tokenizer/`](bengali-tokenizer/)**
is the current deliverable.

## Ways to help

- **Improve the tokenizer.** The grapheme-atom pipeline, corpus sourcing, or
  the evaluation/comparison scripts under `bengali-tokenizer/`.
- **Add a source to the comparison.** Extend `scripts/compare.py` with another
  real, public tokenizer, run on the same held-out text.
- **Website, docs, accessibility.** The project site is a light Material 3
  build under `docs/`.
- **Bug reports and feature requests.** Use the issue templates.
- **Native Bengali speakers and West Bengal dialect speakers, linguists, and
  archives** are also needed for the programme's broader roadmap; see the
  [website](https://konkomaji.github.io/bornomala/) and root README.

## Ground rules

1. **No fabricated numbers.** Every reported figure is either exact (measured
   against a real tokenizer, on a stated held-out set) or clearly labelled as
   an estimate. This is Bornomala rule E4 and is non-negotiable.
2. **CPU only for the core.** `bntok` must not require a GPU.
3. **Do not use em dashes in project content.** Use hyphen, colon, or comma.
4. **Unicode correctness.** Text handling normalises to NFC and operates on
   grapheme clusters (UAX #29), not codepoints; no token may split a
   grapheme cluster.

## Development

```bash
cd bengali-tokenizer
pip install -e ".[dev]"
python -m bntok gate-g1        # sanity check: Bengali shapes correctly
pytest                         # run tests
ruff check bntok scripts tests # lint
```

## Commit and PR style

- Small, focused commits. Conventional-style prefixes are welcome
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- Update `bengali-tokenizer/CHANGELOG.md` under `[Unreleased]`.
- If a change affects the tokenizer or corpus, rerun
  `python scripts/compare.py` and update every place the numbers are stated.
- Open a PR against `main`. CI must pass. Fill in the PR template.

## Data ethics

Any human-collected data (speech, dialect, annotation) follows the consent,
compensation, anonymisation, and licensing rules in the parent specification,
section 16. Do not contribute scraped personal data.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## MotherTongueIndex

The multilingual tokenizer-efficiency analyzer that used to live in this
repository (`mothertongueindex/`) is now its own project:
**[github.com/konkomaji/mothertongueindex](https://github.com/konkomaji/mothertongueindex)**.
Contribute there for anything related to it.
