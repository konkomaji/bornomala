## What this changes

<!-- A short description of the change and why. -->

## Type

- [ ] feat (new capability)
- [ ] fix (bug fix)
- [ ] docs
- [ ] refactor / chore
- [ ] test

## Checklist

- [ ] `pytest` passes and `ruff check .` is clean (in `bengali-tokenizer/`).
- [ ] I updated `bengali-tokenizer/CHANGELOG.md` under `[Unreleased]`.
- [ ] No fabricated numbers; exact vs estimate stays clearly labelled.
- [ ] `bntok` still runs on CPU only (no GPU or external API required).
- [ ] No em dashes in added content.
- [ ] If I changed the tokenizer or corpus, I reran
      `python scripts/compare.py` and updated the affected benchmark numbers
      wherever they are stated (README, paper, website, docs).

## Notes for reviewers

<!-- Anything that needs context, tradeoffs, or follow-ups. -->
