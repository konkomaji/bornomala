## What this changes

<!-- A short description of the change and why. -->

## Type

- [ ] feat (new capability)
- [ ] fix (bug fix)
- [ ] docs
- [ ] refactor / chore
- [ ] test

## Checklist

- [ ] `pytest` passes and `ruff check .` is clean (in `mothertongueindex/`).
- [ ] I updated `mothertongueindex/CHANGELOG.md` under `[Unreleased]`.
- [ ] No fabricated numbers; exact vs estimate stays clearly labelled.
- [ ] The core `mti` package still runs on CPU with no GPU or API key.
- [ ] No em dashes in added content.
- [ ] If I added a model, I set an honest availability tier and verified it
      locally with `python -m mti --models <id> "test"`.

## Notes for reviewers

<!-- Anything that needs context, tradeoffs, or follow-ups. -->
