# Contributing to Project Bornomala

Thank you for considering a contribution. Bornomala is a Bengali-first language
technology programme built as focused subprojects. Most contributions today land
in the **[MotherTongueIndex](mothertongueindex/)** subproject.

## Ways to help

- **Add a model tokenizer.** The highest-value, lowest-friction contribution.
  Add one row to `mothertongueindex/mti/registry.py` with the correct backend
  and availability tier. See "Adding a model" below.
- **Add or correct a language sample.** Extend `mothertongueindex/data/samples.json`
  with verified parallel text (UDHR Article 1 is the anchor). Cite the source.
- **Improve metrics or the capability model.** Keep derived and measured numbers
  strictly separated.
- **Website, docs, accessibility.** The site is a light Material 3 build under
  `mothertongueindex/web/`.
- **Bug reports and feature requests.** Use the issue templates.

## Ground rules

1. **No fabricated numbers.** Every reported figure is either exact (from a real
   tokenizer) or clearly labelled as an estimate. This mirrors Project Bornomala
   rule E4 and is non-negotiable.
2. **Exact vs estimate must stay visible.** If a value is heuristic, it carries
   the `estimated` flag through to the output.
3. **CPU only for the core.** The `mti` package must not require a GPU. Anything
   that needs a model or an API key belongs in `eval/` or a separate tool.
4. **Do not use em dashes in project content.** Use hyphen, colon, or comma.
5. **Unicode correctness.** Text handling normalises to NFC and operates on
   grapheme clusters (UAX #29), not codepoints.

## Adding a model

```python
# in mothertongueindex/mti/registry.py, add one row to _ROWS:
Model("mymodel", "My Model (Vendor)", "family", "ungated",
      _hf("org/repo-with-tokenizer"), "short note"),
```

- Use `_tk("encoding")` for OpenAI tiktoken encodings, `_hf("org/repo")` for a
  Hugging Face tokenizer, or `_est("label")` for a model with no public
  tokenizer (add a scale entry in `backends.EstimateBackend._MODEL_SCALE`).
- Set `tier` honestly: `ungated`, `gated`, or `estimate`.
- Verify locally: `python -m mti --models mymodel "test text"`.

## Development

```bash
cd mothertongueindex
pip install -e ".[hf,dev]"
python -m mti --list          # sanity check
pytest                        # run tests
ruff check .                  # lint
```

## Commit and PR style

- Small, focused commits. Conventional-style prefixes are welcome
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- Update `mothertongueindex/CHANGELOG.md` under `[Unreleased]`.
- Open a PR against `main`. CI must pass. Fill in the PR template.

## Data ethics

Any human-collected data (speech, dialect, annotation) follows the consent,
compensation, anonymisation, and licensing rules in the parent specification,
section 17. Do not contribute scraped personal data.

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
