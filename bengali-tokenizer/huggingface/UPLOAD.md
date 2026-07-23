# Publishing to Hugging Face

This folder is an upload-ready Hugging Face model repository: the tokenizer files
plus the model card (`README.md`). Publishing needs your own Hugging Face account
and token; it cannot be done on your behalf.

## Files in this folder

- `README.md` : the model card (with YAML frontmatter).
- `tokenizer.json` : the atom-space subword model.
- `atoms.json` : the grapheme-cluster to atom map.
- `bornomala_config.json` : the tokenizer config (algorithm, vocab size, policy).
  Rename to `config.json` after download when loading with `bntok`.

## One-time setup

```bash
pip install huggingface_hub
huggingface-cli login        # paste a write token from https://huggingface.co/settings/tokens
```

## Create and upload

```bash
# create the repo (model type)
huggingface-cli repo create bornomala-bengali-tokenizer --type model

# upload everything in this folder to the repo root
huggingface-cli upload konkomaji/bornomala-bengali-tokenizer . . --repo-type model
```

Your model then lives at
`https://huggingface.co/konkomaji/bornomala-bengali-tokenizer`.

## After the arXiv preprint is announced

Add the arXiv id to the model card (a line like
`arxiv: 26xx.xxxxx` in the frontmatter, and a link in the body) and to
`CITATION.cff`, then re-upload the `README.md`.

## Notes

- Keep the license as `apache-2.0` to match the repository.
- The three tokenizer files must sit in one directory to load with
  `BengaliTokenizer.load`; on the Hub they are at the repo root, so after
  `snapshot_download` copy or rename `bornomala_config.json` to `config.json`.
