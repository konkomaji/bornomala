# Paper: arXiv preprint

The technical preprint for the Bornomala Bengali tokenizer. Version 0.1
(preliminary): results are on Bengali Wikipedia and will be updated as larger
literary-weighted and dialect datasets are added.

## Files

- `main.tex` : self-contained LaTeX. Compiles with `pdflatex` out of the box (no
  external style files, no native Bengali glyphs, examples in transliteration).
- References are embedded in `main.tex` (`thebibliography`).

## Build locally

```bash
pdflatex main.tex
pdflatex main.tex     # run twice so references resolve
```

Output: `main.pdf`.

## Submit to arXiv

1. Create or log in to an arXiv account (a first submission may need an
   endorsement in cs.CL).
2. Submit the source, not the PDF: upload `main.tex` (arXiv compiles it). You can
   `zip` the paper folder if you add figures later.
3. Primary category: **cs.CL** (Computation and Language). Cross-list optional:
   cs.AI.
4. Title, authors, and abstract are taken from `main.tex`. License: choose
   arXiv's non-exclusive licence, or CC BY 4.0 to match the project.
5. After it is announced you get an arXiv id (for example arXiv:26xx.xxxxx). Put
   that id in `CITATION.cff` and in the Hugging Face model card.

## Update for a venue

For WILDRE, ICON, LREC-COLING, or an ACL venue, reformat `main.tex` with that
venue's official style file. The content maps one to one; only the preamble and
citation style change.

## Reproduce the numbers in the paper

```bash
cd ..
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --skip 15000 --limit 800
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register literary_formal --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register general_web --limit 1000
python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register news --limit 1000
```

## Roadmap (will update the preprint as these land)

- Literary-weighted induction corpus: done (Wikisource + Sangraha, see `docs/known-issues.md` point 6; not confirmed pre-1950 for the Sangraha portion).
- Literary-register, general-web, and news held-out evaluation: done (`bntok.corpus.build_register_held_out`). West Bengal dialect evaluation set: not started.
- Vocabulary-size ablation (32k/48k/64k): done, see `docs/known-issues.md` point 7. BPE vs Unigram ablation: not started.
