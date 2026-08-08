# Track A2: corpus dedup + quality-filter survival ratio (Gate G3)

Spec section 16.2, Gate G3, month 6: "After dedup and quality filtering, what
fraction of raw Bengali web text survives? Is the surviving corpus ≥5B clean
tokens?" Section 16.3's "First 30 days" plan says exactly how to answer it:
"Download Bengali Wikipedia and a Sangraha Bengali shard. Run dedup + quality
filtering. Measure the survival ratio on real data." This is that measurement.

## Pipeline

`bntok/dedup.py`, three stages, each independently tested
(`tests/test_dedup.py`):

1. **Exact dedup** - set-based, removes byte-identical repeated lines.
2. **Near dedup** - MinHash LSH (`datasketch`) over word 5-gram shingles,
   Jaccard threshold 0.8, catches near-identical lines exact dedup misses.
3. **Quality filter** - `is_clean_bengali_line` (Bengali/ASCII character
   ratio ≥ 0.75, min length 4) plus rejection of digit-dominated and
   repeated-character-dominated lines (page numbers, table dumps, rule
   lines).

**No LM-perplexity stage.** The `kenlm` PyPI wheel is query-only - it has
no `lmplz` trainer, so there is no way to train a Bengali ARPA model from
this environment without building kenlm from source (not attempted this
round). Rule-based filtering alone is not a placeholder: Sangraha and CCNet
both use non-LM rule filters as one of their stages, not only LM
perplexity.

## Measurement

Two real sources, streamed via the existing `bntok.corpus` loaders,
measured separately and pooled (`scripts/corpus_survival.py`):

- **Bengali Wikipedia**, 3,000 articles (`stream_wikipedia`).
- **Sangraha web-typed, raw** (`stream_sangraha(doc_type="web", clean=False)`),
  10,000 documents - unlike the tokenizer's own training path, no
  `is_clean_bengali_line` pre-filter applied before this pipeline runs.

| Source | Raw lines | Raw words | Survival (lines) | Survival (words) |
|---|--:|--:|--:|--:|
| Wikipedia | 159,136 | 2,718,289 | 76.3% | 96.9% |
| Sangraha web (raw) | 86,788 | 3,084,853 | 96.6% | 98.7% |
| **Pooled** | **245,924** | **5,803,142** | **83.5%** | **97.9%** |

Full breakdown by removal stage: `track-a2-corpus-survival.json` (this
directory). Reproduce: `python scripts/corpus_survival.py` from
`bengali-tokenizer/`.

## Reading the numbers honestly

- **Word-level survival (the token proxy) is high on both sources: 96.9%
  and 98.7%.** This is well above what a raw-CommonCrawl pipeline (CCNet,
  RefinedWeb) typically reports (often 30-50% survival after full dedup +
  LM filtering) - because neither source measured here is genuinely raw,
  unfiltered web crawl. Sangraha's "verified" tier (what this project
  already trains on) has already been through AI4Bharat's own quality
  pipeline before publication; Wikipedia is edited encyclopedic prose.
  Calling the Sangraha row "raw" is accurate relative to this project's
  own training path (`clean=False`, no `is_clean_bengali_line` applied),
  not relative to the open web. **CC-100 (2018 CommonCrawl-derived,
  already documented elsewhere in this repo as noisier and non-literary)
  would be a stronger true-raw-web proxy for a follow-up measurement**,
  not run this round.
- **Line-level survival is much lower for Wikipedia (76.3%) than word-level
  (96.9%).** Not a contradiction: Wikipedia articles repeat short
  boilerplate section headers across thousands of pages (references,
  external-links, see-also), which exact-dedup correctly removes as
  duplicate lines while barely touching total word count.
- **The absolute "≥5B clean tokens" threshold is not directly answered by
  this measurement.** This is a 245,924-line sample (5.8M raw words), not
  the full available corpus. `bntok/corpus.py`'s own `stream_indiccorp_v2`
  docstring cites AI4Bharat's published figure of 30.0B tokens for
  IndicCorp v2 alone; applying this measurement's pooled 97.9% word
  survival ratio to that figure would project roughly 29B surviving
  tokens, comfortably above 5B - **but that is an extrapolation from a
  sample of different, more-curated sources, not a verified measurement
  of IndicCorp v2 or CC-100 specifically**, and is reported as such, not
  as a confirmed number.

## Gate G3 verdict

**Provisional pass, not a full-scale confirmed pass.** The measured
survival ratio on real data (83.5% lines / 97.9% words, pooled) is far
above the 5-10% floor the gate's "if NO" clause treats as a crisis, on the
two sources the spec's own 30-day plan names. The open item is scale, not
direction: running this same pipeline against IndicCorp v2 and/or CC-100
(the two bulk, noisier sources already wired into `bntok/corpus.py` but
not included in this pass) would turn the extrapolated ~29B-token estimate
into a measured one. Until then, this does not trigger the gate's "if NO"
clause (re-weight the whole programme toward Track B / OCR), but should
not be cited as a fully closed gate either.

## What's still open

- Full-scale run against IndicCorp v2 and/or CC-100 (bulk, noisier
  sources) to convert the extrapolated ~5B+ token estimate into a
  measured one.
- LM-perplexity filtering remains unavailable without building `kenlm`
  from source (no `lmplz` in the PyPI wheel); not blocking, since the
  rule-based pipeline alone already answers the gate's core question.
- This pipeline has not yet been wired into `build_configured_corpus`
  itself - it is a standalone measurement tool (`bntok/dedup.py` +
  `scripts/corpus_survival.py`), not yet applied to the tokenizer's actual
  training corpus. Whether to do so is a separate decision: the shipped
  tokenizer's own corpus already applies `is_clean_bengali_line` at load
  time for OCR-derived sources, so some of this overlaps existing
  practice; exact-dedup and near-dedup are the genuinely new capability.
