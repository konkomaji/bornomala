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

Three real sources, streamed via the existing `bntok.corpus` loaders,
measured separately and pooled (`scripts/corpus_survival.py`):

- **Bengali Wikipedia**, 3,000 articles (`stream_wikipedia`).
- **Sangraha web-typed, raw** (`stream_sangraha(doc_type="web", clean=False)`),
  10,000 documents - unlike the tokenizer's own training path, no
  `is_clean_bengali_line` pre-filter applied before this pipeline runs.
- **AI4Bharat IndicCorp v2** (`stream_indiccorp_v2`), 2,000,000 lines - a
  large-scale follow-up pass (`--indiccorp-limit`), added specifically to
  replace an earlier extrapolation with a real measurement on the actual
  bulk source Gate G3's ≥5B-token threshold is about, not a proxy for it.

| Source | Raw lines | Raw words | Survival (lines) | Survival (words) |
|---|--:|--:|--:|--:|
| Wikipedia | 159,136 | 2,718,289 | 76.3% | 96.9% |
| Sangraha web (raw) | 86,788 | 3,084,853 | 96.6% | 98.7% |
| Pooled (wiki + sangraha) | 245,924 | 5,803,142 | 83.5% | 97.9% |
| **IndicCorp v2** | **2,000,000** | **90,590,866** | **97.9%** | **99.3%** |
| **Pooled, all three** | **2,245,924** | **96,394,008** | **96.1%** | **99.0%** |

Full breakdown by removal stage: `track-a2-corpus-survival.json` (this
directory). Reproduce: `python scripts/corpus_survival.py --indiccorp-limit 2000000`
from `bengali-tokenizer/` (near-dedup on 2M lines took roughly 90-100
minutes on this machine, single-threaded pure-Python MinHash - the
2,000,000-line default is 0, so it must be requested explicitly).

## Reading the numbers honestly

- **Word-level survival is high on all three sources: 96.9%, 98.7%, and
  99.3%.** This is well above what a raw-CommonCrawl pipeline (CCNet,
  RefinedWeb) typically reports (often 30-50% survival after full dedup +
  LM filtering) - because none of the three sources measured here is
  genuinely raw, unfiltered web crawl. Sangraha's "verified" tier (what
  this project already trains on) and IndicCorp v2 have both already been
  through AI4Bharat's own quality pipeline before publication; Wikipedia
  is edited encyclopedic prose. Calling the Sangraha row "raw" is accurate
  relative to this project's own training path (`clean=False`, no
  `is_clean_bengali_line` applied), not relative to the open web.
  IndicCorp v2's own even-higher survival rate (99.3%, the highest of the
  three) reinforces this: it is a curated, benchmark-grade corpus, not a
  crawl dump. **CC-100 (2018 CommonCrawl-derived, already documented
  elsewhere in this repo as noisier and non-literary) would be a stronger
  true-raw-web proxy for a follow-up measurement**, not run this round.
- **Line-level survival is much lower for Wikipedia (76.3%) than word-level
  (96.9%).** Not a contradiction: Wikipedia articles repeat short
  boilerplate section headers across thousands of pages (references,
  external-links, see-also), which exact-dedup correctly removes as
  duplicate lines while barely touching total word count.
- **The absolute "≥5B clean tokens" threshold is answered with real
  measured data on the actual bulk source now, not a proxy.** The
  IndicCorp v2 row is 2,000,000 lines (90.6M words) sampled directly from
  IndicCorp v2 itself, not a different, more-curated stand-in. Applying
  its measured 99.3% word-survival ratio to AI4Bharat's own published
  30.0B-token figure for the full corpus (`bntok/corpus.py`'s
  `stream_indiccorp_v2` docstring) projects roughly 29.8B surviving
  tokens - comfortably above 5B. **This is still an extrapolation, not an
  exhaustive run**: 2M lines is roughly 0.1-0.15% of IndicCorp v2's full
  size (a pure-Python MinHash pass over the entire 30B-token corpus, at
  this measurement's own throughput, would run into the thousand-hour
  range - not attempted), so the projection assumes the measured ratio
  holds at full scale rather than confirming it line-for-line. That is a
  materially stronger basis than the previous version of this document
  (which extrapolated from Wikipedia/Sangraha, sources of a different
  kind and much smaller total scale, onto IndicCorp v2's figure) - real
  data, on the real source, at 45x the earlier sample size.

## Gate G3 verdict

**High-confidence pass on the ratio question; not a literal full-corpus
run.** Measured survival is 96.1% lines / 99.0% words pooled across all
three sources (96.4M raw words), far above the 5-10% floor the gate's
"if NO" clause treats as a crisis. The ≥5B-token threshold is cleared by
projection from a real, large, source-matched measurement (IndicCorp v2
itself, 90.6M words, 99.3% survival) against AI4Bharat's own published
corpus size, not by extrapolating from a different, smaller, more-curated
stand-in as the previous version of this document did. This does not
trigger the gate's "if NO" clause (re-weight the whole programme toward
Track B / OCR). The only way to turn this into a literal, non-extrapolated
answer would be a full pass over IndicCorp v2's entire ~30B tokens, which
is a different-order-of-magnitude undertaking (estimated in the
thousand-hour range at this pipeline's measured throughput) and not
something this gate's own spirit - "measure the survival ratio on real
data" - requires.

## What's still open

- A true raw-web proxy (CC-100, 2018 CommonCrawl-derived) has not been
  measured; all three sources measured here are curated/edited to some
  degree, so this likely overstates survival relative to genuinely raw
  crawl text. Worth a follow-up if a harder lower-bound number is needed.
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
