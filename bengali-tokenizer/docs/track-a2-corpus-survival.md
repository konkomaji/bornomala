# Track A2: corpus dedup + quality-filter survival ratio (Gate G3)

Gate G3 (`PROJECT_BORNOMALA.md` §16, risk R6): after dedup and quality
filtering, what fraction of raw Bengali web text survives, and does the
surviving corpus clear 5 billion clean tokens? This is that measurement,
run on real data: Bengali Wikipedia, a Sangraha Bengali shard, IndicCorp
v2, and CC-100 as a genuine raw-web proxy.

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

Four real sources, streamed via the existing `bntok.corpus` loaders,
measured separately and pooled (`scripts/corpus_survival.py`):

- **Bengali Wikipedia**, 3,000 articles (`stream_wikipedia`).
- **Sangraha web-typed, raw** (`stream_sangraha(doc_type="web", clean=False)`),
  10,000 documents - unlike the tokenizer's own training path, no
  `is_clean_bengali_line` pre-filter applied before this pipeline runs.
- **AI4Bharat IndicCorp v2** (`stream_indiccorp_v2`), 2,000,000 lines - a
  large-scale follow-up pass (`--indiccorp-limit`), added to replace an
  earlier extrapolation with a real measurement on a bulk source.
- **CC-100** (`stream_cc100`), 1,000,000 lines - the genuine raw-web
  proxy: 2018 CommonCrawl-derived, no AI4Bharat/Wikimedia curation
  pipeline behind it, added specifically because the other three sources
  all turned out to be curated/verified corpora, not raw crawl text.

| Source | Raw lines | Raw words | Survival (lines) | Survival (words) |
|---|--:|--:|--:|--:|
| Wikipedia | 159,136 | 2,718,289 | 76.3% | 96.9% |
| Sangraha web (raw) | 86,788 | 3,084,853 | 96.6% | 98.7% |
| Pooled (wiki + sangraha) | 245,924 | 5,803,142 | 83.5% | 97.9% |
| IndicCorp v2 | 2,000,000 | 90,590,866 | 97.9% | 99.3% |
| Pooled (wiki + sangraha + indiccorp) | 2,245,924 | 96,394,008 | 96.1% | 99.0% |
| **CC-100 (raw-web proxy)** | **1,000,000** | **9,467,566** | **63.2%** | **79.6%** |
| Pooled (wiki + sangraha + cc100) | 1,245,924 | 15,270,708 | 67.1% | 86.5% |

Full breakdown by removal stage: `track-a2-corpus-survival.json`
(wikipedia/sangraha/indiccorp v2) and `track-a2-corpus-survival-cc100.json`
(wikipedia/sangraha/cc100 - a separate run and output file, so the
already-computed, ~90-minute IndicCorp v2 near-dedup did not need to be
repeated; the two "pooled" rows above are therefore each pooled against
wikipedia+sangraha independently, not all four sources in one combined
pool). Reproduce: `python scripts/corpus_survival.py --indiccorp-limit 2000000`
and `python scripts/corpus_survival.py --cc100-limit 1000000 --out docs/track-a2-corpus-survival-cc100.json`
from `bengali-tokenizer/` (near-dedup took roughly 90-100 minutes for the
2M-line IndicCorp v2 pass and roughly 30-45 minutes for the 1M-line CC-100
pass on this machine, single-threaded pure-Python MinHash; both limits
default to 0/skipped, so must be requested explicitly).

## Reading the numbers honestly

- **CC-100 is the outlier, and it is the real signal.** Word survival is
  79.6%, well below Wikipedia/Sangraha/IndicCorp v2's 96.9-99.3% - because
  CC-100 is the only source of the four with no AI4Bharat or Wikimedia
  curation pipeline behind it. The other three all turned out to be
  curated/verified corpora, not raw crawl text: Sangraha's "verified"
  tier and IndicCorp v2 have both already been through AI4Bharat's own
  quality pipeline before publication; Wikipedia is edited encyclopedic
  prose. Calling the Sangraha row "raw" is accurate relative to this
  project's own training path (`clean=False`, no `is_clean_bengali_line`
  applied there), not relative to the open web.
- **Even CC-100's loss is mostly exact duplication, not garbage content.**
  364,723 of 1,000,000 lines (36.5%) were removed as exact duplicates -
  templated/boilerplate page text, a known CommonCrawl pattern - versus
  only 1,029 lines (0.1%) rejected by the quality filter and 2,302 (0.2%)
  by near-dedup. Once deduplicated, what remains is not obviously dirtier
  than the other sources; there is simply much more repetition in raw
  crawl text to begin with.
- **CC-100 is itself not literally unprocessed crawl text either** - Wenzek
  et al. (2020) built it with a language-ID filter over CommonCrawl, so
  79.6% is a floor on how raw this measurement gets, not a measurement of
  zero-processed web text. True open-web survival could be lower still;
  not measurable without running an original crawl, out of scope here.
- **Line-level survival is much lower than word-level for every source**
  (e.g. Wikipedia 76.3% vs 96.9%, CC-100 63.2% vs 79.6%). Not a
  contradiction: short boilerplate lines (navigation, headers, templated
  fragments) repeat far more often than full sentences, so exact-dedup
  removes many more *lines* than it removes *words*.
- **The absolute "≥5B clean tokens" threshold holds even under the worst
  (truest raw-web) measured ratio.** IndicCorp v2's own published size is
  30.0B tokens (`bntok/corpus.py`'s `stream_indiccorp_v2` docstring).
  Applying IndicCorp v2's own measured ratio (99.3% words) projects ~29.8B
  surviving tokens; applying CC-100's much lower measured ratio (79.6%
  words) to the same 30.0B figure still projects ~23.9B - both comfortably
  above 5B. The available Bengali corpora are roughly two orders of
  magnitude larger than the threshold at any of the ratios measured here,
  so which ratio is "the right one to use" does not change the answer.
  **This is still an extrapolation, not an exhaustive run**: 2M
  (IndicCorp v2) and 1M (CC-100) lines are each well under 1% of their
  respective full corpora - a full pass over either at this measurement's
  own throughput would run into the thousand-hour range, not attempted.

## Gate G3 verdict

**High-confidence pass, now stress-tested against a genuine raw-web
proxy.** The worst-case measured survival ratio across all four sources
(CC-100, 63.2% lines / 79.6% words) is still far above the 5-10% floor
the gate's "if NO" clause treats as a crisis, and still projects well
above the 5B-token threshold when applied to any of the available bulk
corpora's published sizes. This does not trigger the gate's "if NO"
clause (re-weight the whole programme toward Track B / OCR). The only way
to turn this into a literal, non-extrapolated answer would be a full pass
over IndicCorp v2's and/or CC-100's entire corpora, a different-order-of-
magnitude undertaking (thousand-hour range at this pipeline's measured
throughput) not required by the gate's own instruction to "measure the
survival ratio on real data."

## What's still open

- LM-perplexity filtering remains unavailable without building `kenlm`
  from source (no `lmplz` in the PyPI wheel); not blocking, since the
  rule-based pipeline alone already answers the gate's core question.
- **Now wired into `build_configured_corpus` itself, opt-in** (`dedup=True`,
  CLI `--dedup` on `train`/`bmbt-train`): runs exact dedup,
  near dedup, and quality filtering on each source's lines before
  `weighted_corpus` combines them - per-source, not on the final weighted
  output, since `weighted_corpus` deliberately cycles a thin source (e.g.
  Wikisource) to hit its target share, and deduping the final output
  would strip out that intentional repetition. Default `False`: this
  changes what a retrained artifact would contain versus the shipped
  `bn-bpe-64k`/`bmbt-64k`, so it does not silently change the existing
  default. Not yet used to retrain either shipped artifact - that is
  still a separate decision.
- No single run pools all four sources together (IndicCorp v2 and CC-100
  were measured in separate invocations to avoid re-running the
  expensive IndicCorp v2 near-dedup); if a single combined-pool number is
  ever needed, it requires one more full run across all four sources at
  once.
