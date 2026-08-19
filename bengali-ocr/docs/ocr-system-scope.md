# BMBO (Bornomala's Bengali OCR): system scope

I want an engine that reads Bengali correctly no matter what's put in
front of it - modern print, a 1930s letterpress scan, someone's
handwriting, a phone photo of a page, a multi-column newspaper, a PDF with
tables. This scopes what that actually takes, honestly, against
`PROJECT_BORNOMALA.md` section 10 (the existing plan) plus two real gaps
I found this pass that the spec doesn't cover yet.

## 1. Benchmark baseline list

The spec (section 10.5) already commits to beating every measurable
Bengali OCR baseline on one published benchmark: Tesseract Bengali,
Surya 2, PaddleOCR-VL, dots.ocr, Qwen3-VL, Gemini 3 Flash, Claude Opus,
GPT-5.2, Mistral OCR. Adding **Google Cloud Vision OCR and Baidu OCR** to
that list - neither publishes a Bengali-specific accuracy number to cite
directly, but both are real, testable systems via public API, and the
benchmark itself produces the number the same way it does for every other
row.

## 2. Coverage matrix

Everything in spec section 10.3's stratification, plus section 10.5's nine
benchmark categories - not repeating the tables here, they're already
right. What I'm adding on top:

**Script-era drift, not just image-quality degradation.** The spec's
categories (modern print, 1880-1950 letterpress, handwriting, degraded
scans, etc.) are mostly about *image condition*. There's a separate axis:
**the letterforms themselves changed across eras.** A model trained mostly
on modern Bengali typefaces has no guarantee of reading 1880s letterpress
type-cuts correctly even on a clean scan, and manuscript-era Bengali
(pre-print, palm-leaf, the Old/Middle Bengali bucket in
`data-collection/dataset-scope.md` section 2) is handwriting on top of
archaic letterforms - a compounding problem, not just "handwriting" as
one category. Concretely: I want typeface/letterform era tracked as its
own stratification variable in the benchmark, alongside image condition,
so a strong number on modern print doesn't quietly stand in for
"handles the Charyapada-through-1950 range" when it hasn't been tested on
that range at all.

## 3. Architecture: what the spec already has, plus two real additions

**Already in section 10.4, not repeating the design**: Tier 1 fast
(layout/line detector -> line recogniser -> CTC head emitting **grapheme
clusters, not codepoints**) and Tier 2 accurate (VLM fine-tune, LoRA
first). Two-model agreement filtering and hallucination scoring (section
10.4's admission gate) are also already specified.

**Addition 1: legacy pre-Unicode font recovery. Triage flag built, real
conversion still open.** A different bug class from OCR entirely - this is
about *born-digital* Bengali text that was never real Unicode to begin
with. Bengali web pages and PDFs from roughly before 2010 often used
proprietary fonts (Bijoy, SutonnyMJ, and similar ANSI-era font hacks) that
mapped Bengali glyphs onto Latin-range byte slots for display, not real
Bengali codepoints. Pull that text out of its original font context and
it's mojibake - not an OCR problem, not fixed by `bntok.normalize`'s
NFC/ZWJ policy, because the underlying bytes were never Bengali Unicode.
`bntok.corpus.flag_possible_legacy_encoding` now exists: given a source
declared/expected to be Bengali, it flags near-zero-Bengali-codepoint text
as worth human review - a triage signal, honestly scoped, not a fixer (its
own docstring says so). **Real recovery is still open**: detect *which*
legacy font, then convert via a verified per-font mapping table - study
existing open converters like Bijoy2Unicode/Avro's own reverse-conversion
logic as prior art, don't copy blindly, verify output against the same
Bengali-codepoint + akshara-validity checks below before trusting it.

**Addition 2: `bntok/akshara.py` as a post-OCR structural validator. Built
for raw text, not yet wired into an actual OCR pipeline.** BMBT's akshara
parser is already a real, tested, formal finite-state grammar of what a
legal Bengali conjunct/akshara structure looks like - built for
tokenization, but nothing about it is tokenization-specific.
`bntok.corpus.bengali_structural_validity_ratio` and
`is_structurally_valid_bengali` now exist, reusing that same grammar to
catch text that passes the plain Bengali-script-ratio check
(`is_clean_bengali_line`) but is structurally broken - a corrupted
conjunct, an orphaned matra/virama with no valid base, garbled reordering.
13 real tests in `tests/test_corpus.py`, including a genuine miss caught
mid-build: a mid-word orphan-matra placement first read back as
structurally valid (context-sensitive grammar behaviour, not a bug),
fixed by testing against the exact isolated case `test_akshara.py` already
proves. **Not yet wired into any actual OCR output pipeline** (there is no
OCR pipeline yet) or into `data-collection/validate.py` as a review-time
check - both real next steps, not done in this pass.

## 4. Metrics: extend, don't replace, section 10.6

GCER as headline, WER at whitespace-word granularity, hallucination rate
via the 4-gram formula - all already specified and correct, keep them.
**New metric to add: conjunct structural validity rate**, using the same
akshara grammar as Addition 2 above - the fraction of output aksharas that
parse as legal under `bntok/akshara.py`. This is the OCR-side sibling of
the tokenizer benchmark's own "destructive rate" metric
(`bengali-tokenizer/benchmarks/bengali-comparison.md`) - same grammar,
same validity concept, applied to a different pipeline stage. Keeping
Track A and Track B methodologically consistent rather than inventing a
second, unrelated notion of "correct" for OCR output.

## 5. Handwriting: realistic ceiling, not overpromised

Reuse and extend BN-HTRd and BanglaWriting per spec section 10.3's row 6.
Handwriting is a materially harder problem than print - even Tesseract
Bengali's own best published number (87.26% weighted F1 with
post-correction, spec section 4.5.2) is a print-only figure, and no
Bengali handwriting system in the spec's own literature survey
(section 4.5, Appendix A) claims comparable accuracy. I'm not setting a
handwriting target number here because I don't have a real baseline to
set it against yet - that's itself the honest scope statement: handwriting
needs its own baseline-gathering pass before a target means anything.

**A real collection-methodology precedent, checked directly, not just
cited by name**: BanglaLekha-Isolated (Biswas et al., *Data in Brief*,
2017, DOI 10.1016/j.dib.2017.03.035 - already in spec Appendix B.3, this
is the actual protocol behind that citation). 84 character classes (50
basic letters, 10 numerals, 24 of the most frequently used compound
characters/conjuncts), writers aged 6-28, 2,000 samples collected per
character under **timed writing tasks specifically to capture varying
handwriting quality** rather than only careful best-case samples, 166,105
images kept after discarding mistakes/scribbles, age and gender recorded
per writer. Preprocessing pipeline, in order: foreground/background
inversion to a consistent black-background/white-glyph convention, median
filter for noise removal, edge-thickening filter, then square resize with
aspect-ratio-preserving padding. Two things worth carrying into any future
BMBO handwriting collection: **timed writing as a deliberate quality-range
control** (not just recruiting "good handwriting"), and **per-writer
age/gender metadata** as a first-class field, not an afterthought - both
directly reusable design choices, not just a citation to point at.

**Access note**: the paper itself sits behind ScienceDirect's paywall;
this summary comes from its PMC-indexed abstract and corroborating
secondary sources (arXiv companion paper, dataset documentation), not the
publisher's own page - stated plainly rather than implying direct access
that wasn't there.

## 6. The self-learning loop (spec section 10.7, detailed)

The "flywheel" is already named in the spec; concretely, the loop is:
run current model on new scans -> route low-confidence/low-agreement pages
to human correction (the hallucination-score and two-model-agreement gates
from section 10.4 decide what counts as "low confidence") -> corrected
pages become new training data -> retrain -> re-run on more scans, with
correction volume needed per page dropping over successive rounds. **That
correction-volume-per-page curve is itself a metric worth tracking and
publishing** - it's the honest evidence that the flywheel is actually
turning, not just asserted to exist.

## 7. What this is not: a build plan for right now

This document is design, not an implementation - the model-training stages
(section 3) have not started. It exists so the design work is done ahead
of that, not so building starts today.
