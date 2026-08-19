# BMBO (Bornomala's Bengali OCR): system scope

I want an engine that reads Bengali correctly no matter what's put in
front of it - modern print, a 1930s letterpress scan, someone's
handwriting, a phone photo of a page, a multi-column newspaper, a PDF with
tables. This scopes what that actually takes, honestly, against
`PROJECT_BORNOMALA.md` section 10 (the existing plan) plus two real gaps
I found this pass that the spec doesn't cover yet.

## 1. What "Google Vision / Baidu OCR level" actually means as a target

Neither publishes a Bengali-specific accuracy number. Google Cloud Vision
and Baidu OCR are general multilingual APIs - there is no published figure
for either on Bengali specifically, so "beat them" isn't a number I can
verify against, and stating one anyway would break my own transparency
rule (`PROJECT_BORNOMALA.md`'s rule E4: never present an estimate as
measured). What I can hold myself to instead, and what the spec already
sets up (section 10.5): **beat every baseline that's actually
measurable on Bengali, on one published benchmark, including the frontier
general models** - Tesseract Bengali, Surya 2, PaddleOCR-VL, dots.ocr,
Qwen3-VL, Gemini 3 Flash, Claude Opus, GPT-5.2, Mistral OCR. I'm adding
**Google Cloud Vision OCR and Baidu OCR (via their public APIs) to that
baseline list** now, since both are namable, testable competitors even
without a pre-existing published number - the benchmark itself produces
the number, same as it does for the others. That's the honest version of
"unlimited-OCR level": not a claim I can make today, a benchmark I commit
to running.

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

**Addition 1: legacy pre-Unicode font recovery.** A different bug class
from OCR entirely - this is about *born-digital* Bengali text that was
never real Unicode to begin with. Bengali web pages and PDFs from roughly
before 2010 often used proprietary fonts (Bijoy, SutonnyMJ, and similar
ANSI-era font hacks) that mapped Bengali glyphs onto Latin-range byte
slots for display, not real Bengali codepoints. Pull that text out of its
original font context and it's mojibake - not an OCR problem, not fixed
by `bntok.normalize`'s NFC/ZWJ policy, because the underlying bytes were
never Bengali Unicode. This isn't in the spec anywhere. Right now,
`data-collection/schema.py`'s Bengali-codepoint check catches and rejects
this junk on the way in (legacy bytes don't decode as Bengali-block
codepoints), which is a real safety net, but rejection isn't recovery -
correctly-legacy-encoded pages are just being dropped, not converted.
Scoped as its own small module: detect (byte-pattern/font-declaration
heuristics on old HTML/PDF sources) then convert (known legacy-font
mapping tables - study existing open converters like Bijoy2Unicode/Avro's
own reverse-conversion logic as prior art, don't copy blindly, verify
output against the same Bengali-codepoint + akshara-validity checks below
before trusting it).

**Addition 2: `bntok/akshara.py` as a post-OCR structural validator.**
BMBT's akshara parser is already a real, tested, formal finite-state
grammar of what a legal Bengali conjunct/akshara structure looks like -
built for tokenization, but nothing about it is tokenization-specific.
Running raw OCR or Tier-1/Tier-2 output through it as a post-processing
pass would flag (and in some cases mechanically repair - e.g. an orphaned
matra/virama with no valid base to attach to, the same kind of check
`banglish_tier3.py`'s beam re-rank already does for a different pipeline,
per `bengali-tokenizer/docs/known-issues.md`) structurally illegal
conjunct sequences before they ever reach a training set. This is a real,
not-yet-built idea, not a citation to outside work - flagging it here so
it doesn't get lost before Track B implementation starts.

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

GPU-gated (spec section 10.4, months 9-14 on rented GPU), and I don't have
GPU access in this environment - same blocker as the Banglish tier-3
neural retrain. This document exists so the design work is done before
that blocker clears, not so building starts today.
