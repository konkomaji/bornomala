# Data collection tooling

For datasets this project collects by hand, not pulled automatically from a
public source: Track C dialect text/speech, Track B OCR ground truth, and
general curated Bengali text. One schema, one validator, so nothing you
collect drifts out of the format the rest of the pipeline expects.

Fields are not invented - they come directly from
`PROJECT_BORNOMALA_Technical_Specification.md` section 10.3 (Track B
annotation protocol) and section 11.3 (Track C method and metadata list).

**Track B is not being simplified away here - it stays a real, standalone
research track**, building an actual Bengali OCR system (synthetic data
engine, two model tiers, a published benchmark, spec section 10.2-10.6),
motivated first by **preservation**: recovering West Bengal's pre-1950
literary and periodical print, much of it fragile and scattered, into
digital text efficiently and reliably - a contribution on its own, not only
a corpus source for Track A/E (spec section 10.1). Building those models is
months of separate work, though, so this tool is the practical way to start
recovering text NOW, before either model tier exists: use an existing OCR
tool or a frontier VLM, correct it by hand, record it as `ocr_ground_truth`
here. Every page digitised this way is both a preserved text today and
real training/eval data for the custom models later - not two efforts.

## Workflow

1. Copy the template for what you're collecting (`templates/*.csv`) to a
   working file, open it in any spreadsheet program.
2. Fill in real rows. Delete the `example-*` row(s) or leave them - the
   validator doesn't care, but don't ship them as real data.
3. Validate before it goes anywhere near the repo or a training run:

   ```bash
   cd data-collection
   python validate.py --type dialect_text --csv your_file.csv
   ```

   Non-zero exit code and a list of exact row numbers/reasons if anything's
   wrong - fix and re-run, don't guess.
4. Once it passes, emit clean JSONL:

   ```bash
   python validate.py --type dialect_text --csv your_file.csv --emit-jsonl out/your_file.jsonl
   ```

   This also NFC-normalises every text field (`bntok.normalize`, the same
   normalisation the tokenizer and every held-out benchmark in this project
   already use) - so ground truth is stored the way the spec requires
   without needing to remember to do it by hand.

## The four record types

| Type | What | Real fields, not invented |
|---|---|---|
| `dialect_text` | Standard Bengali -> dialect parallel sentence pairs | `dialect_group`, `district`, phonological/lexical/syntactic divergence notes - spec 11.3's elicitation + cross-check protocol |
| `dialect_speech` | Transcribed spontaneous dialect speech | Full speaker metadata list from spec 11.3 (age band, gender, education, first language, self-reported dialect, urban/rural) - **never a name, enforced by the validator, not just a convention** |
| `ocr_ground_truth` | Page image -> transcript | `category` matches spec 10.5's benchmark categories exactly (`letterpress_1880_1950` is the actual moat category); double-annotation fields for the 10% inter-annotator-agreement sample spec 10.3 requires |
| `general_text` | Any other hand-curated Bengali text | Lighter schema - category, source, collector, split |

## Dialect groups (`dialect_group` column, spec section 11.2)

`rarhi`, `manbhumi`, `varendri`, `sundarbani`, `kamrupi_rangpuri` - see
`schema.py`'s `DIALECT_GROUPS` for the district/zone each one covers.
**Document, do not adjudicate** contested classifications (Kāmrūpī/Rangpuri
specifically, per the spec).

## OCR categories (`category` column, spec section 10.5)

`modern_print`, `letterpress_1880_1950`, `newspaper_multicolumn`,
`forms_tables`, `phone_capture`, `handwriting`, `low_res_degraded`,
`code_mixed` - see `schema.py`'s `OCR_CATEGORIES`.

## What this does not do

No labeling UI - this is a spreadsheet-first workflow, deliberately, for a
solo/small collection effort. No automatic PII scrubbing beyond the
enforced no-speaker-name rule on `dialect_speech` - review manually before
anything is published, especially audio file names and any free-text
notes fields, which the validator does not and cannot fully police.
