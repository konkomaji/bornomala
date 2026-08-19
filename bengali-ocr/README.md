# BMBO: Bornomala's Bengali OCR

Track B, named the same way BMBT (Bornomala's Bengali Tokenizer) is named.
This is the document-recognition track: turning page images, of any era and
any condition, into clean NFC-normalised Bengali text. `PROJECT_BORNOMALA.md`
section 10 is the source of truth for the plan; `docs/ocr-system-scope.md`
in this folder is a deeper scoping pass on top of it, written after real
OCR noise turned up in the first archive.org pulls for the data-collection
programme (see `data-collection/dataset-scope.md` section 10).

Nothing here is built yet. This is design, not code - GPU-gated per the
spec's own timeline (section 10.4, months 9-14 on rented GPU), and I don't
have GPU access in this environment right now (same blocker as the
Banglish tier-3 neural retrain, `bengali-tokenizer/docs/known-issues.md`).
Scoping it now means it's ready to build the day that blocker clears,
instead of starting the design work from zero then.

## Naming

**BMBO**, matching **BMBT**'s own naming pattern in the tokenizer track -
Bornomala's [Bengali Tokenizer / Bengali OCR], same lineage, same
Borno- root as the project name itself.
