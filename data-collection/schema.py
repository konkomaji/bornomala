r"""
Shared record schema for every dataset this project collects by hand
(dialect text/speech - Track C; OCR ground truth - Track B; general
curated text supplementing the automated web/Wikipedia/Sangraha corpus).

Fields here are not invented: they come directly from
`PROJECT_BORNOMALA.md` section 10.3 (Track B
annotation protocol) and section 11.3 (Track C method, dialect metadata
list, transcription convention), plus `dataset-scope.md`'s section 2 era
taxonomy (the `ERAS` enum below - composition era, orthogonal to
`OCR_CATEGORIES`'s image condition). One schema, four record types, so a
single validator/ingestion path works for all of it and nothing drifts
from what the spec actually asks each track to record.

This module has one external dependency, `bengali-tokenizer`'s own
`bntok.normalize` (NFC + ZWJ/ZWNJ policy) - the spec is explicit that
ground truth is stored NFC-normalised (Track B) and that a documented
transcription convention is itself a deliverable (Track C), and reusing
this project's own normalize() is the concrete way to keep that promise
mechanical rather than a style guideline nobody checks.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field

_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bengali-tokenizer")
if _TOKENIZER_DIR not in sys.path:
    sys.path.insert(0, _TOKENIZER_DIR)

from bntok.normalize import normalize  # noqa: E402 - path set up above

# --- Track C: dialect groups the spec targets (section 11.2) ---------------
DIALECT_GROUPS = {
    "rarhi": "Hooghly, Nadia, Howrah, Kolkata, Burdwan, Birbhum - prestige dialect, basis of literary standard",
    "manbhumi": "Purulia, Bankura, Jhargram, Medinipur, N. Burdwan; Jharkhand/Odisha border villages",
    "varendri": "Malda division; adjoining Bihar/Jharkhand villages",
    "sundarbani": "Presidency Division, Sundarbans belt",
    "kamrupi_rangpuri": "Cooch Behar, Jalpaiguri, Alipurduar - contested classification, document don't adjudicate",
}

# --- Track B: OCR categories the spec targets (section 10.5) ---------------
OCR_CATEGORIES = {
    "modern_print", "letterpress_1880_1950", "newspaper_multicolumn",
    "forms_tables", "phone_capture", "handwriting", "low_res_degraded", "code_mixed",
}

# --- Composition era, not print/scan date (data-collection/dataset-scope.md
# section 2) - orthogonal to OCR_CATEGORIES above, which describes image
# condition, not when the underlying text was written. A 1940 reprint of a
# 15th-century Krittibas Ramayan is category=modern_print, era=old_middle_bengali.
ERAS = {
    "old_middle_bengali": "8th-12th c. CE (Charyapada) through ~1800 - manuscript-tradition literature before print",
    "pre_vidyasagar": "~1800-1850 - print era begins, prose standardisation not yet done",
    "colonial_renaissance": "1850-1947 - Vidyasagar's prose standardisation onward through independence",
    "post_independence": "1947-1991 - partition through pre-liberalisation",
    "modern": "1991-present - almost entirely still in copyright, verify rights before collecting",
}


class ValidationError(Exception):
    """A record failed schema validation. Carries the row number and reason
    so a collector can find and fix the exact spreadsheet row."""


@dataclass
class ValidationResult:
    total: int
    valid: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# Required columns per record type, and which of those must be non-empty
# Bengali text (gets NFC-normalised on ingestion, not just checked).
_SCHEMAS = {
    "dialect_text": {
        "required": ["id", "standard_bengali", "dialect_text", "dialect_group", "district", "collector"],
        "text_fields": ["standard_bengali", "dialect_text"],
        "enum_fields": {"dialect_group": set(DIALECT_GROUPS)},
        "optional": ["second_annotator", "phonological_notes", "lexical_notes", "syntactic_notes"],
    },
    "dialect_speech": {
        # No speaker name, ever - spec section 11.3 is explicit about this.
        "required": ["id", "audio_path", "transcript", "dialect_group", "district",
                     "speaker_age_band", "speaker_gender", "speaker_education",
                     "first_language", "self_reported_dialect", "urban_rural",
                     "consent_ref", "topic_stimulus"],
        "text_fields": ["transcript"],
        "enum_fields": {"dialect_group": set(DIALECT_GROUPS), "urban_rural": {"urban", "rural"}},
        "optional": ["clip_seconds", "annotator"],
        "forbidden": ["speaker_name", "name", "full_name"],
    },
    "ocr_ground_truth": {
        "required": ["id", "image_path", "transcript", "category", "source", "annotator"],
        "text_fields": ["transcript"],
        "enum_fields": {"category": OCR_CATEGORIES, "era": set(ERAS)},
        "optional": ["double_annotated", "second_annotator", "hallucination_flag", "era"],
    },
    "general_text": {
        "required": ["id", "text", "category", "era", "source", "collector"],
        "text_fields": ["text"],
        "enum_fields": {"era": set(ERAS)},
        "optional": ["split", "notes"],
    },
}


def schema_for(record_type: str) -> dict:
    if record_type not in _SCHEMAS:
        raise ValidationError(f"unknown record_type {record_type!r}, expected one of {sorted(_SCHEMAS)}")
    return _SCHEMAS[record_type]


def validate_row(record_type: str, row: dict, row_num: int) -> list[str]:
    """Validate one CSV row. Returns a list of error strings (empty = valid).
    Does not raise - callers accumulate errors across the whole file so one
    bad row doesn't stop the rest of the file from being checked.
    """
    schema = schema_for(record_type)
    errors = []

    for col in schema["required"]:
        if not row.get(col, "").strip():
            errors.append(f"row {row_num}: missing required field '{col}'")

    for col in schema.get("forbidden", []):
        if row.get(col, "").strip():
            errors.append(f"row {row_num}: forbidden field '{col}' is set - never record speaker names "
                           f"(spec section 11.3)")

    for col, allowed in schema.get("enum_fields", {}).items():
        val = row.get(col, "").strip()
        if val and val not in allowed:
            errors.append(f"row {row_num}: '{col}'={val!r} not one of {sorted(allowed)}")

    for col in schema["text_fields"]:
        val = row.get(col, "")
        if val and not any(0x0980 <= ord(c) <= 0x09FF for c in val):
            errors.append(f"row {row_num}: '{col}' has no Bengali-block codepoints at all - "
                           f"check this is really Bengali text, not empty/placeholder")

    return errors


def normalize_row(record_type: str, row: dict) -> dict:
    """NFC-normalise every text field in a row (spec: ground truth is
    stored NFC-normalised). Non-text fields pass through unchanged."""
    schema = schema_for(record_type)
    out = dict(row)
    for col in schema["text_fields"]:
        if out.get(col):
            out[col] = normalize(out[col])
    return out


def validate_csv(record_type: str, path: str) -> ValidationResult:
    """Validate an entire CSV file against `record_type`'s schema. Also
    checks for duplicate ids across the whole file, not just per-row."""
    schema_for(record_type)  # raises on an unknown type before opening the file
    errors: list[str] = []
    seen_ids: set[str] = set()
    total = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # header is row 1
            total += 1
            row_errors = validate_row(record_type, row, row_num)
            rid = row.get("id", "").strip()
            if rid and rid in seen_ids:
                row_errors.append(f"row {row_num}: duplicate id {rid!r}")
            elif rid:
                seen_ids.add(rid)
            errors.extend(row_errors)
    return ValidationResult(total=total, valid=total - len({e.split(":")[0] for e in errors}), errors=errors)
