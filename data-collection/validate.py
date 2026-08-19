r"""
CLI: validate a filled-in collection CSV against its schema, and optionally
emit a clean, NFC-normalised JSONL ready to drop into the repo.

Usage:
  python validate.py --type dialect_text --csv my_collected_rows.csv
  python validate.py --type ocr_ground_truth --csv scans_batch1.csv --emit-jsonl out/scans_batch1.jsonl

Exit code is non-zero if any row fails validation - safe to use in a
pre-commit check or a CI step once real data starts landing in the repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from schema import normalize_row, schema_for, validate_csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True,
                    choices=["dialect_text", "dialect_speech", "ocr_ground_truth", "general_text",
                             "annotator_profile"])
    p.add_argument("--csv", required=True)
    p.add_argument("--emit-jsonl", help="write clean, NFC-normalised JSONL here if validation passes")
    args = p.parse_args(argv)

    result = validate_csv(args.type, args.csv)
    print(f"{args.csv}: {result.total} rows, {len(result.errors)} error(s)", file=sys.stderr)
    for e in result.errors:
        print(f"  {e}", file=sys.stderr)

    if not result.ok:
        print("\nFAILED - fix the rows above before this data is usable.", file=sys.stderr)
        return 1

    print("PASSED", file=sys.stderr)
    if args.emit_jsonl:
        schema_for(args.type)  # re-validate type is known before writing
        with open(args.csv, encoding="utf-8-sig", newline="") as fin, \
             open(args.emit_jsonl, "w", encoding="utf-8") as fout:
            for row in csv.DictReader(fin):
                clean = normalize_row(args.type, row)
                fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
        print(f"wrote {args.emit_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
