"""Track A2, Gate G3: measure the real dedup+quality-filter survival ratio
on real Bengali text, per the spec's own "First 30 days" instruction
(section 16.3): "Download Bengali Wikipedia and a Sangraha Bengali shard.
Run dedup + quality filtering. Measure the survival ratio on real data."

Two sources, measured separately and pooled:
  - Bengali Wikipedia (via bntok.corpus.stream_wikipedia) - already
    fairly clean encyclopedic text, the "easy" case.
  - Sangraha web-typed, RAW (clean=False, unlike the tokenizer's own
    training path which applies is_clean_bengali_line up front) - the
    closer proxy for "raw Bengali web text" Gate G3 actually asks about.

Not a full-corpus run (Sangraha/IndicCorp v2 run into the billions of
tokens; downloading and near-deduping that locally is a different-scale
undertaking). This measures the survival RATIO precisely on a real,
sizeable sample, and is explicit that projecting the ratio onto an
absolute "5B token" threshold needs the full corpus size, which this
script does not have - reported as a limitation, not glossed over.

Usage: python scripts/corpus_survival.py [--wikipedia-limit N] [--sangraha-limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok.corpus import stream_sangraha, stream_wikipedia
from bntok.dedup import survival_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wikipedia-limit", type=int, default=3000, help="Wikipedia articles")
    ap.add_argument("--sangraha-limit", type=int, default=10000, help="Sangraha web-typed docs")
    ap.add_argument("--out", default="docs/track-a2-corpus-survival.json")
    args = ap.parse_args()

    print(f"streaming Wikipedia ({args.wikipedia_limit} articles) ...", file=sys.stderr)
    wiki = stream_wikipedia("bn", limit=args.wikipedia_limit)
    print(f"  {len(wiki)} lines", file=sys.stderr)

    print(f"streaming Sangraha web-typed, RAW ({args.sangraha_limit} docs) ...", file=sys.stderr)
    sangraha = stream_sangraha("ben", doc_type="web", limit_docs=args.sangraha_limit, max_files=4, clean=False)
    print(f"  {len(sangraha)} lines", file=sys.stderr)

    print("running survival pipeline: wikipedia ...", file=sys.stderr)
    wiki_report = survival_report(wiki)
    print(json.dumps(wiki_report, indent=2), file=sys.stderr)

    print("running survival pipeline: sangraha web (raw) ...", file=sys.stderr)
    sangraha_report = survival_report(sangraha)
    print(json.dumps(sangraha_report, indent=2), file=sys.stderr)

    print("running survival pipeline: pooled ...", file=sys.stderr)
    pooled_report = survival_report(wiki + sangraha)
    print(json.dumps(pooled_report, indent=2), file=sys.stderr)

    result = {
        "wikipedia": wiki_report,
        "sangraha_web_raw": sangraha_report,
        "pooled": pooled_report,
        "params": {
            "wikipedia_limit_articles": args.wikipedia_limit,
            "sangraha_limit_docs": args.sangraha_limit,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
