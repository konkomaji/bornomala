"""Track A2, Gate G3: measure the real dedup+quality-filter survival ratio
on real Bengali text, per the spec's own "First 30 days" instruction
(section 15.3): "Download Bengali Wikipedia and a Sangraha Bengali shard.
Run dedup + quality filtering. Measure the survival ratio on real data."

Two sources, measured separately and pooled:
  - Bengali Wikipedia (via bntok.corpus.stream_wikipedia) - already
    fairly clean encyclopedic text, the "easy" case.
  - Sangraha web-typed, RAW (clean=False, unlike the tokenizer's own
    training path which applies is_clean_bengali_line up front) - the
    closer proxy for "raw Bengali web text" Gate G3 actually asks about.

Not a full-corpus run (IndicCorp v2 alone is a published 30.0B tokens;
downloading and near-deduping the literal entirety locally, at this
codebase's pure-Python near_dedup throughput (~1,800 lines/sec measured
on this machine), would take on the order of a thousand hours - not
attempted). `--indiccorp-limit` opts into a much larger, still-bounded
IndicCorp v2 sample (default 0 = skip; the original Wikipedia+Sangraha
pair remains the default run) to replace the earlier extrapolation with
a real measurement on IndicCorp v2 itself - the actual bulk, unedited
web-origin source, not a curated proxy for it. Projecting that measured
ratio onto the full 30.0B-token corpus is still an extrapolation (a
bigger sample, not the whole corpus), reported as such.

`--cc100-limit` adds a fourth, genuinely-raw-web pass: CC-100 (Wenzek et
al. 2020, the CommonCrawl-derived corpus behind XLM-R's training data,
already documented elsewhere in this repo as noisier and non-literary,
2018-vintage). Every other source measured here (Wikipedia, Sangraha,
IndicCorp v2) turned out to be a curated/verified pipeline output, not
open crawl text - CC-100 is this measurement's actual raw-web proxy.

Usage: python scripts/corpus_survival.py [--wikipedia-limit N] [--sangraha-limit N] [--indiccorp-limit N] [--cc100-limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok.corpus import stream_cc100, stream_indiccorp_v2, stream_sangraha, stream_wikipedia
from bntok.dedup import survival_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wikipedia-limit", type=int, default=3000, help="Wikipedia articles")
    ap.add_argument("--sangraha-limit", type=int, default=10000, help="Sangraha web-typed docs")
    ap.add_argument("--indiccorp-limit", type=int, default=0, help="IndicCorp v2 lines (0 = skip)")
    ap.add_argument("--cc100-limit", type=int, default=0, help="CC-100 lines (0 = skip)")
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

    extra_sources: list[str] = []
    extra_lines: list[str] = []

    if args.indiccorp_limit > 0:
        print(f"streaming IndicCorp v2 ({args.indiccorp_limit} lines) ...", file=sys.stderr)
        indiccorp = stream_indiccorp_v2("bn", limit_lines=args.indiccorp_limit)
        print(f"  {len(indiccorp)} lines", file=sys.stderr)

        print("running survival pipeline: indiccorp v2 ...", file=sys.stderr)
        indiccorp_report = survival_report(indiccorp)
        print(json.dumps(indiccorp_report, indent=2), file=sys.stderr)

        result["indiccorp_v2"] = indiccorp_report
        result["params"]["indiccorp_limit_lines"] = args.indiccorp_limit
        extra_sources.append("indiccorp_v2")
        extra_lines += indiccorp

    if args.cc100_limit > 0:
        print(f"streaming CC-100 ({args.cc100_limit} lines) ...", file=sys.stderr)
        cc100 = stream_cc100("bn", limit_lines=args.cc100_limit)
        print(f"  {len(cc100)} lines", file=sys.stderr)

        print("running survival pipeline: cc-100 ...", file=sys.stderr)
        cc100_report = survival_report(cc100)
        print(json.dumps(cc100_report, indent=2), file=sys.stderr)

        result["cc100"] = cc100_report
        result["params"]["cc100_limit_lines"] = args.cc100_limit
        extra_sources.append("cc100")
        extra_lines += cc100

    if extra_sources:
        label = " + ".join(["wikipedia", "sangraha"] + extra_sources)
        print(f"running survival pipeline: pooled_all ({label}) ...", file=sys.stderr)
        pooled_all_report = survival_report(wiki + sangraha + extra_lines)
        print(json.dumps(pooled_all_report, indent=2), file=sys.stderr)
        result["pooled_all"] = pooled_all_report
        result["params"]["pooled_all_sources"] = label

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
