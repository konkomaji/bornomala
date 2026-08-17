r"""
Assemble a real, large-scale Bengali pretraining corpus - the first concrete
step toward Track E's from-scratch decision (see _personal/SESSION_LOG.md
session 9). This is NOT `bntok.corpus.build_configured_corpus`: that
function is built for TOKENIZER induction (a small, weighted, cycled sample
of each source, capped around a `total_lines` budget of ~1.5M lines total)
and reuses its source loaders at their tiny induction-scale limits
(WIKIPEDIA_TRAIN_ARTICLES=15,000, INDICCORP_V2_TRAIN_LINES=300,000, etc, out
of sources that are orders of magnitude larger - IndicCorp v2 alone
publishes 30.0B tokens). A pretraining corpus needs to MAXIMIZE real,
deduped text, not sample a small representative slice of it.

Why this doesn't attempt the full ~29B-token G3 projection: near-dedup
(MinHash LSH, `bntok.dedup.near_dedup`) is inherently sequential - each
line's dedup decision depends on every prior line already inserted into the
LSH index - and was measured at roughly 2,000,000 lines per ~90 minutes
single-threaded (see docs/track-a2-corpus-survival.md). Reaching the full
target would need an estimated 400+ hours of near-dedup alone: not
achievable in this environment. Per an explicit scope decision (session 9),
this script runs exact-dedup + the rule-based quality filter (both
comfortably faster - see docs/known-issues.md's Gate G3 measurements, where
exact duplication was consistently the dominant loss factor, e.g. CC-100:
36.5% of lines exact-duplicate vs 0.2% near-duplicate) and skips near-dedup
for the bulk run. `--near-dedup` opts back in per-source for anyone willing
to pay the wall-clock cost.

Also unlike `build_configured_corpus`, this streams each source to its own
output file as soon as it is cleaned, rather than holding everything in
memory at once - a plain `set()`-based exact_dedup already holds one
source's lines in memory, and holding ALL sources simultaneously at this
scale is not safe to assume.

Usage:
  python scripts/assemble_pretrain_corpus.py --out-dir artifacts/pretrain-corpus-v1 \
    --lines-per-bulk-source 5000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok.corpus import (
    stream_cc100,
    stream_indiccorp_v2,
    stream_sangraha,
    stream_wikipedia,
    stream_wikisource,
    stream_xlsum,
)
from bntok.dedup import exact_dedup, near_dedup, quality_filter


def _clean(lines: list[str], do_near_dedup: bool, log) -> list[str]:
    before = len(lines)
    lines, removed_exact = exact_dedup(lines)
    if do_near_dedup:
        lines, removed_near = near_dedup(lines)
    else:
        removed_near = 0
    lines, removed_quality = quality_filter(lines)
    log(f"    {before} -> {len(lines)} lines "
        f"(exact -{removed_exact}, near -{removed_near}, quality -{removed_quality})")
    return lines


def _write_source(out_dir: str, name: str, lines: list[str]) -> tuple[int, int]:
    path = os.path.join(out_dir, f"{name}.txt")
    n_words = 0
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            n_words += len(line.split())
    return len(lines), n_words


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lines-per-bulk-source", type=int, default=5_000_000,
                    help="cap for IndicCorp v2, CC-100, Sangraha web - the sources large enough "
                         "that 'all of it' is not realistic in one run")
    p.add_argument("--wikipedia-limit", type=int, default=500_000, help="article cap, not line cap")
    p.add_argument("--sangraha-pdf-limit-docs", type=int, default=200_000)
    p.add_argument("--sangraha-pdf-max-files", type=int, default=20)
    p.add_argument("--xlsum-limit-docs", type=int, default=20_000)
    p.add_argument("--near-dedup", action="store_true",
                    help="also run near-dedup (slow, ~2M lines/90min single-threaded) - "
                         "off by default per the scope decision in this script's own docstring")
    p.add_argument("--skip", nargs="*", default=[],
                    choices=["wikisource", "sangraha_pdf", "sangraha_web", "wikipedia",
                             "xlsum", "cc100", "indiccorp_v2"],
                    help="source names to skip (e.g. to resume after a partial run)")
    args = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}
    t0 = time.time()

    def run_source(name: str, loader_fn, *fn_args, **fn_kwargs):
        if name in args.skip:
            log(f"[{name}] skipped by --skip")
            return
        out_path = os.path.join(args.out_dir, f"{name}.txt")
        if os.path.exists(out_path):
            log(f"[{name}] {out_path} already exists, skipping (delete it to redo)")
            return
        log(f"[{name}] streaming ...")
        try:
            lines = loader_fn(*fn_args, **fn_kwargs)
        except Exception as e:  # noqa: BLE001 - one source failing must not kill the whole run
            log(f"[{name}] FAILED to stream: {type(e).__name__}: {e}")
            stats[name] = {"error": str(e)}
            return
        log(f"  {len(lines)} raw lines, cleaning ...")
        lines = _clean(lines, args.near_dedup, log)
        n_lines, n_words = _write_source(args.out_dir, name, lines)
        stats[name] = {"lines": n_lines, "words": n_words}
        log(f"[{name}] done: {n_lines} lines, {n_words} words -> {out_path}")

    # Naturally-bounded sources: pull everything available (all defaults
    # large - overridable via CLI, e.g. for a quick correctness smoke test
    # with small numbers on every source, not just the bulk ones).
    run_source("wikisource", stream_wikisource, "bn")
    run_source("sangraha_pdf", stream_sangraha, "ben", doc_type="pdf",
               limit_docs=args.sangraha_pdf_limit_docs, max_files=args.sangraha_pdf_max_files, clean=True)
    run_source("wikipedia", stream_wikipedia, "bn", limit=args.wikipedia_limit)
    run_source("xlsum", stream_xlsum, "bengali", limit_docs=args.xlsum_limit_docs)

    # Bulk sources: capped at --lines-per-bulk-source, still an order of
    # magnitude (or more) beyond the tokenizer-induction constants.
    run_source("sangraha_web", stream_sangraha, "ben", doc_type="web",
               limit_docs=args.lines_per_bulk_source, max_files=40)
    run_source("cc100", stream_cc100, "bn", limit_lines=args.lines_per_bulk_source)
    run_source("indiccorp_v2", stream_indiccorp_v2, "bn", limit_lines=args.lines_per_bulk_source)

    total_lines = sum(s.get("lines", 0) for s in stats.values())
    total_words = sum(s.get("words", 0) for s in stats.values())
    # Rough token estimate: this project's own tokenizers measure ~1.2-1.5
    # tokens/word on real held-out Bengali (see benchmarks/), so this is a
    # labeled estimate, not a measured one - the real count comes from
    # actually tokenizing the assembled corpus (a separate step).
    estimated_tokens_low = int(total_words * 1.2)
    estimated_tokens_high = int(total_words * 1.5)

    summary = {
        "sources": stats,
        "total_lines": total_lines,
        "total_words": total_words,
        "estimated_tokens_range": [estimated_tokens_low, estimated_tokens_high],
        "estimate_basis": "words * [1.2, 1.5], this project's own measured fertility range on "
                           "held-out Bengali - NOT a measured token count. Tokenize the assembled "
                           "corpus for a real number.",
        "near_dedup_applied": args.near_dedup,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\nTOTAL: {total_lines} lines, {total_words} words, "
        f"estimated {estimated_tokens_low:,}-{estimated_tokens_high:,} tokens "
        f"(estimate, not measured - see summary.json)")
    log(f"elapsed: {summary['elapsed_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
