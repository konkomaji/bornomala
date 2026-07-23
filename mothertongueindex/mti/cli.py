"""
MotherTongueIndex command-line interface.

Examples:
  python -m mti "আমি বাংলায় গান গাই"
  python -m mti --models gpt-4o,sarvam1,claude "the quick brown fox"
  python -m mti --file sample.txt --json
  python -m mti --list
  python -m mti --show "কি খবর" --models gpt-4o,sarvam1   # show token split
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyze import analyze, cost_explanation
from .registry import DEFAULT_MODELS, GROUPS, list_models

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# Bengali, arrows, or box characters. Force UTF-8 so multilingual output works.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def _fmt_table(results) -> str:
    cols = ["model", "tokens", "words", "fert.", "xEN", "STRR", "b/tok", "gc/tok", ""]
    rows = []
    for r in results:
        if not r.available:
            rows.append([r.display, "-", "-", "-", "-", "-", "-", "-", "unavailable"])
            continue
        m = r.metrics
        flag = "est" if m.estimated else ""
        xen = f"{r.vs_english:.2f}x" if r.vs_english else "-"
        rows.append([
            r.display,
            str(m.n_tokens),
            str(m.n_words),
            f"{m.fertility:.2f}",
            xen,
            f"{m.strr:.2f}",
            f"{m.bytes_per_token:.1f}",
            f"{m.gc_per_token:.2f}",
            flag,
        ])
    widths = [max(len(str(x)) for x in [c] + [row[i] for row in rows]) for i, c in enumerate(cols)]
    def line(vals):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))
    out = [line(cols), line(["-" * w for w in widths])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _show_tokens(results, text: str) -> str:
    out = []
    for r in results:
        if not r.available:
            out.append(f"\n## {r.display}: unavailable ({r.error})")
            continue
        if not r.tokens:
            out.append(f"\n## {r.display}: {r.metrics.n_tokens} tokens (no surface strings - estimate)")
            continue
        joined = "│".join(t.replace("\n", "⏎") for t in r.tokens)
        out.append(f"\n## {r.display}: {r.metrics.n_tokens} tokens\n{joined}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mti", description="Multilingual tokenizer efficiency analyzer.")
    p.add_argument("text", nargs="?", help="text to analyze (or use --file / stdin)")
    p.add_argument("--file", help="read text from a file")
    p.add_argument("--models", help="comma-separated model ids (default: a no-auth set)")
    p.add_argument("--group", help="preset model group: default, openai, frontier, indian, multilingual, open")
    p.add_argument("--list", action="store_true", help="list known models (and groups) and exit")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--show", action="store_true", help="show the token split per model")
    p.add_argument("--why", action="store_true", help="print plain-language cost explanation")
    p.add_argument("--capability", action="store_true",
                   help="show derived reasoning-capability impact (effective context vs English)")
    args = p.parse_args(argv)

    if args.list:
        for m in list_models():
            print(f"{m.id:12s} {m.tier:8s} {m.display}   {m.note}")
        print("\nGroups (use --group NAME):")
        for g, ids in GROUPS.items():
            print(f"  {g:12s} {', '.join(ids)}")
        return 0

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        p.error("provide text, --file, or pipe via stdin")

    if args.models:
        models = [s.strip() for s in args.models.split(",")]
    elif args.group:
        if args.group not in GROUPS:
            p.error(f"unknown group '{args.group}'. Choices: {', '.join(GROUPS)}")
        models = GROUPS[args.group]
    else:
        models = DEFAULT_MODELS
    results = analyze(text, models, want_tokens=args.show)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
        return 0

    print(_fmt_table(results))
    if args.show:
        print(_show_tokens(results, text))
    if args.why:
        print("\nvs English (same model, tokens/word ratio to the English anchor):")
        any_en = False
        for r in results:
            if r.available and r.vs_english:
                any_en = True
                tag = " (estimate)" if r.metrics.estimated else ""
                print(f"  - {r.display}{tag}: {r.vs_english:.2f}x English "
                      f"- ~{r.vs_english:.2f}x the tokens, and cost, per word.")
        if not any_en:
            print("  - (input looks like English, or no model loaded)")
        print("\nAcross models (which tokenizer is most efficient for THIS text):")
        for line in cost_explanation(results):
            print("  - " + line)

    if args.capability:
        from .capability import assess, summary_line
        print("\nReasoning-capability impact (DERIVED from tokenization, not measured):")
        shown = False
        for r in results:
            if not (r.available and r.vs_english):
                continue
            ci = assess(r.model_id, r.vs_english, estimated=r.metrics.estimated)
            if ci:
                shown = True
                print("  - " + summary_line(ci))
        if not shown:
            print("  - (input looks like English, or no model loaded)")
        print("  note: measured reasoning accuracy requires eval/reasoning_probe.py "
              "(runs on a GPU/API machine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
