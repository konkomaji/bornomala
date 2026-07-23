r"""
Command line for the Track A Bengali tokenizer.

  python -m bntok gate-g1                       run shaping validation (Gate G1)
  python -m bntok train --input data/*.txt --algo bpe --vocab-size 32000 --out out/tok
  python -m bntok train --wikipedia bn --limit 5000 --vocab-size 32000 --out out/tok
  python -m bntok evaluate --tokenizer out/tok --input held_out.txt
  python -m bntok encode --tokenizer out/tok --text "আমি বাংলায় গান গাই"

Every command validates its inputs and reports errors as clear messages with a
non-zero exit code, never a raw traceback.
"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import BnTokError

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

_DEFAULT_SAMPLES = [
    "আমি বাংলায় গান গাই",
    "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে",
    "যুক্তাক্ষর: ক্ষ জ্ঞ ত্র দ্ধ ঙ্ক্ষ র্ম র্য",
    "রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের শ্রেষ্ঠ কবি",
]


def _load_input(args) -> list[str]:
    from .corpus import load_paths, stream_wikipedia
    if args.wikipedia:
        return stream_wikipedia(args.wikipedia, limit=args.limit)
    if args.input:
        return load_paths(args.input)
    raise BnTokError("provide --input FILES or --wikipedia LANG")


def cmd_gate_g1(args) -> int:
    from .shaping import validate
    samples = _DEFAULT_SAMPLES
    if args.input:
        from .corpus import load_paths
        samples = load_paths(args.input)[:2000]
    rep = validate(samples, font_path=args.font)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if rep["status"] == "fail":
        print("Gate G1 FAILED: shaping produced missing glyphs or cluster mismatches.", file=sys.stderr)
        return 2
    if rep["status"] == "skipped":
        print(f"Gate G1 skipped: {rep['reason']}.", file=sys.stderr)
        return 0
    print(f"Gate G1 passed: {rep['checked']} samples shaped cleanly.")
    return 0


def cmd_train(args) -> int:
    from .tokenizer import BengaliTokenizer
    from .evaluate import evaluate
    corpus = _load_input(args)
    print(f"loaded {len(corpus)} lines; training {args.algo} vocab={args.vocab_size} ...", file=sys.stderr)
    tok = BengaliTokenizer.train(
        corpus, algo=args.algo, vocab_size=args.vocab_size,
        min_atom_freq=args.min_atom_freq, zwnj_policy=args.zwnj,
    )
    tok.save(args.out)
    rep = evaluate(tok, corpus[: min(len(corpus), 2000)])
    print(json.dumps({"saved": args.out, "config": tok.config, "metrics_on_sample": rep.as_dict()},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    from .tokenizer import BengaliTokenizer
    from .evaluate import evaluate
    from .corpus import load_paths
    tok = BengaliTokenizer.load(args.tokenizer)
    texts = load_paths(args.input)
    rep = evaluate(tok, texts)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_encode(args) -> int:
    from .tokenizer import BengaliTokenizer
    tok = BengaliTokenizer.load(args.tokenizer)
    ids = tok.encode(args.text)
    print(json.dumps({
        "text": args.text,
        "n_tokens": len(ids),
        "ids": ids,
        "tokens": tok.encode_tokens(args.text),
        "roundtrip_ok": tok.roundtrip_ok(args.text),
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bntok", description="Project Bornomala Track A Bengali tokenizer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate-g1", help="run HarfBuzz shaping validation (Gate G1)")
    g.add_argument("--input", nargs="*", help="text files to validate (default: built-in samples)")
    g.add_argument("--font", help="path to a Bengali font (default: auto-detect)")
    g.set_defaults(func=cmd_gate_g1)

    t = sub.add_parser("train", help="train a tokenizer")
    t.add_argument("--input", nargs="*", help="text files or glob patterns")
    t.add_argument("--wikipedia", help="stream Wikipedia for this language code (e.g. bn)")
    t.add_argument("--limit", type=int, default=5000, help="max Wikipedia articles")
    t.add_argument("--algo", choices=["bpe", "unigram"], default="bpe")
    t.add_argument("--vocab-size", type=int, default=32000, dest="vocab_size")
    t.add_argument("--min-atom-freq", type=int, default=2, dest="min_atom_freq")
    t.add_argument("--zwnj", choices=["preserve", "strip"], default="preserve")
    t.add_argument("--out", required=True, help="output directory for the tokenizer")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved tokenizer")
    e.add_argument("--tokenizer", required=True, help="tokenizer directory")
    e.add_argument("--input", nargs="+", required=True, help="held-out text files")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("encode", help="encode a string")
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--text", required=True)
    c.set_defaults(func=cmd_encode)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except BnTokError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
