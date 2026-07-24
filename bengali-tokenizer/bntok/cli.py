r"""
Command line for the Track A Bengali tokenizer.

  python -m bntok gate-g1                       run shaping validation (Gate G1)
  python -m bntok train --input data/*.txt --algo bpe --vocab-size 64000 --out out/tok
  python -m bntok train --wikipedia bn --limit 5000 --vocab-size 64000 --out out/tok
  python -m bntok train --corpus-config configs/bpe-64k.json --vocab-size 64000 --out out/tok
  python -m bntok evaluate --tokenizer out/tok --input held_out.txt
  python -m bntok encode --tokenizer out/tok --text "আমি বাংলায় গান গাই"
  python -m bntok akshara --text "স্ত্রী ক্ষ্ম আকাঙ্ক্ষা"
  python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --out out/bmbt
  python -m bntok bmbt-encode --tokenizer out/bmbt --text "আমি বাংলায় গান গাই"
  python -m bntok bmbt-evaluate --tokenizer out/bmbt --input held_out.txt
  python -m bntok bmbt-featurize --text "স্ত্রী ক্ষ্ম আকাঙ্ক্ষা"

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
    from .corpus import build_configured_corpus, load_paths, stream_wikipedia
    if args.corpus_config:
        with open(args.corpus_config, encoding="utf-8") as f:
            config = json.load(f)
        return build_configured_corpus(config, log=lambda msg: print(msg, file=sys.stderr))
    if args.wikipedia:
        return stream_wikipedia(args.wikipedia, limit=args.limit)
    if args.input:
        return load_paths(args.input)
    raise BnTokError("provide --input FILES, --wikipedia LANG, or --corpus-config FILE")


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
    from .evaluate import evaluate
    from .tokenizer import BengaliTokenizer
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
    from .corpus import load_paths
    from .evaluate import evaluate
    from .tokenizer import BengaliTokenizer
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


def cmd_akshara(args) -> int:
    from .akshara import aksharas
    chunks = aksharas(args.text)
    print(json.dumps({
        "text": args.text,
        "n_chunks": len(chunks),
        "chunks": [
            {
                "text": c.text,
                "kind": c.kind,
                "start": c.start,
                "end": c.end,
                "codepoints": [f"U+{ord(cp):04X}" for cp in c.text],
            }
            for c in chunks
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_bmbt_train(args) -> int:
    from .bmbt import BMBT
    from .evaluate import evaluate
    corpus = _load_input(args)
    print(f"loaded {len(corpus)} lines; training BMBT {args.algo} vocab={args.vocab_size} ...",
          file=sys.stderr)
    tok = BMBT.train(
        corpus, algo=args.algo, vocab_size=args.vocab_size,
        min_atom_freq=args.min_atom_freq, zwnj_policy=args.zwnj,
    )
    tok.save(args.out)
    rep = evaluate(tok, corpus[: min(len(corpus), 2000)])
    print(json.dumps({"saved": args.out, "config": tok.config, "metrics_on_sample": rep.as_dict()},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_bmbt_evaluate(args) -> int:
    from .bmbt import BMBT
    from .corpus import load_paths
    from .evaluate import evaluate
    tok = BMBT.load(args.tokenizer)
    texts = load_paths(args.input)
    rep = evaluate(tok, texts)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_bmbt_encode(args) -> int:
    from .bmbt import BMBT
    tok = BMBT.load(args.tokenizer)
    ids = tok.encode(args.text)
    print(json.dumps({
        "text": args.text,
        "n_tokens": len(ids),
        "ids": ids,
        "tokens": tok.encode_tokens(args.text),
        "roundtrip_ok": tok.roundtrip_ok(args.text),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_bmbt_featurize(args) -> int:
    import dataclasses

    from .bmbt import AksharaFeatures, featurize
    chunks = featurize(args.text)
    out = []
    for c in chunks:
        if isinstance(c, AksharaFeatures):
            out.append({"kind": "akshara", **dataclasses.asdict(c)})
        else:
            out.append({"kind": c.kind, "text": c.text, "start": c.start, "end": c.end})
    print(json.dumps({"text": args.text, "n_chunks": len(chunks), "chunks": out},
                     ensure_ascii=False, indent=2))
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
    t.add_argument("--corpus-config", dest="corpus_config",
                    help="JSON config with weighted corpus_sources (see configs/bpe-64k.json)")
    t.add_argument("--algo", choices=["bpe", "unigram"], default="bpe")
    t.add_argument("--vocab-size", type=int, default=64000, dest="vocab_size")
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

    a = sub.add_parser("akshara", help="segment a string with the v2 akshara finite-state parser")
    a.add_argument("--text", required=True)
    a.set_defaults(func=cmd_akshara)

    bt = sub.add_parser("bmbt-train", help="train BMBT (Bornomala's Bengali Tokenizer, v2)")
    bt.add_argument("--input", nargs="*", help="text files or glob patterns")
    bt.add_argument("--wikipedia", help="stream Wikipedia for this language code (e.g. bn)")
    bt.add_argument("--limit", type=int, default=5000, help="max Wikipedia articles")
    bt.add_argument("--corpus-config", dest="corpus_config",
                     help="JSON config with weighted corpus_sources (see configs/bpe-64k.json)")
    bt.add_argument("--algo", choices=["bpe", "unigram"], default="bpe")
    bt.add_argument("--vocab-size", type=int, default=64000, dest="vocab_size")
    bt.add_argument("--min-atom-freq", type=int, default=2, dest="min_atom_freq")
    bt.add_argument("--zwnj", choices=["preserve", "strip"], default="preserve")
    bt.add_argument("--out", required=True, help="output directory for the tokenizer")
    bt.set_defaults(func=cmd_bmbt_train)

    be = sub.add_parser("bmbt-evaluate", help="evaluate a saved BMBT tokenizer")
    be.add_argument("--tokenizer", required=True, help="tokenizer directory")
    be.add_argument("--input", nargs="+", required=True, help="held-out text files")
    be.set_defaults(func=cmd_bmbt_evaluate)

    bc = sub.add_parser("bmbt-encode", help="encode a string with a saved BMBT tokenizer")
    bc.add_argument("--tokenizer", required=True)
    bc.add_argument("--text", required=True)
    bc.set_defaults(func=cmd_bmbt_encode)

    bf = sub.add_parser("bmbt-featurize",
                         help="segment and decompose a string into onset/vowel/modifier features")
    bf.add_argument("--text", required=True)
    bf.set_defaults(func=cmd_bmbt_featurize)

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
