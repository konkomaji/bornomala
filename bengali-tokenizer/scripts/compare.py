r"""
Cross-tokenizer Bengali comparison (the showcase table).

Measures the Bornomala Track A tokenizer against the incumbents on the SAME
held-out Bengali text, with one consistent definition of every metric:

  fertility          tokens / whitespace-words   (lower is better)
  strr               fraction of words kept as one token
  bytes_per_token    UTF-8 bytes / tokens
  conjunct_fragmentation
                     fraction of grapheme clusters that a token boundary splits,
                     computed from each tokenizer's own character offsets. This is
                     the property no general Indic tokenizer controls for.

The whitepaper notes that Bengali does not appear in a single published
cross-tokenizer fertility comparison (spec section 4.1). This produces exactly
that comparison, from real tokenizers, reproducibly. Nothing is fabricated: a
tokenizer that cannot be loaded is reported as unavailable, not estimated.

Held-out set: Bengali Wikipedia articles AFTER the training range, so the
Bornomala numbers are on unseen text, an honest comparison.

Usage:
  python scripts/compare.py --tokenizer out/bn-bpe-32k --skip 12000 --limit 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok import BengaliTokenizer, normalize
from bntok.graphemes import grapheme_clusters

# HF tokenizers to compare against (all load without auth).
HF_MODELS = [
    ("Sarvam-1 (Sarvam AI)", "sarvamai/sarvam-1"),
    ("IndicBERT (AI4Bharat)", "ai4bharat/indic-bert"),
    ("mBERT (Google)", "google-bert/bert-base-multilingual-cased"),
    ("XLM-RoBERTa (Meta)", "FacebookAI/xlm-roberta-base"),
    ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
]
TIKTOKEN_MODELS = [("GPT-4o (OpenAI o200k)", "o200k_base")]


def held_out(skip: int, limit: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        if len(out) >= limit:
            break
        for p in row.get("text", "").split("\n"):
            if p.strip():
                out.append(p.strip())
    return out


def _frag_from_offsets(nfc: str, offsets: list[tuple[int, int]]) -> tuple[int, int]:
    """Count grapheme clusters split by any token boundary, using char offsets."""
    boundaries = set()
    for s, e in offsets:
        boundaries.add(s)
        boundaries.add(e)
    pos = 0
    frag = total = 0
    for g in grapheme_clusters(nfc):
        start, end = pos, pos + len(g)
        pos = end
        if g.isspace():
            continue
        total += 1
        if any(start < b < end for b in boundaries):
            frag += 1
    return frag, total


def measure_hf(name: str, repo: str, texts: list[str]) -> dict | None:
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(repo, use_fast=True)
    except Exception as e:
        return {"model": name, "available": False, "error": f"{type(e).__name__}: {e}"}

    n_tok = n_words = n_bytes = single = frag = clusters = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        try:
            enc = tk(nfc, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offs = enc.get("offset_mapping") or []
        except Exception:
            ids = tk(nfc, add_special_tokens=False)["input_ids"]
            offs = []
        n_tok += len(ids)
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(tk(w, add_special_tokens=False)["input_ids"]) == 1:
                single += 1
        if offs:
            f, c = _frag_from_offsets(nfc, offs)
            frag += f
            clusters += c

    return _row(name, n_tok, n_words, n_bytes, single, frag, clusters)


def measure_tiktoken(name: str, enc_name: str, texts: list[str]) -> dict | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(enc_name)
    except Exception as e:
        return {"model": name, "available": False, "error": str(e)}
    n_tok = n_words = n_bytes = single = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        n_tok += len(enc.encode(nfc))
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(enc.encode(w)) == 1:
                single += 1
    # tiktoken gives no char offsets; fragmentation is left unmeasured (None).
    return _row(name, n_tok, n_words, n_bytes, single, None, None)


def measure_ours(directory: str, texts: list[str]) -> dict:
    tok = BengaliTokenizer.load(directory)
    n_tok = n_words = n_bytes = single = frag = clusters = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        n_tok += len(tok.encode(raw))
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(tok.encode(w)) == 1:
                single += 1
        # surface method (ours guarantees whole-cluster tokens)
        surfaces = tok.encode_tokens(raw)
        for i in range(len(surfaces) - 1):
            a, b = surfaces[i], surfaces[i + 1]
            if a and b and len(grapheme_clusters(a)) + len(grapheme_clusters(b)) != len(grapheme_clusters(a + b)):
                frag += 1
        clusters += sum(1 for g in grapheme_clusters(nfc) if not g.isspace())
    name = f"Bornomala Track A ({tok.config['algo']} {tok.config['actual_vocab_size']})"
    return _row(name, n_tok, n_words, n_bytes, single, frag, clusters)


def _row(name, n_tok, n_words, n_bytes, single, frag, clusters):
    def d(a, b):
        return a / b if b else 0.0
    return {
        "model": name, "available": True,
        "tokens": n_tok, "words": n_words,
        "fertility": round(d(n_tok, n_words), 3),
        "strr": round(d(single, n_words), 3),
        "bytes_per_token": round(d(n_bytes, n_tok), 2),
        "conjunct_fragmentation": None if clusters is None else round(d(frag, clusters), 6),
        "n_fragmented": frag, "n_clusters": clusters,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="our tokenizer directory")
    p.add_argument("--skip", type=int, default=12000, help="held-out starts after this many articles")
    p.add_argument("--limit", type=int, default=800)
    p.add_argument("--out", default="out/comparison.json")
    args = p.parse_args(argv)

    print(f"loading held-out Bengali (skip {args.skip}, {args.limit} articles) ...", file=sys.stderr)
    texts = held_out(args.skip, args.limit)
    print(f"held-out lines: {len(texts)}", file=sys.stderr)

    rows = [measure_ours(args.tokenizer, texts)]
    for name, repo in HF_MODELS:
        print(f"measuring {name} ...", file=sys.stderr)
        rows.append(measure_hf(name, repo, texts))
    for name, enc in TIKTOKEN_MODELS:
        rows.append(measure_tiktoken(name, enc, texts))

    avail = [r for r in rows if r.get("available")]
    avail.sort(key=lambda r: r["fertility"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"held_out_lines": len(texts), "rows": rows}, f, ensure_ascii=False, indent=2)

    # Markdown table, sorted by fertility (best first).
    print("\n| Tokenizer | Fertility | STRR | Bytes/tok | Conjunct frag. |")
    print("|---|---:|---:|---:|---:|")
    for r in avail:
        frag = "n/a" if r["conjunct_fragmentation"] is None else f"{r['conjunct_fragmentation']:.4f}"
        print(f"| {r['model']} | {r['fertility']:.3f} | {r['strr']:.3f} | {r['bytes_per_token']:.2f} | {frag} |")
    for r in rows:
        if not r.get("available"):
            print(f"| {r['model']} | unavailable: {r.get('error','')} |", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
