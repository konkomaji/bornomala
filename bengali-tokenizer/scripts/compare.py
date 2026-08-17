r"""
Cross-tokenizer Bengali comparison (the showcase table).

Measures the Bornomala Track A tokenizer against the incumbents on the SAME
held-out Bengali text, with one consistent definition of every metric:

  fertility          tokens / whitespace-words   (lower is better)
  strr               fraction of words kept as one token
  bytes_per_token    UTF-8 bytes / tokens
  conjunct_fragmentation
                     LEGACY, kept for continuity with every already-published
                     number: fraction of grapheme clusters that a token boundary
                     splits, computed from each tokenizer's own character offsets.
                     Binary (any split counts the same) and its denominator
                     includes clusters that could never be split - both defects
                     are documented in bntok/fragmentation.py.
  destructive_rate   HEADLINE, the corrected replacement: graded via
  any_split_rate     bntok.fragmentation.count_splits_from_offsets, denominator
                     restricted to clusters that COULD be split, and splits
                     classified by what they actually sever (a stranded virama
                     or detached nukta vs. a harmless consonant-cluster/matra
                     seam) rather than counted as one undifferentiated bucket.
                     destructive_rate is what "conjunct fragmentation" was
                     always meant to measure. None where no character offsets
                     were available (tiktoken has none at all).

The whitepaper notes that Bengali does not appear in a single published
cross-tokenizer fertility comparison (spec section 4.1). This produces exactly
that comparison, from real tokenizers, reproducibly. Nothing is fabricated: a
tokenizer that cannot be loaded is reported as unavailable, not estimated.

Held-out set: Bengali Wikipedia articles AFTER the training range, so the
Bornomala numbers are on unseen text, an honest comparison. The current
official artifact (artifacts/bn-bpe-64k) trains on the first 15,000 Wikipedia
articles (bntok.corpus.WIKIPEDIA_TRAIN_ARTICLES), so --skip must be >= 15000
for it; --register selects a non-Wikipedia held-out slice instead (see
bntok.corpus.build_register_held_out).

Usage:
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --skip 15000 --limit 800
  python scripts/compare.py --tokenizer artifacts/bn-bpe-64k --register literary_formal --limit 1000

Pass --bmbt-tokenizer to also measure BMBT (Bornomala's Bengali Tokenizer,
v2 roadmap step 5, bntok.bmbt.BMBT). Unlike the raw akshara-parser row below
(measure_akshara, pre-vocabulary chunk counts, a deliberately different kind
of number), a trained BMBT artifact has a real vocabulary and real merges
the same way bn-bpe-64k does, so its row is a genuine like-for-like fertility
comparison and is sorted into the same table, not kept separate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bntok import BengaliTokenizer, normalize
from bntok.akshara import aksharas
from bntok.fragmentation import count_splits_from_offsets
from bntok.graphemes import grapheme_clusters

# HF tokenizers to compare against (all load without auth).
#
# Two named baselines from the v2 design doc's own roadmap step 4 could not
# be added, honestly rather than faked:
#   - IndicSuperTokenizer (Rana, Menezes et al., Krutrim AI, arXiv:2511.03237):
#     no public code/tokenizer release found (checked the arXiv abstract page
#     directly for a Hugging Face/GitHub link; none listed).
#   - BengaliBPE (Patwary & Noman, 2025, arXiv:2511.05324): the only
#     similarly-named public HF repo found (spitfire4794/bengali_bpe_tokenizer_hf)
#     fails to load (custom tokenizer class not supported by `transformers`)
#     and is not verifiably the paper's own artifact, so it is not used as a
#     stand-in.
#
# 2026-08: added India competitor + global frontier rows, verified loadable
# (or honestly reported unavailable, never faked) directly against
# transformers.AutoTokenizer before being added here - see
# docs/known-issues.md for the checked-but-excluded list (Hanooman/SML: no
# public HF tokenizer repo found as of this check; Gemini: no open tokenizer
# release, Gemma family used as the closest open proxy).
#
# 2026-08-18: Gemma-2 access request accepted on this project's HF account -
# loads and measures fine now (fertility 3.732 on FLORES+, far behind ours).
# This row still requires a logged-in huggingface_hub session with that
# account's access grant, so a CI run or a different machine/account without
# it will correctly report this row as unavailable rather than fail loudly -
# expected, not a regression.
#
# 2026-08-16: BrahmicTokenizer-131K added. It is the third and last baseline
# named in the whitepaper's own Gate G2 list, and unlike IndicSuperTokenizer
# and BengaliBPE above it does have a real, loadable public release
# (theschoolofai/BrahmicTokenizer-131K, Apache-2.0, arXiv:2605.29379): a
# PreTrainedTokenizerFast with working offset_mapping, so it gets a full row
# including the conjunct-fragmentation column, not a partial one. It had
# simply never been attempted before this date, which is a coverage gap in
# our own benchmark, not an availability problem on their side.
# Note the vocabulary asymmetry runs in BOTH directions and is stated in
# benchmarks/bengali-comparison.md rather than left for a reader to notice:
# BrahmicTokenizer has 131,072 tokens to our 64,000, but spreads them across
# 12 Brahmic-script languages, where ours are spent entirely on Bengali.
#
# 2026-08-17: BanglaBERT and BanglaT5 (csebuetnlp, BUET CSE NLP group) added.
# Both are Bengali-monolingual models, not multilingual, so this is the first
# monolingual-vs-monolingual comparison in the table rather than a monolingual
# vs multilingual one - a genuinely harder bar. Verified loadable directly:
# csebuetnlp/banglabert (ElectraTokenizerFast, 32k vocab) and
# csebuetnlp/banglat5 (T5TokenizerFast, 32,100 vocab).
HF_MODELS = [
    ("Sarvam-1 (Sarvam AI)", "sarvamai/sarvam-1"),
    ("SUTRA (TWO AI)", "TWO/sutra-mlt256-v2"),
    ("BrahmicTokenizer-131K (TSAI)", "theschoolofai/BrahmicTokenizer-131K"),
    ("Krutrim (Krutrim AI)", "krutrim-ai-labs/Krutrim-2-instruct"),
    ("Param2-17B (BharatGen)", "bharatgenai/Param2-17B-A2.4B-Thinking"),
    ("IndicBERTv2 (AI4Bharat)", "ai4bharat/IndicBERTv2-MLM-only"),
    ("BanglaBERT (csebuetnlp)", "csebuetnlp/banglabert"),
    ("BanglaT5 (csebuetnlp)", "csebuetnlp/banglat5"),
    ("mBERT (Google)", "google-bert/bert-base-multilingual-cased"),
    ("XLM-RoBERTa (Meta)", "FacebookAI/xlm-roberta-base"),
    ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
    ("Llama-3.1 (Meta)", "meta-llama/Llama-3.1-8B"),
    ("Gemma-2 (Google)", "google/gemma-2-9b"),
    ("Mistral-7B (Mistral AI)", "mistralai/Mistral-7B-v0.3"),
    ("Qwen2.5 (Alibaba)", "Qwen/Qwen2.5-7B"),
]
TIKTOKEN_MODELS = [
    ("GPT-4o (OpenAI o200k)", "o200k_base"),
    ("GPT-4 (OpenAI cl100k)", "cl100k_base"),
]


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


REGISTER_HELD_OUT_SOURCES = {"literary_formal", "general_web", "news", "banglish", "flores"}


def register_held_out(register: str, limit_docs: int) -> list[str]:
    """Held-out text for a non-Wikipedia register.

    literary_formal/general_web/news use bntok.corpus.build_register_held_out,
    which reads Sangraha/XL-Sum starting exactly where build_configured_corpus's
    training budget ends, disjoint from training by construction. banglish and
    flores are different in kind: banglish is romanized-script CC-100 bn_rom,
    flores is FLORES+'s professionally translated parallel sentences - neither
    is used in any training config in this repository, so there is no training
    range to stay disjoint from and bntok.corpus's own held-out builders are
    used directly. `limit_docs` is ignored for flores (it has a fixed size,
    ~2000 sentences across dev+devtest - the point is a same-corpus comparison
    against a specific external paper's own methodology, not a scalable slice).
    """
    log = lambda m: print(m, file=sys.stderr)
    if register == "banglish":
        from bntok.corpus import build_banglish_held_out
        return build_banglish_held_out(limit_lines=limit_docs, log=log)
    if register == "flores":
        from bntok.corpus import build_flores_held_out
        return build_flores_held_out(log=log)
    from bntok.corpus import build_register_held_out
    slices = build_register_held_out(limit_docs=limit_docs, log=log)
    return slices[register]


def _frag_from_offsets(nfc: str, offsets: list[tuple[int, int]]) -> tuple[int, int]:
    """Count grapheme clusters split by any token boundary, using char offsets.

    This is the LEGACY binary counter (any split counts the same, and the
    denominator includes clusters that could never be split - see
    bntok/fragmentation.py's own docstring for both defects). Kept unchanged,
    not replaced, so every already-published `conjunct_fragmentation` number
    stays comparable; `_grade_from_offsets` below is the corrected measure,
    reported alongside it, not instead of it.
    """
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


def _grade_from_offsets(nfc: str, offsets: list[tuple[int, int]]):
    """Graded fragmentation (bntok.fragmentation.count_splits_from_offsets),
    fed the same union-of-cut-points every measure_* function already builds
    for the legacy counter above - a cut is a cut whether an offset mapping
    reports it as a token's start or a token's end, and different
    tokenizers are not consistent about which they report.
    """
    boundaries = set()
    for s, e in offsets:
        boundaries.add(s)
        boundaries.add(e)
    return count_splits_from_offsets(nfc, boundaries)


def measure_hf(name: str, repo: str, texts: list[str]) -> dict | None:
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(repo, use_fast=True)
    except Exception as e:  # noqa: BLE001 - report ANY load failure as "unavailable", not estimated
        return {"model": name, "available": False, "error": f"{type(e).__name__}: {e}"}

    n_tok = n_words = n_bytes = single = frag = clusters = 0
    n_destructive = n_modifier = n_onset_rime = n_splittable = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        try:
            enc = tk(nfc, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offs = enc.get("offset_mapping") or []
        except Exception:  # noqa: BLE001 - some tokenizers don't support offset_mapping; fall back
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
            g = _grade_from_offsets(nfc, offs)
            n_destructive += g.destructive
            n_modifier += g.modifier
            n_onset_rime += g.onset_rime
            n_splittable += g.splittable_clusters

    if clusters == 0:
        return _row(name, n_tok, n_words, n_bytes, single, None, None)
    return _row(name, n_tok, n_words, n_bytes, single, frag, clusters,
                n_destructive, n_modifier, n_onset_rime, n_splittable)


def measure_tiktoken(name: str, enc_name: str, texts: list[str]) -> dict | None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(enc_name)
    except Exception as e:  # noqa: BLE001 - report ANY load failure as "unavailable", not estimated
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
    """Measure our tokenizer the same way the HF path is measured: a token
    boundary counts as fragmentation only if it falls strictly inside a
    grapheme cluster's own codepoint span (via _frag_from_offsets), not
    whenever two individually-whole-cluster atoms simply go unmerged. An
    earlier version of this function used a pairwise cluster-count heuristic
    (compare len(clusters(a)) + len(clusters(b)) to len(clusters(a+b))) that
    produced false positives for exactly that "unmerged but intact" case, for
    example a lone consonant token followed by a lone vowel-sign token: both
    are complete, correctly-decoding atoms, so nothing was actually split, but
    the heuristic could not tell "adjacent complete clusters" from "one
    cluster's codepoints torn across a boundary." Offsets, derived from
    cumulative surface-token lengths (the surfaces reconstruct the normalised
    text exactly, per the round-trip guarantee), do not have that ambiguity.

    Two subtleties cost real debugging time and are worth recording:

    1. The Metaspace pre-tokenizer prefixes the FIRST token's surface with its
       word-boundary marker (a leading space) that is not present in `nfc`
       itself (this is what `BengaliTokenizer.decode` strips via
       `.strip(" ")`). Naively joining raw surfaces and using cumulative
       lengths as offsets is therefore off by one space for every position
       after the first token, which manufactures thousands of false
       "fragmentation" hits. Corrected by trimming the same leading space here
       before computing offsets.

    2. `encode_tokens()` is a human-readable DEBUG view, not a guaranteed exact
       reconstruction: for a codepoint genuinely outside this tokenizer's
       coverage (foreign scripts quoted inside Bengali text -- Greek, Arabic,
       Japanese all occur in this corpus's Wikipedia/news held-out sets), the
       underlying BPE model emits the literal `<unk>` special token, and
       `encode_tokens()`'s fallback (`readable if readable else t`) returns
       that literal 5-character string in place of the missing text rather
       than an empty placeholder. This means surface concatenation can silently
       diverge from `nfc` on such lines. Foreign-script coverage is already an
       out-of-scope, documented limitation (`docs/known-issues.md` point 4),
       not a fragmentation question, so lines that do not cleanly round-trip
       are skipped for fragmentation purposes specifically (fertility/STRR/
       bytes still count them: those metrics don't depend on surface
       reconstruction). An assertion on the lines that remain still holds, so
       a genuine offset bug on in-scope text cannot silently corrupt the count
       again.
    """
    tok = BengaliTokenizer.load(directory)
    n_tok = n_words = n_bytes = single = frag = clusters = 0
    n_destructive = n_modifier = n_onset_rime = n_splittable = 0
    skipped_for_frag = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        n_tok += len(tok.encode(raw))
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(tok.encode(w)) == 1:
                single += 1
        if not tok.roundtrip_ok(raw):
            skipped_for_frag += 1
            continue
        surfaces = tok.encode_tokens(raw)
        joined = "".join(surfaces)
        trimmed = joined.lstrip(" ")
        lead = len(joined) - len(trimmed)
        assert trimmed == nfc, (
            f"surface reconstruction mismatch on a round-trippable line: "
            f"{trimmed!r} != {nfc!r} (this would silently corrupt the fragmentation count)"
        )
        offsets = []
        pos = -lead
        for s in surfaces:
            offsets.append((pos, pos + len(s)))
            pos += len(s)
        f, c = _frag_from_offsets(nfc, offsets)
        frag += f
        clusters += c
        g = _grade_from_offsets(nfc, offsets)
        n_destructive += g.destructive
        n_modifier += g.modifier
        n_onset_rime += g.onset_rime
        n_splittable += g.splittable_clusters
    if skipped_for_frag:
        print(f"  ({skipped_for_frag}/{len(texts)} lines skipped for fragmentation: "
              f"out-of-coverage codepoints, see docs/known-issues.md point 4)", file=sys.stderr)
    name = f"Bornomala Track A ({tok.config['algo']} {tok.config['actual_vocab_size']})"
    return _row(name, n_tok, n_words, n_bytes, single, frag, clusters,
                n_destructive, n_modifier, n_onset_rime, n_splittable)


def measure_bmbt(directory: str, texts: list[str]) -> dict:
    """Measure BMBT (bntok.bmbt.BMBT) the same way measure_ours() measures
    bn-bpe-64k: a genuine like-for-like row, since a trained BMBT artifact
    has a real vocabulary and real merges, unlike measure_akshara()'s raw,
    pre-vocabulary chunk count. Structurally identical to measure_ours(),
    down to reusing the same offset-based fragmentation logic - see that
    function's docstring for why offsets (not a pairwise cluster-count
    heuristic) are the correct way to detect a split conjunct, and why
    lines that do not cleanly round-trip are skipped for fragmentation
    specifically (out-of-coverage codepoints), not for fertility/STRR/bytes.
    """
    from bntok.bmbt import BMBT
    tok = BMBT.load(directory)
    n_tok = n_words = n_bytes = single = frag = clusters = 0
    n_destructive = n_modifier = n_onset_rime = n_splittable = 0
    skipped_for_frag = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        n_tok += len(tok.encode(raw))
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(tok.encode(w)) == 1:
                single += 1
        if not tok.roundtrip_ok(raw):
            skipped_for_frag += 1
            continue
        surfaces = tok.encode_tokens(raw)
        joined = "".join(surfaces)
        trimmed = joined.lstrip(" ")
        lead = len(joined) - len(trimmed)
        assert trimmed == nfc, (
            f"surface reconstruction mismatch on a round-trippable line: "
            f"{trimmed!r} != {nfc!r} (this would silently corrupt the fragmentation count)"
        )
        offsets = []
        pos = -lead
        for s in surfaces:
            offsets.append((pos, pos + len(s)))
            pos += len(s)
        f, c = _frag_from_offsets(nfc, offsets)
        frag += f
        clusters += c
        g = _grade_from_offsets(nfc, offsets)
        n_destructive += g.destructive
        n_modifier += g.modifier
        n_onset_rime += g.onset_rime
        n_splittable += g.splittable_clusters
    if skipped_for_frag:
        print(f"  ({skipped_for_frag}/{len(texts)} lines skipped for fragmentation: "
              f"out-of-coverage codepoints, see docs/known-issues.md point 4)", file=sys.stderr)
    name = f"Bornomala BMBT ({tok.config['algo']} {tok.config['actual_vocab_size']})"
    return _row(name, n_tok, n_words, n_bytes, single, frag, clusters,
                n_destructive, n_modifier, n_onset_rime, n_splittable)


def measure_akshara(texts: list[str]) -> dict:
    """Measure the v2 akshara finite-state parser (roadmap step 4).

    This is NOT a like-for-like comparison with the tokenizer rows above, and
    is reported separately, not folded into the fertility-sorted table: the
    parser has no vocabulary and no merges yet (v2 roadmap step 5, featural
    encoding + statistical fallback, is not built), so its chunk count is the
    PRE-compression granularity, the same kind of number as grapheme-cluster
    count, not the POST-BPE-merge token count every tokenizer row above
    reports. Expect it to need more chunks per word than any trained
    tokenizer; that is not a regression, it is what "no compression yet"
    means, and is exactly the number the design doc's own roadmap (step 4)
    asks to be measured and reported honestly before building further.

    Conjunct fragmentation is the one metric that IS a fair, like-for-like
    comparison: `Akshara.start`/`.end` are exact codepoint offsets carried by
    the parser itself (no surface reconstruction needed, unlike the BPE rows
    above), fed straight into the same `_frag_from_offsets` used for
    everything else.
    """
    n_tok = n_words = n_bytes = single = frag = clusters = 0
    n_destructive = n_modifier = n_onset_rime = n_splittable = 0
    for raw in texts:
        nfc = normalize(raw)
        words = nfc.split()
        chunks = aksharas(nfc)
        n_tok += len(chunks)
        n_words += len(words)
        n_bytes += len(nfc.encode("utf-8"))
        for w in words:
            if len(aksharas(w)) == 1:
                single += 1
        offsets = [(c.start, c.end) for c in chunks]
        f, c = _frag_from_offsets(nfc, offsets)
        frag += f
        clusters += c
        g = _grade_from_offsets(nfc, offsets)
        n_destructive += g.destructive
        n_modifier += g.modifier
        n_onset_rime += g.onset_rime
        n_splittable += g.splittable_clusters
    return _row("Bornomala v2 akshara parser (pre-vocabulary, no merges)",
                n_tok, n_words, n_bytes, single, frag, clusters,
                n_destructive, n_modifier, n_onset_rime, n_splittable)


def _row(name, n_tok, n_words, n_bytes, single, frag, clusters,
         n_destructive=None, n_modifier=None, n_onset_rime=None, n_splittable=None):
    def d(a, b):
        return a / b if b else 0.0
    row = {
        "model": name, "available": True,
        "tokens": n_tok, "words": n_words,
        "fertility": round(d(n_tok, n_words), 3),
        "strr": round(d(single, n_words), 3),
        "bytes_per_token": round(d(n_bytes, n_tok), 2),
        "conjunct_fragmentation": None if clusters is None else round(d(frag, clusters), 6),
        "n_fragmented": frag, "n_clusters": clusters,
        # Graded replacement (bntok/fragmentation.py): destructive_rate is the
        # HEADLINE ("conjunct fragmentation" was always meant to say this),
        # any_split_rate is the corrected-denominator counterpart of the
        # legacy field above. None when no offsets were available (tiktoken).
        "destructive_rate": None, "any_split_rate": None,
        "n_destructive": n_destructive, "n_modifier": n_modifier,
        "n_onset_rime": n_onset_rime, "splittable_clusters": n_splittable,
    }
    if n_splittable is not None:
        row["destructive_rate"] = round(d(n_destructive, n_splittable), 6)
        row["any_split_rate"] = round(d(n_destructive + n_modifier + n_onset_rime, n_splittable), 6)
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="our tokenizer directory")
    p.add_argument("--skip", type=int, default=15000, help="held-out starts after this many articles (wikipedia source only)")
    p.add_argument("--limit", type=int, default=800)
    p.add_argument("--register", choices=sorted(REGISTER_HELD_OUT_SOURCES),
                    help="use a non-Wikipedia held-out register instead (see bntok.corpus.build_register_held_out)")
    p.add_argument("--out", default="out/comparison.json")
    p.add_argument("--bmbt-tokenizer", dest="bmbt_tokenizer",
                    help="also measure a trained BMBT directory (bntok.bmbt.BMBT) as a normal row")
    args = p.parse_args(argv)

    if args.register:
        print(f"loading held-out register '{args.register}' ({args.limit} docs) ...", file=sys.stderr)
        texts = register_held_out(args.register, args.limit)
    else:
        print(f"loading held-out Bengali (skip {args.skip}, {args.limit} articles) ...", file=sys.stderr)
        texts = held_out(args.skip, args.limit)
    print(f"held-out lines: {len(texts)}", file=sys.stderr)

    rows = [measure_ours(args.tokenizer, texts)]
    if args.bmbt_tokenizer:
        print("measuring BMBT ...", file=sys.stderr)
        rows.append(measure_bmbt(args.bmbt_tokenizer, texts))
    for name, repo in HF_MODELS:
        print(f"measuring {name} ...", file=sys.stderr)
        rows.append(measure_hf(name, repo, texts))
    for name, enc in TIKTOKEN_MODELS:
        rows.append(measure_tiktoken(name, enc, texts))

    print("measuring v2 akshara parser (pre-vocabulary, no merges) ...", file=sys.stderr)
    akshara_row = measure_akshara(texts)

    avail = [r for r in rows if r.get("available")]
    avail.sort(key=lambda r: r["fertility"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"held_out_lines": len(texts), "rows": rows, "akshara_v2": akshara_row},
                   f, ensure_ascii=False, indent=2)

    # Markdown table, sorted by fertility (best first). Both fragmentation
    # measures shown, per bntok/fragmentation.py's own instruction: reporting
    # only the legacy binary field would hide the graded cost/benefit split
    # (e.g. a morphology-enabled artifact's onset/rime splits by design);
    # reporting only destructive_rate would silently change what every
    # already-published "conjunct fragmentation" number means.
    print("\n| Tokenizer | Fertility | STRR | Bytes/tok | Conjunct frag. (legacy) | Destructive rate |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in avail:
        frag = "n/a" if r["conjunct_fragmentation"] is None else f"{r['conjunct_fragmentation']:.4f}"
        destr = "n/a" if r["destructive_rate"] is None else f"{r['destructive_rate']:.4f}"
        print(f"| {r['model']} | {r['fertility']:.3f} | {r['strr']:.3f} | {r['bytes_per_token']:.2f} | {frag} | {destr} |")
    for r in rows:
        if not r.get("available"):
            print(f"| {r['model']} | unavailable: {r.get('error','')} |", file=sys.stderr)

    print("\nv2 akshara parser (roadmap step 4, NOT a like-for-like row above -- "
          "no vocabulary or merges yet, see measure_akshara()'s docstring):")
    print(f"| {akshara_row['model']} | {akshara_row['fertility']:.3f} | {akshara_row['strr']:.3f} | "
          f"{akshara_row['bytes_per_token']:.2f} | {akshara_row['conjunct_fragmentation']:.4f} | "
          f"{akshara_row['destructive_rate']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
