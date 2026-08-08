# Known issues, limitations, and the bugs we hit

Transparency is a core value of Project Bornomala. This document records the real
limitations of the current tokenizer, the caveats on the comparison, and the bugs
found and fixed during development. Nothing here is hidden.

## Current limitations

1. **Rare-cluster fragmentation is near zero, not exactly zero.** Grapheme
   clusters seen often enough in the corpus (at or above the atom-frequency
   threshold, `min_atom_freq`, default 2) get their own atom and can never be
   split: fragmentation is exactly 0 for them. Clusters below that threshold
   decompose into their codepoints, and a token boundary can fall between those
   codepoints. It is a deliberate efficiency trade, and it is measured and
   reported per-register, not hidden (see point 7). Setting `min_atom_freq=1`
   drives it to exactly 0 for all seen clusters at the cost of a larger atom
   set.

2. **Decode normalises surrounding whitespace.** The word-boundary scheme
   (Metaspace) does not carry exact leading or trailing spaces, so decode returns
   content-exact text with surrounding spaces trimmed. Internal single spaces are
   preserved. This matches how subword tokenizers normally behave; use
   `content_roundtrip_ok` for whitespace-insensitive fidelity checks.

3. **v0.1 was a demonstrator, trained on Wikipedia only.** It was trained on
   12,000 Bengali Wikipedia articles and is no longer shipped (superseded by
   `artifacts/bn-bpe-64k`, kept in git history and on Hugging Face for the
   record). Wikipedia is not weighted toward literary and formal register,
   which is where Bengali conjunct density is highest. At matched (32k) vocab
   size, v0.1 measured a sharper fertility than the first literary-weighted
   attempt on pure Wikipedia text (1.568 vs 1.952) — the same fixed token
   budget spread across four registers instead of one is sharper nowhere;
   see point 7 for how vocab size resolved this.

4. **Exotic codepoints outside Bengali, its shared punctuation, and ASCII may not
   round-trip.** The guaranteed coverage set is the Bengali block, Bengali's two
   shared (non-Bengali-block) sentence punctuation marks (danda U+0964 and double
   danda U+0965, see `docs/bengali-script-reference.md` §4), plus ASCII. This
   covers Bengali and code-mixed English. Emoji or other scripts can map to the
   unknown atom and be dropped on decode. That is out of scope for a
   Bengali-first tokenizer, but it is a real limit, so it is stated.

5. **Unigram support depends on the tokenizers version.** BPE is the default and
   the tested path. Unigram training uses the same atom scheme but relies on the
   installed `tokenizers` version accepting `initial_alphabet` for the Unigram
   trainer.

6. **The literary-weighted corpus substitutes real datasets for two configured
   sources that do not exist publicly.** `government_administrative` and
   `code_mixed_bn_en` have no clean, licensable public Bengali dataset as of
   2026-07, so they are dropped and the remaining four sources' weights are
   renormalised (see `bntok.corpus.build_configured_corpus`), rather than faked
   with a substitute. `public_domain_literature` is itself a compromise: genuine
   Wikisource Bengali public-domain text exists but is tiny (about 90 lines in
   the current Wikimedia snapshot), so it is combined with Sangraha
   verified/ben pdf-typed documents, OCR-noise-filtered
   (`is_clean_bengali_line`), as a formal/book-register proxy. Those PDFs are
   genuinely old-orthography Bengali (spot-checked: 19th/20th-century novel
   prose, a Mahabharata translation) but are not confirmed pre-1950 or public
   domain, only formal/archaic in register. `sangraha_verified_bn` uses
   Sangraha's web-typed documents, and `contemporary_news` uses XL-Sum Bengali
   (about 10k BBC Bangla articles total; training uses the first 8k, the
   remaining ~2k are held out for evaluation, see point 8). Any number reported
   from this corpus should name it as the "literary-weighted corpus" and link
   back to this note.

7. **Vocabulary size matters more than corpus mix alone.** At the original 32k
   vocab size, the literary-weighted corpus measured *worse* fertility than the
   Wikipedia-only v0.1 on Wikipedia held-out text (point 3) — the same 32k
   token budget spread across four registers instead of one is sharper
   nowhere. An ablation across 32k/48k/64k (BPE, same corpus, same weights)
   showed fertility recovering monotonically with vocab size (in-sample: 1.482
   → 1.375 → 1.319). At 64k (`artifacts/bn-bpe-64k`, the current shipped
   artifact) the tokenizer beats IndicBERTv2 on fertility, STRR, *and* conjunct
   fragmentation on every one of the four held-out registers tested
   (Wikipedia, literary/formal via Sangraha pdf-typed, general web via
   Sangraha web-typed, and news via XL-Sum): fertility 1.524/1.320/1.201/1.140
   vs IndicBERTv2's 1.652/1.612/1.395/1.312; conjunct fragmentation
   0.0001/0.0001/0.0001/0.0000 vs IndicBERTv2's 0.0440/0.0562/0.0277/0.0206 —
   roughly 200x to 560x lower on every register, not just Wikipedia. Full
   tables: `benchmarks/bengali-comparison.md`. An earlier pass through this
   comparison, before the measurement bugs in point 8 were found and fixed,
   incorrectly showed fragmentation as a mixed result (better on Wikipedia/
   literary, slightly worse on general web/news); that was entirely a
   measurement artifact, not a real property of the tokenizer.

8. **Held-out reservation for non-Wikipedia registers.** `bntok.corpus.
   build_register_held_out` reserves a disjoint tail of Sangraha pdf-typed,
   Sangraha web-typed, and XL-Sum documents (starting exactly where training's
   document budget ends: see `SANGRAHA_PDF_TRAIN_DOCS`, `SANGRAHA_WEB_TRAIN_DOCS`,
   `XLSUM_TRAIN_DOCS` in `bntok/corpus.py`) so register-specific evaluation never
   touches training documents. Wikisource is not included in held-out: its ~90
   lines are entirely consumed by training and too small to split meaningfully.

## Caveats on the comparison

- **IndicBERT v1 is gated.** AI4Bharat's `ai4bharat/indic-bert` now requires
  authentication, so the comparison uses the ungated `ai4bharat/IndicBERTv2-MLM-only`.
  Both are AI4Bharat tokenizers.
- **GPT-4o fragmentation is unmeasured.** tiktoken exposes no character offsets,
  so conjunct fragmentation cannot be computed for it. It is reported as n/a
  rather than guessed.
- **Vocabulary sizes differ.** Our best tokenizer is 64k; Sarvam-1 is about 68k;
  IndicBERTv2 and the other encoders are smaller. Fertility (tokens per word) is
  a fair, vocab-agnostic per-word measure, but the difference is noted rather
  than hidden.
- **Held-out sets are register-specific and disjoint from training** (point 8).
  Wikipedia held-out uses `scripts/compare.py --skip`; the other three registers
  use `scripts/compare.py --register {literary_formal,general_web,news}`.
- **A small number of held-out lines are excluded from our tokenizer's
  fragmentation count specifically** (point 8): lines quoting genuinely
  out-of-coverage foreign-script text (point 4). At most 11 of up to 28,461
  lines per register. Fertility, STRR, and bytes/token still count these lines
  normally; only the fragmentation denominator/numerator exclude them.

## Bugs found and fixed during development

Kept as an honest record of what went wrong and how it was resolved.

1. **Whitespace was being atomised, breaking word boundaries.** Early on, spaces
   were mapped to atoms like any other character, so the Metaspace word-boundary
   marker never fired and every word ran together. Fixed by keeping whitespace
   literal through the atom layer.

2. **Round-trip failed on a trailing space (227 of 1500 lines).** Decode produced
   the exact Bengali content plus one spurious trailing space, a Metaspace
   artifact, which failed a strict equality check. The Bengali content was always
   correct. Fixed by trimming surrounding spaces on decode, which is standard for
   subword tokenizers.

3. **Unseen codepoints were dropped in a tiny corpus.** With a small training
   corpus, a vowel sign absent from it had no atom and decoded to nothing. Fixed
   by guaranteeing an atom for the whole Bengali block and ASCII and forcing every
   atom into the base vocabulary, so any Bengali or code-mixed text round-trips
   regardless of the corpus.

4. **Vocabulary-size floor.** After guaranteeing coverage, a requested vocabulary
   smaller than the atom alphabet is impossible. This now raises a clear
   `VocabSizeError` with guidance, rather than failing deep in the trainer.

5. **Tooling gaps surfaced honestly.** The comparison first ran without
   `transformers` installed and reported every Hugging Face tokenizer as
   unavailable rather than silently producing a partial table; installing the
   dependency and rerunning gave the full comparison. This is the intended
   behaviour: unavailable means unavailable, never estimated.

6. **Bengali's own sentence punctuation was outside the guaranteed-coverage
   set.** Danda and double danda ("।"/"॥") are shared Devanagari-block
   codepoints (U+0964/U+0965), not part of the Bengali block, so they were
   missing from `GUARANTEED_CODEPOINTS`. In practice this never visibly failed
   (danda is far too frequent in any real corpus to miss the frequency
   threshold), but the codebase's own round-trip guarantee did not structurally
   cover it. Fixed in `graphemes.py`. See `docs/bengali-script-reference.md` §4.

7. **A normalisation docstring claimed something NFC does not do, and the first
   fix attempt was a silent no-op.** `ড়`/`ঢ়`/`য়` have two valid spellings
   (precomposed singleton vs. decomposed base+nukta); we initially assumed NFC
   left real corpora inconsistently encoded between the two. Verified against
   `unicodedata` directly: NFC already unifies both spellings, just onto the
   decomposed form (a permanent Unicode "composition exclusion"), so there was
   no actual cross-source inconsistency bug. The fix now shipped
   (re-composing to the single dedicated codepoint) is a minor efficiency
   improvement, not a correctness fix. The first attempt at even that fix was
   itself a no-op: typing the Bengali literals directly into the source let an
   intermediate tool/editor layer silently re-decompose them before saving, so
   the "singleton" values in the replacement dict were still decomposed
   sequences. Rebuilt using explicit `chr(0x09DC)`-style codepoints instead of
   literal characters. Full writeup in `docs/bengali-script-reference.md` §3
   and §5.

8. **`scripts/compare.py`'s fragmentation counter for our own tokenizer went
   through three versions before it was actually correct** — kept in full since
   two of the three bugs briefly produced numbers that were reported (and
   corrected) mid-session, and the point of this document is not to have that
   quietly disappear.
   - *v1 (heuristic, wrong):* compared `len(clusters(token_a)) +
     len(clusters(token_b))` against `len(clusters(token_a + token_b))` for
     every adjacent token pair. This over-counts: two adjacent tokens that are
     each a complete, correctly-decoding grapheme cluster on their own (a
     consonant atom followed by a separate vowel-sign atom that simply never
     got BPE-merged into one token) get flagged as "fragmentation" even though
     nothing was split at the codepoint level and round-trip is exact.
   - *v2 (offset-based, still wrong):* replaced with the same character-offset
     method used for the HF tokenizer comparisons (`_frag_from_offsets`),
     reconstructing offsets from cumulative surface-token lengths returned by
     `encode_tokens()`. Two bugs remained: (a) the Metaspace pre-tokenizer
     prefixes the *first* token's surface with a leading space that is not in
     the normalised text, so every offset after the first token was shifted by
     one, manufacturing thousands of false fragmentation hits; (b)
     `encode_tokens()` is a human-readable debug view, not a guaranteed exact
     reconstruction — for a codepoint genuinely outside this tokenizer's
     coverage (Greek, Arabic, and Japanese text quoted inside real Bengali
     Wikipedia/news articles all occur in this corpus), the BPE model emits
     the `<unk>` special token, and `encode_tokens()`'s fallback
     (`readable if readable else t`) returns the literal 5-character string
     `"<unk>"` in place of the missing text, silently desynchronising every
     offset after it on that line.
   - *v3 (current, verified correct):* trims the leading Metaspace space
     before computing offsets, and skips fragmentation counting (only) on
     lines that do not cleanly round-trip (`tok.roundtrip_ok(raw)` is False,
     i.e. the out-of-coverage case in point 4 — those lines still count
     normally toward fertility/STRR/bytes). A hard assertion checks the
     surface reconstruction matches the normalised text exactly on every line
     that remains, so a similar bug cannot silently corrupt a comparison
     again — it crashes instead. This version is what produced every number
     in `benchmarks/bengali-comparison.md`.

9. **One training run (32k, both the original `bn-bpe-32k-v2` and the later
   32k ablation point) was missing the atom for a very common cluster (`য়া`,
   YYA+AA-matra) that a controlled, corpus-fixed test proves has nothing to do
   with vocab size** (training two tokenizers from the identical in-memory
   corpus, varying only `vocab_size`, produces an identical atom map both
   times). The 48k and 64k runs, and multiple smaller-scale reproductions of
   the same corpus-building pipeline, all correctly assign this cluster its own
   atom (it appears thousands of times in even a 20k-line sample). The likely
   cause is run-to-run corpus-sampling non-determinism between separate
   `bntok train` invocations (each re-streams from Hugging Face Hub
   independently), though this was not pinned down to full certainty given the
   scale of investigation that would take. It does not affect the 64k
   tokenizer this project currently recommends. Flagged here rather than
   silently left unresolved.

10. **CC-100 is wired in as an available bulk general-web source, but is not
    part of the shipped 64k corpus weights.** `bntok.corpus.stream_cc100`
    streams CC-100 Bengali (`bn.txt.xz`, ~860MB compressed, 2018-vintage
    CommonCrawl text via the same pipeline behind XLM-R's training data;
    Wenzek et al., 2020) and `build_configured_corpus` accepts it as source
    name `cc100_general_web`. It is orders of magnitude larger than any
    current single source, but noisier and not literary-weighted, so adding
    it to `configs/bpe-64k.json`'s weights would require retraining and
    re-benchmarking before any claim about it changes. It is available now
    for a future ablation, not a silent change to what `bn-bpe-64k` was
    actually trained on.

11. **An independent vowel followed by a virama does not chain into a
    further consonant the way a consonant does.** Found by running the v2
    akshara parser (`bntok/akshara.py`) against real Wikipedia held-out
    text as its first roadmap-step-4 measurement, not a synthetic test.
    Unicode's Indic_Conjunct_Break (InCB) rule requires InCB=Consonant on
    both sides of the virama for a conjunct chain to continue; vowels are
    not InCB=Consonant, so `\X` (`regex`'s UAX #29 grapheme clustering)
    clusters "vowel+virama" as its own pair and starts fresh at whatever
    follows. An earlier version of the parser's grammar had no
    trailing-virama slot on the Vowel branch at all, so the bare vowel and
    the following virama ended up as two separately-fragmenting chunks.
    Fixed by giving the Vowel branch the same tail-scanning logic as
    Consonant, just with chain continuation permanently disabled.

12. **A Modifier blocks conjunct-chain continuation; Matra and Nukta do
    not.** Also found via the same held-out measurement. Verified
    empirically against `\X`: `consonant+matra+virama+consonant` clusters
    as one continuing conjunct, but `consonant+modifier+virama+consonant`
    clusters as two (the modifier and virama are absorbed into the first
    chunk as generic Extend characters, but the chain does not continue).
    An earlier version's grammar used a rigid `Matra? Modifier*` tail with
    no re-check for a continuing virama after either, so real
    repeated-matra sequences (found for real: a consonant followed by the
    same vowel sign three or four times in a row, likely OCR/encoding
    noise in the Wikipedia dump) and modifier-then-virama sequences both
    fragmented. Fixed by a unified tail scan (`_scan_tail` in
    `akshara.py`) that absorbs any mix or repetition of
    {Nukta, Matra, Modifier} and only re-checks for a continuing virama
    after the whole run, tracking whether a Modifier occurred in it.

13. **ZWJ and ZWNJ are not tied to a fixed position relative to the
    virama.** The first fix for point 12 above still assumed "virama, then
    optionally ZWJ or ZWNJ" (the design doc's own simplified ordering). A
    second held-out measurement pass (after fixing point 12) found real
    Wikipedia text containing `Consonant ZWJ Virama Consonant` sequences
    that `\X` clusters exactly like `Consonant Virama ZWJ Consonant` -
    continuing the conjunct either way - and a ZWNJ before the virama
    blocks chain continuation the same way a ZWNJ after it does. Fixed by
    folding ZWJ/ZWNJ into the same unified tail-scan run as point 12,
    order-agnostic, tracking only whether a ZWNJ (like a Modifier)
    occurred anywhere in the run. This third pass is what brought conjunct
    fragmentation on the Wikipedia held-out set to exactly 0.0000 (see
    `benchmarks/bengali-comparison.md`'s "v2 roadmap step 4" section); the
    first measurement, before any of points 11-13 were fixed, was 0.0012.

14. **A rare, deliberate scope boundary, not a bug: Bengali consonant
    immediately followed by a Devanagari combining mark.** Found when
    extending the step-4 measurement to the `literary_formal` held-out
    register (Sangraha pdf-typed, 19th/20th-century scanned literature
    including a Mahabharata translation, per point 6 above): 13 out of
    2,868,557 clusters (fragmentation 0.000005) are a Bengali consonant
    directly followed by U+093E/U+093F/U+0902 (DEVANAGARI VOWEL SIGN
    AA/I, DEVANAGARI SIGN ANUSVARA) - almost certainly OCR or font-mapping
    noise from scanning old books, given the two closely related Brahmic
    scripts, and not caught by `is_clean_bengali_line`'s Bengali/ASCII
    ratio filter since only 1-2 stray codepoints appear per otherwise-clean
    line. `\X` clusters these together anyway (Grapheme_Cluster_Break=
    Extend applies to any script's combining marks, not just Bengali's),
    but `akshara.py`'s `MATRAS`/`MODIFIERS` sets are, by design, Bengali
    Unicode block only (`substrate.py`) - this parser is a Bengali-script
    grammar, not an all-scripts one, so it correctly does not absorb a
    foreign-script combining mark into a Bengali consonant's chunk. This is
    left as-is rather than "fixed": widening the grammar to swallow
    arbitrary-script combining marks would blur what a Bengali akshara
    grammar even means, and would mask exactly the kind of encoding noise
    this measurement is useful for surfacing. General web and news
    registers measured exactly 0.0000 (no such contamination); see
    `benchmarks/bengali-comparison.md`.

15. **`stream_cc100` never actually worked until BMBT's training run tried
    to use it for real.** Point 10 above documents CC-100 being "wired in,"
    but the original implementation called
    `datasets.load_dataset("cc100", ...)`, which fails outright on any
    current `datasets` install (`RuntimeError: Dataset scripts are no
    longer supported, but found cc100.py` - the `cc100` Hub repository
    still ships a legacy loader script, and `datasets` dropped script
    support). This was never caught earlier because nothing had actually
    exercised `cc100_general_web` end to end; it was added, documented as
    "available," and never run. Fixed in `bntok/corpus.py` by bypassing
    `datasets` entirely: list the repository's auto-converted parquet
    shards (`huggingface_hub`'s `refs/convert/parquet` revision, the same
    mechanism `stream_wikisource`/`stream_xlsum` already use) and read them
    with `pandas`. A second, unrelated stall surfaced while fixing this:
    `huggingface_hub.hf_hub_download` (version 0.36.2) hangs indefinitely
    on this specific repository's files - verified directly, not assumed:
    a plain HTTP GET to the exact same resolved URL succeeds immediately
    and streams at normal speed (confirmed via `requests`), with or
    without the newer Xet transfer backend, while `hf_hub_download` on the
    identical file/revision does not return even after many minutes. Not
    diagnosed further than that reproduction; `_download_cc100_shard` in
    `corpus.py` bypasses `hf_hub_download` for CC-100 only (plain
    `requests` streaming to a local cache file) - every other `stream_*`
    function in this module still uses `hf_hub_download` successfully and
    is unaffected. Verified after the fix: real Bengali-script text (not
    the `bn_rom` romanized variant CC-100 also has for some languages),
    genuine training runs completed successfully using it (see the
    "v2 roadmap step 4/5" sections below for the resulting numbers).

16. **Corpus was diluted by Bangladesh-sourced text; no public dataset
    exists that is labelled "West Bengal (India) Bengali" specifically.**
    User asked for the training corpus to skew toward Indian, not
    Bangladeshi, Bengali. Checked directly rather than assumed: written
    standard Bengali does not split cleanly by border in any public
    corpus's own metadata, so there is no dataset that filters to
    "West Bengal only." The real, checkable lever is *pipeline origin*:
    `contemporary_news` (XL-Sum bn) is BBC Bangla, a Bangladesh-based
    source, and `cc100`/general web crawls mix both countries' domains
    with no split given. Added `stream_indiccorp_v2` (`bntok/corpus.py`) -
    AI4Bharat IndicCorp v2 Bengali, 30.0B tokens (10.6B verified + 13.8B
    synthetic + 5.6B unverified), the largest published Bengali corpus as
    of 2026-08 and built by an Indian pipeline (AI4Bharat, IIT Madras) -
    as a new `indiccorp_v2_bn` source, wired into `configs/bpe-64k.json`
    at weight 0.10, with `contemporary_news` halved (0.10 -> 0.05) to
    reduce the Bangladesh-sourced share. Streams the source's single large
    per-language text file directly over HTTP, stopping early rather than
    downloading the full multi-GB file, the same line-offset contract as
    `stream_cc100`. **Honest caveat**: IndicCorp v2's own README does not
    itself label per-document country provenance, so "India-origin" here
    means "built by an India-based pipeline for Indian-language NLP," not
    a verified per-line geographic filter - the same caveat applies to
    Sangraha, which the corpus already used before this change.
    Retraining against the updated config had not yet been run as of this
    entry (see the corpus-mix table below once it lands).

    **Survey of who else builds Indic-language AI and what they train on**
    (verified via each org's own materials, not assumed), done to inform
    this decision and to seed the competitor-tokenizer list below:
    AI4Bharat (IIT Madras) is the shared foundation layer (Sangraha,
    IndicCorp v2, IndicTrans2) that both Bhashini (govt API/data platform,
    not itself a model) and BharatGen build on. BharatGen (IIT Bombay-led
    consortium, Dept. of Science & Technology, Param-1 2.9B -> Param2 17B
    MoE, 2026) trains on a 5T-token mix (majority English: FineWeb-Edu/
    DCLM/Common Crawl) whose Indic slice leans on Books-OCR archives,
    government-funded "Udaan" translations, and Sangraha directly - i.e.
    a funded, technically strong player still leans on the same public
    corpus this repo uses. Sarvam AI (govt-backed under IndiaAI Mission,
    models now at 30B/105B as of 2026) is the one exception: their own
    materials state Sangraha lacks the "depth, diversity, and quality"
    needed and describe building a proprietary ~2T-token corpus
    ("Sarvam-2T") instead, not publicly released. Krutrim (Ola) and
    Hanooman/SML (BharatGPT ecosystem, up to 40B params) are further
    Indic players; no public HF tokenizer repo was found for Hanooman
    (checked, not just assumed absent), so it could not be added to the
    benchmark table.

17. **Competitor and frontier tokenizers added to `scripts/compare.py`'s
    `HF_MODELS`/`TIKTOKEN_MODELS`, per user request to also benchmark
    against India's own funded competitors and the global frontier, not
    only past baselines.** Every addition was verified loadable via
    `transformers.AutoTokenizer.from_pretrained` directly against the
    real Hub repo before being added, not assumed from a model card:
    added `bharatgenai/Param2-17B-A2.4B-Thinking` (BharatGen, see point 16
    above), `meta-llama/Llama-3.1-8B` (Meta; the tokenizer files load
    without a HF auth token even though the model weights are gated -
    verified directly), `mistralai/Mistral-7B-v0.3`, `Qwen/Qwen2.5-7B`,
    and a second tiktoken encoding `cl100k_base` (GPT-4/3.5) alongside the
    existing `o200k_base` (GPT-4o). Also added `google/gemma-2-9b` as the
    closest available open proxy for Gemini's own tokenizer (Google has
    not published one): this one **fails to load** in this environment
    (`403: gated repo`, confirmed by direct load attempt) and will report
    as `available: false` at measurement time per `measure_hf`'s existing
    "report unavailable, never estimate" policy - kept in the list rather
    than removed, since a user who configures a HF token can make it
    resolve, and the honest-failure path is exactly what this file's own
    `measure_hf` was built to do.

18. **India-skewed corpus (point 16) measured, not promoted.** Full held-out
    benchmark run 2026-08-07 against `artifacts/bn-bpe-64k-india` /
    `artifacts/bmbt-64k-india` on all four registers, alongside the new
    competitor rows from point 17. Result: a wash-to-slight-regression, not
    a win.

    | Register | Fertility (old default) | Fertility (India-skewed) |
    |---|--:|--:|
    | Wikipedia | 1.524 | 1.527 (worse) |
    | Literary/formal | 1.320 | 1.327 (worse) |
    | General web | 1.201 | 1.200 (same) |
    | News | 1.140 | 1.142 (same) |

    BMBT-india tracks Track-A-india near-identically, the same tie pattern
    as the non-India comparison above. Halving `contemporary_news`'s weight
    (Bangladesh-sourced XL-Sum) and adding `indiccorp_v2_bn` did not
    sharpen fertility on any register and cost a little on two of four.
    **Decision (user, 2026-08-07): keep as a labeled alternative artifact,
    do not promote.** `bn-bpe-64k`/`bmbt-64k` remain the recommended
    default; `*-india` stays available for anyone who wants corpus
    provenance skewed toward an India-based pipeline regardless of the
    small fertility cost. Full rows merged into
    `benchmarks/comparison-{register}.json` alongside this run's new
    competitor baselines (Param2-17B, Llama-3.1, Gemma-2 [gated, reports
    unavailable], Mistral-7B, Qwen2.5, GPT-4 cl100k).

## Track A2: corpus dedup and quality filtering (Gate G3)

`bntok/dedup.py`: exact dedup, MinHash-LSH near dedup (`datasketch`), and a
rule-based quality filter (`is_clean_bengali_line` plus digit- and
repeated-character-dominance rejection). No LM-perplexity stage - the
`kenlm` PyPI wheel is query-only, no `lmplz` trainer, so there is no way to
train a Bengali ARPA model without building kenlm from source (not
attempted). Measured on real data (Bengali Wikipedia + raw Sangraha
web-typed, 245,924 lines pooled) per the spec's own Gate G3 instruction:
83.5% of lines / 97.9% of words survive. Full writeup, the honest caveats
on what "raw" means here, and the gate's provisional-pass verdict:
[`docs/track-a2-corpus-survival.md`](track-a2-corpus-survival.md).

## Roadmap: a proposed v2

Everything above describes the shipped v1: grapheme-atom BPE/Unigram, which
still trains a statistical compressor, just over conjunct-safe atoms instead
of codepoints. A v2 design is proposed: parse the akshara grammar (the virama
rule) with a finite-state machine as the *primary* segmenter, and demote
statistics to a fallback for loanwords, code-mixing, and noise, emitting
featural tokens (onset/vowel/modifier) instead of opaque BPE ids. Full argument
and formal contract in
[`docs/design/reading-bengali-on-its-own-terms.md`](design/reading-bengali-on-its-own-terms.md)
and [`docs/design/FORMAL_SPEC.md`](design/FORMAL_SPEC.md).

**Steps 1-3 of the design's own roadmap are now built and property-tested**
(`bntok/substrate.py`, `bntok/akshara.py`, `tests/test_substrate.py`,
`tests/test_akshara.py`; `python -m bntok akshara --text "..."`). This is a
finite-state parser that segments text into akshara chunks by the grammar
alone, no statistics yet: no vocabulary, no merges, no BPE anywhere in it.

- Two intentional, tested divergences from `regex`'s `\X` (UAX #29 grapheme
  clusters, what v1's `grapheme_clusters()` uses): (1) khanda-ta (ৎ) is
  treated as an ordinary consonant here, so `ৎ + virama + consonant` chains
  into one akshara, where `\X` clusters khanda-ta alone and the next
  consonant separately - a real but malformed sequence either way, asserted
  explicitly in `tests/test_akshara.py` rather than silently matched or
  silently ignored; (2) on *unnormalized* decomposed input (e.g. a nukta
  letter spelled as base+nukta rather than its precomposed singleton),
  boundary alignment with `\X` is not asserted, since `aksharas()`
  deliberately does not normalize internally (same pipeline convention as
  the rest of this package: `normalize()` is a separate, explicit stage).
- On well-formed, normalized Bengali, `aksharas()`'s boundaries are expected
  to be identical to `grapheme_clusters()`'s (verified on the design doc's
  own named hard words: স্ত্রী, ক্ষ্ম, আকাঙ্ক্ষা, ঋত্বিক). This is stated
  honestly, not oversold: `\X` already groups full multi-consonant conjuncts
  correctly via Unicode's own Extend-property rules, so the parser's v1
  value is a provable, Unicode-library-independent grammar and groundwork
  for featural encoding, not novel boundary placement on valid text.
- `bntok/substrate.py` also documents Bengali-block letters not in everyday
  modern use: the archaic Sanskrit-loanword vowels VOCALIC RR/LL (and their
  matra forms), and RA WITH MIDDLE/LOWER DIAGONAL (rare in standard Bengali,
  used in Assamese, which shares this Unicode block). All are included in
  the relevant grammar class rather than left to fall through to the parser's
  "other" bucket unnamed. VEDIC ANUSVARA and the ABBREVIATION SIGN are named
  but deliberately excluded from every grammar class (the former is a
  standalone letter per Unicode's own categorisation, not a combining
  modifier; the latter is punctuation) - both correctly fall to "other".

**Step 4 (Wikipedia held-out only) is now measured, via
`scripts/compare.py`'s `measure_akshara()`.** Full writeup:
`benchmarks/bengali-comparison.md`'s "v2 roadmap step 4" section. Headline,
on the same 828-line held-out set as `bn-bpe-64k`'s own benchmark: fertility
4.527 (vs `bn-bpe-64k`'s 1.524 - expected, since the parser has no
vocabulary/merges yet, roadmap step 5), and conjunct fragmentation **0.0000
exactly**, better than `bn-bpe-64k`'s own 0.0001 (which still carries a
small residual from its atom-frequency threshold, point 1 above; the
parser's guarantee has no such threshold).

**This measurement itself found 3 real bugs**, exactly the point of running
it against real text rather than only synthetic tests: the first
fragmentation number measured (before fixing them) was 0.0012, worse than
`bn-bpe-64k`'s 0.0001, and is kept in the commit history rather than quietly
replaced (the same honesty standard as point 8 above). Full account, in the
same style as the other bugs found during development: see points 11-13 in
"Bugs found and fixed during development" above.

**Step 4 is now complete against v1's own held-out sets (all four
registers) and against real external baselines**: Sarvam-1, SUTRA, and
Krutrim are all measured for real (see `scripts/compare.py`'s `HF_MODELS`);
IndicSuperTokenizer and BengaliBPE have no usable public release (checked
directly - the IndicSuperTokenizer arXiv page has no code/tokenizer link,
and the only similarly-named BengaliBPE HF repo fails to load and isn't
verifiably the paper's own artifact) and are reported as unavailable, not
faked.

**Step 5 (partial) is now built: BMBT (Bornomala's Bengali Tokenizer,
`bntok/bmbt.py`).** Grammar (the akshara parser, reused unchanged) plus a
featural decomposition (`featurize()`: onset consonants, which carry a
nukta, vowel, modifiers, ZWJ/ZWNJ flags - a real, tested output, not an
embedding-layer afterthought) plus a statistical BPE layer over akshara
atoms, the same architecture as v1 with the atomic unit swapped from
grapheme cluster to akshara. **Morphology (root/suffix decomposition,
sandhi) is explicitly NOT built** - deferred, not abandoned. Full
architecture: `docs/bmbt-architecture.md`. `bmbt.py` is deliberately
self-contained: it imports nothing from `atoms.py` or `tokenizer.py`, so
v1 (`bn-bpe-64k`) is completely unaffected by anything here - verified by
`tests/test_tokenizer.py` passing unmodified.

**The measured comparison, trained on the identical corpus as
`bn-bpe-64k` (`configs/bpe-64k.json`, same vocab size 64000), reported
honestly, exactly as it came out - not the outcome anyone assumed in
advance:**

| Register | Fertility (v1 / BMBT) | STRR (v1 / BMBT) | Conjunct frag. (v1 / BMBT) |
|---|--:|--:|--:|
| Wikipedia | 1.524 / 1.524 | 0.722 / 0.722 | 0.000075 / 0.000075 |
| Literary/formal | 1.320 / 1.320 | 0.789 / 0.789 | 0.000104 / 0.000112 |
| General web | 1.201 / 1.201 | 0.861 / 0.861 | 0.000055 / 0.000057 |
| News | 1.140 / 1.140 | 0.893 / 0.894 | 0.000025 / 0.000025 |

On Wikipedia, the two are not just close but byte-for-byte identical down
to the raw counts (17,245 tokens, 11,316 words, 3 fragmented clusters,
both tokenizers - verified directly, not a rounding coincidence: the two
artifacts have genuinely different atom vocabularies, 12,233 atoms for v1
vs 12,199 for BMBT). On the larger registers, tiny real, non-identical
differences appear in both directions: BMBT needs marginally *fewer*
tokens on literary/formal, general web, and news (25-60 fewer out of
370,000-1,230,000, roughly 0.005-0.02%) but has marginally *more*
fragmented clusters on the same three (2-23 more, still in the same
0.00005-0.0001 near-zero band as v1). Neither direction is large enough to
call a win; this is an honest tie, not a hedge.

**Why they tie rather than BMBT clearly losing**, given `FORMAL_SPEC.md`'s
own proof that a constrained BPE cannot beat an unconstrained one on raw
token count: akshara-grammar boundaries are already nearly identical to
`\X`'s grapheme-cluster boundaries on well-formed Bengali (the step-3/4
measurement's own finding, points 11-14 above), so constraining BPE to
respect akshara boundaries instead of grapheme-cluster boundaries barely
constrains anything further in practice - the two atom schemes are close
to isomorphic on real text, so BPE trained to the same vocabulary size
over either converges to near-identical tokenization behaviour, even
though the actual vocabularies differ.

**What BMBT adds, independent of this tie**: a provable, Unicode-library-
independent grammar instead of delegated trust in `regex`'s own `\X`
implementation, and `featurize()` - a real structural decomposition
v1 never had at all, at zero fertility cost either way.

**CC-100 ablation** (`configs/bpe-64k-cc100.json`, the same corpus plus
`cc100_general_web`, trained on both architectures as `artifacts/*-cc100`):
adding CC-100 very slightly *hurts* Wikipedia fertility (1.531 vs 1.524
without it - the same 64k-token vocabulary budget now split across five
sources instead of four, diluting Wikipedia-specific coverage slightly)
and very slightly *helps* general web fertility (1.199 vs 1.201 - the
register CC-100 actually targets). Both effects are in the third decimal
place, directionally sensible, and together amount to a wash, not a case
for or against adopting CC-100 in the default weights. `bn-bpe-64k` and
`bmbt-64k` (no CC-100) remain the recommended artifacts.

**Unigram-vs-BPE algorithm ablation** (`artifacts/bn-bpe-64k-unigram`,
`artifacts/bmbt-64k-unigram`: same corpus, same atom scheme, same 64k
vocab budget as the shipped artifacts, only the `tokenizers` trainer
algorithm swapped from BPE to Unigram): Unigram is worse than BPE on
every register, for both v1 and BMBT, by a consistent 2.5-5.4%:

| Register | Fertility (BPE) | Fertility (Unigram) | Worse by |
|---|--:|--:|--:|
| Wikipedia | 1.524 | 1.562 | 2.5% |
| Literary/formal | 1.320 | 1.392 | 5.5% |
| General web | 1.201 | 1.235 | 2.8% |
| News | 1.140 | 1.175 | 3.1% |

STRR moves the same direction (worse) on every register; conjunct
fragmentation is unaffected either way (same near-zero band as BPE, the
atom scheme, not the merge algorithm, is what controls fragmentation).
`bn-bpe-64k`/`bmbt-64k` (BPE) remain the recommended artifacts; Unigram
is kept as a measured, negative ablation result, not deleted.

**BMBT-Hybrid: frequency-adaptive akshara atoms** (`bntok/bmbt_hybrid.py`,
artifact `artifacts/bmbt-64k-hybrid-v2`, actual vocab 64,355 - close to but
not identical to v1/BMBT's 64,000, see below). Splits a rare akshara chunk
into an onset atom and a tail atom so frequency can accumulate across
combinations that would otherwise each burn their own opaque atom, fusing
only the top 200 most frequent chunks (`k_fused`) where splitting is pure
overhead. Two false starts before this shipped (an unsound merge-order-
prediction fix, then a working reserved-vocab-id guard) are documented in
the module's own docstring, not repeated here.

Measured on all four held-out registers, model tag
`Bornomala BMBT-Hybrid (bpe 64355)` in `benchmarks/comparison-*.json`.
**One methodological caveat up front**: the Wikipedia row is genuinely
apples-to-apples with v1/BMBT (same 800-line `held_out(15000, 800)` sample,
11,316 words for all three models), but the literary_formal/general_web/news
rows are not - they were measured on the first 800 lines of each register's
held-out slice (a fair, matched sample *across the four Hybrid rows
themselves*, the reason that recap was done), while the existing v1/BMBT
rows in those same three files were measured earlier on the *full*
held-out slice (932,211 / 305,557 / 485,356 words respectively, 10-30x
more text). Fertility is a ratio and reasonably robust to this, but the
fragmented-cluster counts below are not, so rates (not raw counts) are
what's compared:

| Register | Fertility (v1/BMBT) | Fertility (Hybrid) | Worse by | Frag. rate (v1/BMBT) | Frag. rate (Hybrid) |
|---|--:|--:|--:|--:|--:|
| Wikipedia | 1.524 | 1.648 | 8.1% | 0.000075 | 0.00005 |
| Literary/formal | 1.320 | 1.347 | 2.0% | 0.000104 | 0.000083 |
| General web | 1.201 | 1.310 | 9.1% | 0.000055 | 0.000011 |
| News | 1.140 | 1.207 | 5.9% | 0.000025 | 0.0 |

A real trade-off, not a clean win or loss. Fertility is worse by 2-9% on
every register (STRR moves the same direction, worse) - the opposite of
what an earlier, preliminary measurement against a larger-effective-vocab
version of this artifact suggested (see the module's own docstring); that
comparison did not hold once corrected for vocab size against the fair,
64,355-vocab `-v2` artifact actually benchmarked here. What does hold
directionally is fragmentation: the rate is lower on every register,
including exactly zero on news - but the literary_formal/general_web/news
Hybrid rates rest on very small absolute counts (8, 1, and 0 fragmented
clusters out of 31,912/28,558/14,125 words) against an already near-zero
baseline, so this is suggestive, not as statistically solid as the
Wikipedia row, which is both apples-to-apples and shows the same direction
(2 fragmented clusters versus 3, out of 11,316 words).

**Decision: keep as a labeled experimental artifact, do not promote.**
This project ranks tokenizers by fertility first (fewest tokens per word);
a 2-9% fertility cost is not worth paying to shrink a fragmentation rate
that is already near zero for v1/BMBT. `bn-bpe-64k`/`bmbt-64k` remain the
recommended default; `bmbt-64k-hybrid-v2` stays available, clearly
labeled, for anyone who weights conjunct-fragmentation headroom above raw
token count.

**Still not done**: morphology (root/suffix decomposition, sandhi) -
BMBT's featural output has no morphological layer yet, so it cannot yet
claim the "quality-per-token" advantage the design doc's own risk section
frames as the actual bet worth making. The named risk from the design
doc - that a constrained tokenizer can raise fertility even as it adds
structure - did not materialise here (the tie holds, it did not get
worse), but that was measured, not assumed, and the deeper claim (a
featural, eventually morphology-aware tokenizer produces genuinely better
downstream model quality per token) remains completely unmeasured; this
project has no downstream task evaluation at all yet.

## How to report a new issue

Open an issue on the repository with the exact input, the command, and the
versions. Reproducibility is the point.
