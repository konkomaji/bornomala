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
   attempt on pure Wikipedia text (1.568 vs 1.952). The same fixed token
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
   Wikipedia-only v0.1 on Wikipedia held-out text (point 3). The same 32k
   token budget spread across four registers instead of one is sharper
   nowhere. An ablation across 32k/48k/64k (BPE, same corpus, same weights)
   showed fertility recovering monotonically with vocab size (in-sample: 1.482
   → 1.375 → 1.319). At 64k (`artifacts/bn-bpe-64k`, the current shipped
   artifact) the tokenizer beats IndicBERTv2 on fertility, STRR, *and* conjunct
   fragmentation on every one of the four held-out registers tested
   (Wikipedia, literary/formal via Sangraha pdf-typed, general web via
   Sangraha web-typed, and news via XL-Sum): fertility 1.524/1.320/1.201/1.140
   vs IndicBERTv2's 1.652/1.612/1.395/1.312; conjunct fragmentation
   0.0001/0.0001/0.0001/0.0000 vs IndicBERTv2's 0.0440/0.0562/0.0277/0.0206,
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
   through three versions before it was actually correct**, kept in full since
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
     reconstruction, for a codepoint genuinely outside this tokenizer's
     coverage (Greek, Arabic, and Japanese text quoted inside real Bengali
     Wikipedia/news articles all occur in this corpus), the BPE model emits
     the `<unk>` special token, and `encode_tokens()`'s fallback
     (`readable if readable else t`) returns the literal 5-character string
     `"<unk>"` in place of the missing text, silently desynchronising every
     offset after it on that line.
   - *v3 (current, verified correct):* trims the leading Metaspace space
     before computing offsets, and skips fragmentation counting (only) on
     lines that do not cleanly round-trip (`tok.roundtrip_ok(raw)` is False,
     i.e. the out-of-coverage case in point 4, those lines still count
     normally toward fertility/STRR/bytes). A hard assertion checks the
     surface reconstruction matches the normalised text exactly on every line
     that remains, so a similar bug cannot silently corrupt a comparison
     again. It crashes instead. This version is what produced every number
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
    text as its first design-step-4 measurement, not a synthetic test.
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
    `benchmarks/bengali-comparison.md`'s "v2 design step 4" section); the
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
    "v2 design step 4/5" sections below for the resulting numbers).

16. **Corpus was diluted by Bangladesh-sourced text; no public dataset
    exists that is labelled "West Bengal (India) Bengali" specifically.**
    The design goal is a training corpus skewed toward Indian, not
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

17. **A broader set of tokenizers added to `scripts/compare.py`'s
    `HF_MODELS`/`TIKTOKEN_MODELS`, extending coverage to major Indic and
    global frontier models, not only the original baseline set.** Every
    addition was verified loadable via
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

19. **A coverage gap in our own benchmark, found and closed: the third
    Gate G2 baseline had a public release all along and had never been
    run.** The whitepaper's Gate G2 names three external baselines. Two of
    them, IndicSuperTokenizer (arXiv:2511.03237) and BengaliBPE
    (arXiv:2511.05324), were checked directly and honestly reported as
    having no usable public artifact (point 17 and `scripts/compare.py`'s
    header comment). The third, **BrahmicTokenizer-131K**
    (`theschoolofai/BrahmicTokenizer-131K`, Apache-2.0, arXiv:2605.29379),
    was never checked at all - it was simply never attempted. It does have
    a real release: it loads as a `PreTrainedTokenizerFast` with working
    `offset_mapping`, needs no auth token and no `trust_remote_code`, and
    therefore gets a **full** row here including the conjunct-fragmentation
    column, not a partial one.

    This is recorded as our own omission, not as an availability problem on
    their side. Until 2026-08-16 the claim "outperforms existing public
    tokenizers on our benchmark" was carrying an untested public tokenizer,
    which is exactly the kind of gap this file exists to record.

    **Measured result (Wikipedia held-out, same 828 lines as every other
    row): fertility 2.620, STRR 0.154, 6.62 bytes/token, conjunct
    fragmentation 0.2209.** It does not change the ranking - it lands 8th,
    a hair behind GPT-4o (2.608) and ahead of mBERT (2.777).

    **The interesting part is that it is Indic-targeted and still lands
    there.** BrahmicTokenizer is built specifically as an Indic-capable
    drop-in replacement for OpenAI's o200k, and it carries 131,072 tokens
    against our 64,000 - more than twice the vocabulary budget. It still
    fragments 22.1% of Bengali conjuncts, worse than script-blind mBERT's
    18.0%, because it is still a byte-level BPE with no notion of a
    grapheme cluster. Targeting Indic scripts in the training mix is not
    the same as constraining the merge space to the script's own units.

    **The vocabulary asymmetry is stated in both directions, because it
    cuts both ways.** BrahmicTokenizer has roughly twice our raw budget,
    which favours it; but it spreads that budget across 12 Brahmic-script
    languages (~11k effective per language) where ours is spent entirely
    on Bengali, which favours us. Neither framing alone is honest. The
    general caveat stands for this whole comparison table: every external
    baseline in it is multilingual, and a monolingual Bengali tokenizer
    beating multilingual ones on Bengali is an expected consequence of
    that design choice, not a surprising discovery. The comparison is
    still worth making, because these are the tokenizers Bengali text
    actually gets encoded by in practice.

## BMBT-Hybrid: a failed experiment, removed

BMBT-Hybrid was an attempt to improve fertility by changing WHAT the atomic
unit is, rather than how BPE merges it. It has been **removed from the
codebase**: the module, its tests, its three artifacts, its three CLI
subcommands and its row in `scripts/compare.py` are all gone. It is recorded
here because the reason it failed is structural and worth not rediscovering.

### The idea, which was sound

A rare akshara spends its whole corpus frequency on one opaque atom. Factoring
it into an onset atom (the consonant chain) and a tail atom (matra plus
modifiers) lets the onset accumulate frequency across every vowel it occurs
with, and the tail across every onset it attaches to. v1 cannot do this at all:
its atoms are opaque grapheme-cluster strings with no internal boundary to
split on. Only the akshara grammar knows where to cut. Factoring the very
highest-frequency syllables is pure overhead, so only the long tail was
factored and the top `k_fused` were left whole.

At small scale it looked right: 8k vocab on 3,000 Wikipedia articles gave
fused-only 1.9587, always-factored 1.8700, hybrid k=200 1.8454.

### Why it failed, in four stages

1. **Factoring breaks the conjunct guarantee.** At production scale fertility
   did improve 1.2-3.8%, but conjunct fragmentation got 20-70x worse
   (0.18-0.70% against a flat 0.01%). Factoring creates a cut point and BPE is
   under no obligation to heal it: wherever the learned merges did not happen
   to re-fuse an onset+tail pair, the boundary survived as a real split.

2. **The obvious fix was unsound.** Baking low-priority merge rules to force
   re-fusion required predicting the intermediate symbols a left-to-right fold
   would produce. Real BPE applies whichever adjacent pair has the globally
   lowest rank first, not left to right, so a high-priority learned merge can
   fuse a middle pair before the predicted chain reaches it. Caught on a real
   case, `ঙ্ক্ষ` partially merging to `ঙ্` + `ক্` + `ষ`, before shipping.

3. **The working fix spent the vocabulary the factoring saved.**
   `_reserve_guaranteed_chunk_vocab` sidestepped merge-order prediction by
   reserving a dedicated id for every multi-atom chunk's full span, plus an
   encode-time guard. Fragmentation went to zero, and the first build ballooned
   to 90,433 effective ids against a requested 64,000.

4. **The fair rerun reversed the result.** Vocab-matched at 64,355 and
   benchmarked across all four held-out registers, BMBT-Hybrid came out
   **2-9% WORSE on fertility than v1/BMBT on every register**. The original
   "1.2-3.8% better" had been measured against the inflated-vocab build.

### Why this is structural, not a tuning failure

Factoring buys compression by sharing sub-akshara pieces. Preserving the
conjunct guarantee forces a reserved whole-span id for every multi-atom chunk,
which re-spends exactly what the sharing saved. The two goals are in direct
tension and the guarantee wins. No value of `k_fused` changes that.

One result survived: fragmentation was genuinely lower than v1/BMBT on every
register, zero on news.

### What replaced it

The morphology layer reaches the same underlying goal by a different route.
Instead of factoring aksharas by FREQUENCY and then trying to re-fuse them, it
factors them only at MORPHEME seams, where the split is linguistically real and
no re-fusion is wanted. That needs no reservation mechanism, so it does not
spend the vocabulary, and it keeps conjunct integrity absolute. See
[`docs/bmbt-morphology.md`](bmbt-morphology.md).

### Housekeeping this closes

`artifacts/bmbt-64k-hybrid` and `artifacts/bmbt-64k-hybrid-fixed` had been
byte-identical since they were committed: the docstring described a real fix
between them that was never baked into a regenerated artifact. Both, and
`bmbt-64k-hybrid-v2`, are now deleted. The long-standing duplicate-artifact
item is closed by removal rather than by choosing between them.

## Track A2: corpus dedup and quality filtering (Gate G3)

`bntok/dedup.py`: exact dedup, MinHash-LSH near dedup (`datasketch`), and a
rule-based quality filter (`is_clean_bengali_line` plus digit- and
repeated-character-dominance rejection). No LM-perplexity stage - the
`kenlm` PyPI wheel is query-only, no `lmplz` trainer, so there is no way to
train a Bengali ARPA model without building kenlm from source (not
attempted). Measured on real data per the spec's own Gate G3 instruction,
across four sources: Bengali Wikipedia + raw Sangraha web-typed (245,924
lines pooled: 83.5% lines / 97.9% words survive); AI4Bharat IndicCorp v2
at scale (2,000,000 lines, 90.6M words: 97.9% lines / 99.3% words
survive); and **CC-100 (1,000,000 lines) as the genuine raw-web proxy -
no AI4Bharat/Wikimedia curation pipeline behind it, unlike the other
three - measuring the real outlier: 63.2% lines / 79.6% words survive**,
still comfortably above the gate's 5-10% crisis floor and still
projecting well above 5B tokens even at that lower ratio. Full writeup,
the honest caveats on what "raw" means for each source, and the gate's
high-confidence-pass verdict, now stress-tested against real raw-web
data: [`docs/track-a2-corpus-survival.md`](track-a2-corpus-survival.md).
Now wired into `build_configured_corpus` itself as an opt-in step
(`dedup=True`; CLI `--dedup` on `train`/`bmbt-train`/`hybrid-train`),
per-source before weighting, default off - neither shipped artifact has
been retrained with it yet.

## Design steps: a proposed v2

Everything above describes the shipped v1: grapheme-atom BPE/Unigram, which
still trains a statistical compressor, just over conjunct-safe atoms instead
of codepoints. A v2 design is proposed: parse the akshara grammar (the virama
rule) with a finite-state machine as the *primary* segmenter, and demote
statistics to a fallback for loanwords, code-mixing, and noise, emitting
featural tokens (onset/vowel/modifier) instead of opaque BPE ids. Full argument
and formal contract in
[`docs/design/reading-bengali-on-its-own-terms.md`](design/reading-bengali-on-its-own-terms.md)
and [`docs/design/FORMAL_SPEC.md`](design/FORMAL_SPEC.md).

**Steps 1-3 of the design's own build sequence are now built and property-tested**
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
`benchmarks/bengali-comparison.md`'s "v2 design step 4" section. Headline,
on the same 828-line held-out set as `bn-bpe-64k`'s own benchmark: fertility
4.527 (vs `bn-bpe-64k`'s 1.524 - expected, since the parser has no
vocabulary/merges yet, design step 5), and conjunct fragmentation **0.0000
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
sandhi) has since been built; see `docs/bmbt-morphology.md`. Full
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
project has no downstream task evaluation at all yet. A named candidate
for when that becomes relevant (Track E, Gate G6): **IndicGenBench**
(Google Research) - 29 Indic languages including Bengali, four generation
tasks (CrossSum-IN summarization, Flores-IN translation, XQuAD-IN reading
comprehension, XorQA-IN cross-lingual QA), canary-stringed against
accidental training-set contamination. Ready-to-use ahead of the
natively-authored cultural-register benchmark Track E's own spec calls
for (section 13.5); ready-made is not a substitute for that, since it is
translated/parallel-constructed rather than authored Bengali-native, but
it is a real, immediately available generation-quality signal - not yet
wired into anything, correctly, since no model exists to run it against.

## Banglish: a real gap found, measured, and partially closed

Both tokenizers (v1 and BMBT, tied) were checked against romanized Bengali
chat text (\"tumi kemon acho\" - the WhatsApp/social-media register most real
Bengali internet writing actually happens in) and measured, not assumed, to
be dead last: 17th of 17 loadable tokenizers on a real held-out set,
including behind BanglaBERT and BanglaT5. This section records what was
found and what was built in response.

**The held-out set.** `scripts/compare.py --register banglish` streams
CC-100's `bn_rom` config (Wenzek et al., 2020 - real, verified present on
the Hub, 2 shards), filtered by `bntok.corpus.is_clean_banglish_line`
(a lexicon-based heuristic: rejects lines without a minimum count and share
of common romanized Bengali function words, since the script-ratio trick
`is_clean_bengali_line` uses does not apply to text that is already all
Latin script). Never used in any training config in this repository, so
there is no offset/disjointness bookkeeping needed the way the other three
registers require (`bntok.corpus.build_banglish_held_out`).

**Baseline, measured 2026-08-17, 2000 filtered real lines, 33,560 Latin
words:** our tokenizer (v1 and BMBT identically, since BMBT's grammar is
Bengali-script-only and gets no edge here) scored fertility 2.827, worse
than every other tokenizer measured, including Sarvam-1 (2.596), BanglaBERT
(2.381), and BanglaT5 (2.266); SUTRA led at 1.850. Cause: this tokenizer's
64k vocabulary is spent entirely on Bengali script by design, so it has
learned essentially no Latin-script merges, where every competitor's
training mix has at least incidental English/multilingual exposure.

**The fix does not compete on raw Latin-script BPE.** Throwing more
`bn_rom` corpus at BPE would mean fighting corpus-scale battles against
groups with larger crawls, and would dilute the Bengali-script vocabulary
budget that is this project's actual advantage. Instead: transliterate
Banglish to real Bengali script first, then hand the result to the
tokenizer that already wins by a wide margin on Bengali script (1.140-1.524
fertility across all four registers, see the comparison table above). A
frequency-tiered cascade does this cheaply, the same grammar/frequency-
first-then-statistics philosophy BMBT already applies to script itself,
one layer up:

- **Tier 0** - Bengali-script text passes through untouched, zero new cost.
- **Tier 1** - a real-word lookup table (`bntok/data` conceptually,
  currently `artifacts/banglish-lookup.tsv`, built by
  `scripts/build_banglish_lookup.py`): 155,615 entries, 125,733 from real
  Dakshina v1.0 data (Google Research, MIT-licensed - the official
  `storage.googleapis.com/gresearch/dakshina` release, verified directly,
  word lexicon train+dev splits plus the natural-sentence word alignment;
  the lexicon test split is reserved, untouched, for future honest
  evaluation) and 29,882 synthetic gap-fill entries for words absent from
  Dakshina (real entries never overwritten by synthetic ones). O(1) dict
  lookup, no model.
- **Tier 2** - `bntok.banglish.NgramClassifier`, a character-bigram/trigram
  Naive Bayes classifier trained from scratch (no pretrained model, no
  fine-tuning) on the tier-1 table's real Banglish words versus a public
  10,000-word English list (`first20hours/google-10000-english`, verified
  reachable), deciding whether a word tier 1 missed is real English (leave
  untouched - real code-mixed English inside Banglish sentences is common)
  or unresolved Banglish. Held-out accuracy: 89.8% on Banglish words, 80.7%
  on English words.
- **Tier 3** - the actual seq2seq transliteration model for whatever tiers
  1-2 leave unresolved. **Not built yet.** Scoped: byte/akshara-level
  Transformer, trained from scratch (no fine-tuning, per design decision),
  on Dakshina plus `bntok.banglish_synth`'s synthetic pairs, meant to run on
  Colab's free tier (small model, checkpointed to Drive for session-limit
  resilience). Until it exists, `bntok.banglish.transliterate()` passes
  unresolved words through unchanged and reports the count honestly rather
  than hiding the gap.

**`bntok/banglish_synth.py`**: generates synthetic (noisy_latin,
canonical_bengali) training pairs from real Bengali text already in this
repository's corpus, via a reverse phonetic table (Avro-Phonetic-convention
spellings) reusing BMBT's own `featurize()` output (onset/vowel/modifiers)
rather than a new parser. Deliberately does NOT invent digit-substitution
slang rules (e.g. \"kor6o\") - those patterns are not verifiable as
systematic without real examples, so inventing them would be exactly the
kind of fabricated pattern this document exists to flag, not fix. Validated
against Dakshina's real lexicon (a 3000-word sample, 15-seed coverage per
word), not assumed correct:

| Table state | Word-level hit rate vs real Dakshina spellings |
|---|--:|
| First cut | 62.6% |
| + inherent vowel reweighted (medial "o" preferred over dropping it) | 69.5% |
| + word-final inherent-vowel drop + positional য (glide vs onset) fix | 75.2% |
| + positional ব (glide vs onset) fix, same pattern as য (2026-08-18) | **77.8%*** |

\* Measured on `artifacts/banglish-translit-data/dev.tsv` (2,500-word sample,
15 seeds/word) via `scripts/validate_banglish_synth.py`, not a fresh
3,000-word Dakshina draw like the rows above - close enough in methodology to
compare directly (same real-spelling-match criterion, same seed count), not
run on the identical sample.

Each fix was traced to a concrete real mismatch (e.g. \"protyahar\" render-
ing as \"protzahar\"/\"protjahar\" before the positional য fix; \"smorthoner\"
instead of \"shomorthoner\" before the inherent-vowel reweight), not tuned
blind. Remaining gaps are lexical (loanwords with fixed English spellings,
e.g. ঈদ - \"Eid\", not phonetic \"id\"/\"ii\") or rare special ligatures
(ক্ষ), not systematic rule bugs - not chased further, matching this
table's documented scope (\"heuristic approximation, not authoritative\").

**End-to-end result, tiers 0-2 only, no model, measured on the same 2000-
line held-out set the 2.827 baseline came from:** of 33,560 Latin words,
79.5% resolved at tier 1, 6.1% correctly left alone as real English, 14.4%
remain unresolved (passed through raw, tier 3 not built). Fertility on the
transliterated output: **1.740** - better than every tokenizer on the
17-way leaderboard, including the previous leader SUTRA (1.850), achieved
with zero GPU and zero trained model. Tier 3 has only upside from here: it
targets exactly the remaining 14.4%.

**Efficiency, measured not asserted.** The lookup table's coverage follows
a real Zipfian curve on Dakshina's 100,483 real word occurrences (natural
sentence context): top 1,000 words cover 47.6%, top 5,000 cover 69.6%, top
20,000 cover 89.8%. Most real Banglish traffic resolves at tier 1, meaning
most traffic never touches a model at all - the design's actual answer to
computational cost, not a claim that any one model call is fast.

**Three follow-up items closed out same day, each checked against real
data before deciding scope, not assumed:**

- **Tier-2 classifier's English recall.** Diagnosed directly: the training
  split is 113k Banglish words versus 9k English words (an artifact of
  Dakshina's size versus a 10k-word English list, not a real-world class
  ratio), which skewed the Naive Bayes prior hard toward "banglish" and was
  the actual cause of real English words like "arizona"/"bosnia"/"by"
  being misclassified. Fixed to balanced (uniform) priors. Recall moved
  from 89.8%/80.7% (banglish/english) to 85.9%/87.6% - a small banglish-
  recall cost for a real english-recall gain, the correct direction once
  tier 3 exists: misclassifying real English as Banglish would feed it
  into a transliteration model and mangle it, the costlier failure mode.
- **Tier-1 collision handling.** Measured first, not assumed: only 6.4% of
  distinct Latin spellings (7,995 of 125,733) are even ambiguous, and
  inspecting the close calls shows most are the SAME word under Bengali's
  own real orthographic inconsistency (ন/ণ, ং/ঙ spelling variance - e.g.
  "ankito" -> অঙ্কিত/অংকিত), not genuine different-word ambiguity. A
  context-aware disambiguator (a bigram language model over resolved
  words) would be real engineering for a problem this small and mostly
  harmless - kept out of scope for that reason, not left unconsidered. The
  runner-up candidate is now stored and exposed
  (`load_lookup_table_full`) rather than silently discarded, so a future
  pass has what it needs without rebuilding the table.
- **Self-growing cache.** `bntok.banglish.transliterate()` now takes an
  optional `tier3_fn` hook: when it resolves a word, the result is written
  back into the shared `lookup` dict, so the same spelling is a tier-1 hit
  from then on - a novel word costs the model once, never again. The
  mechanism is tested (`tests/test_banglish.py`) against a stub `tier3_fn`,
  proving the write-back and no-second-call behaviour work correctly,
  independent of what the real model turns out to be.

**Tier 3, fully scoped and ready to run, not yet run:**

- `scripts/assemble_banglish_translit_dataset.py`: builds train/dev/test-
  ready data. train.tsv (Dakshina lexicon train split + the aligned
  natural-sentence pairs + large-scale `bntok.banglish_synth` synthetic
  augmentation from our own corpus) and dev.tsv (Dakshina lexicon dev
  split, real only, deliberately no synthetic, so checkpoint selection is
  judged against real human spellings). The lexicon TEST split stays
  untouched, reserved since the tier-1 table was first built, for the one
  honest blind evaluation.
  A first, naive assembly run measured a target (Bengali) vocabulary of
  1,077 symbols spanning a dozen foreign scripts, emoji, and control
  characters - real contamination in Dakshina's Wikipedia-sourced data and
  this project's own corpus (both legitimately quote foreign-script text
  inside otherwise-clean Bengali lines; `is_clean_bengali_line`'s line-
  level ratio filter correctly lets those lines through, but a single
  contaminated word from within one still is not a valid training pair).
  Caught by inspecting the output before ever starting a Colab run, not
  after: `_is_valid_pair` now requires pure-Latin source and pure-Bengali-
  block target, dropping 354,197 of 733,908 raw pairs (contamination and
  duplicates together) to a clean 379,711-pair train set. Final vocabulary:
  33 source symbols, 78 target symbols - the expected range for character-
  level Latin/Bengali.
- `scripts/train_banglish_translit.py`: a small (~5-10M parameter)
  character-level Transformer encoder-decoder, trained fully from scratch
  (random init, no fine-tuning), own PyTorch training loop, checkpointed
  every `--save-every` steps to survive Colab's session limits - re-running
  with the same `--ckpt-dir` resumes automatically. Verified end to end on
  a tiny CPU smoke-test dataset: loss drops, checkpointing and resume both
  work correctly (confirmed a resumed run picks up at the exact saved
  step), dev exact-match accuracy climbs as expected. A real bug was caught
  in this pass too: the model's hyperparameters were not saved with the
  checkpoint, so evaluating a checkpoint trained with non-default settings
  failed to load. Fixed by saving the full model config inside every
  checkpoint.
- `scripts/eval_banglish_translit.py`: the honest blind evaluation against
  Dakshina's reserved test split, reporting two numbers deliberately (not
  one) - strict word-level exact-match, and character error rate for
  partial credit on close-but-wrong output, since fertility or accuracy
  alone can hide a model that looks plausible but is not actually correct.
- `colab/train_banglish_tier3.ipynb`: mounts Drive, clones the repo,
  trains, evaluates, points at where to wire the trained checkpoint into
  `bntok.banglish.transliterate()`'s already-tested `tier3_fn` hook.

**Trained and measured (session 9, 2026-08-17/18).** The notebook ran on a
real Colab T4: 20,000 steps, small model (d_model=256, 4 layers). Blind eval
against Dakshina's reserved test split (9,146 distinct Latin spellings, never
touched by training/dev):

| Decoder | Exact-match | CER |
|---|--:|--:|
| Greedy | 45.1% (4,126/9,146) | 16.9% |
| Beam search (size 5) | **53.9% (4,929/9,146)** | **13.5%** |

Beam search added afterward (`TranslitTransformer.beam_decode`/
`beam_decode_batch`, per-example, `--beam-size` on `eval_banglish_translit.py`)
once the greedy result was in - a real +8.8pp exact-match / -3.4pp CER lift
on the exact same checkpoint, zero retraining. Verified as a genuine decoder
change and not a fluke: `beam_size=1` produces token-for-token identical
output to `greedy_decode` on an untrained model (both always argmax at each
step). A second lever - a bigger model (d_model 384, 6 layers) and longer
training (30k steps) - is built and ready in the notebook's optional
section, not yet run: beam search alone already moved the number
substantially, and that was tried first per its zero cost.

Neither checkpoint's weights are committed to the repo (too large for git) -
they live on Drive and on a local machine.

**Wired into `transliterate()` for real (session 9, same day).**
`bntok.banglish_tier3.load_tier3_fn(ckpt_path, vocab_path)` loads a trained
checkpoint and returns a real `str -> str | None` callable, beam search
(size 5) by default - the better decoder should be the default for anyone
just wiring this in, not an opt-in they have to know to ask for. Required
moving `TranslitTransformer` out of `scripts/train_banglish_translit.py`
(where it was first written) into `bntok/banglish_tier3.py`: the library
needs the model class too, and a library should not import from `scripts/`,
so this is the correct direction, not a shortcut - both training/eval
scripts now import the class from `bntok` instead of each carrying their
own copy that could drift apart. torch stays an optional dependency of
this one module (lazily imported), not of `bntok` as a whole.

Verified against the real local checkpoint, not just that it imports:
`load_tier3_fn()` loads `step-20000.pt` successfully, and resolves real
out-of-lookup words correctly in isolation (`ghurte` -> `ঘুরতে`, `notun` ->
`নতুন`) and honestly imperfectly where the measured 53.9% accuracy predicts
it should be (`shopno` -> `শপন`, wrong - should be `স্বপ্ন`). Full pipeline
integration verified too: `transliterate("... ghurte jabo", lookup,
classifier, tier3_fn=tier3_fn)` correctly routes the out-of-lookup word to
tier3, resolves it, and grows the shared cache (`tier3_hits=1,
cache_growth=1`) exactly as designed.

## BMBT downstream eval: scoped and built, the actual unmeasured bet

Fertility ties exactly between v1 and BMBT (see the comparison table
above); `bmbt-64k-morph` costs +21.7% to +42.1% fertility on top of that,
depending on register (see "BMBT morphology" section). The one thing that
was never measured for either variant is whether the grammar-first
structure - and, for BMBT-morph, the added morphology layer - makes a
language model better PER TOKEN, not just tie (or lose) on raw token
count. That is the actual claim the design doc's own risk section names
as the real bet worth making, and it is the only way to know if
BMBT-morph's fertility cost buys back anything downstream.

**The controlled experiment**: three small decoder-only Transformers
(`scripts/train_lm.py`, a from-scratch GPT-style model, no pretrained
weights), IDENTICAL architecture and hyperparameters, trained on real text
(`scripts/assemble_lm_corpus.py`, the same literary-weighted source mix as
`configs/bpe-64k.json`, at a smaller scale for a bounded Colab run) -
differing only in which tokenizer (`scripts/prepare_lm_tokens.py`, run
once per tokenizer: `bn-bpe-64k`, `bmbt-64k`, `bmbt-64k-morph`) produced
the input token stream. Compared on held-out
**bits-per-byte**, not raw per-token perplexity: v1 and BMBT have
different vocabularies, so cross-entropy is normalized against the held-
out text's fixed original UTF-8 byte count, the standard way to compare
language models fairly across tokenizers with different token counts.

**A real disjointness bug caught during this build, not after.** An early
version of `assemble_lm_corpus.py` built the held-out Wikipedia slice by
slicing `stream_wikipedia`'s output list at index `WIKIPEDIA_TRAIN_ARTICLES`
(15,000) - but that function's `limit` parameter counts ARTICLES while its
return value is a flattened list of LINES (many per article), so the
slice landed somewhere around line 15,000 (roughly article 500, given
~29 lines/article), not after article 15,000 at all. This would have
badly overlapped the held-out set with `build_configured_corpus`'s own
`bengali_wikipedia` training source. Caught by a smoke-test run producing
an obviously wrong held-out size (439,303 lines from a 20-article
request). Fixed by skipping ROWS directly against the streaming dataset,
mirroring `scripts/compare.py`'s own `held_out()` function, the pattern
this whole project already relies on successfully for every register
comparison run this session - not a new invention, the same proven
approach applied where it had been missed.

**Verification, confirmed live, not just by code inspection.** A rerun of
the corrected `assemble_lm_corpus.py` (20-article held-out request)
produced 474 held-out lines (about 23.7 lines/article - sane), not the
439,303-line result the bug produced. `train_lm.py`'s own mechanics
(model construction against a real 64,000-entry vocabulary, checkpointing,
resume, and the bits-per-byte calculation) were separately verified end to
end on real tokenized data from both `bn-bpe-64k` and `bmbt-64k`
(identical token counts on the same sample text, consistent with the
fertility tie), including a correctness sanity check on synthetic random
data: training loss correctly plateaus at ln(vocab_size) nats, the
theoretical entropy floor for unlearnable random tokens - confirms the
cross-entropy path is implemented correctly, not just plausible-looking.

**What genuinely remains**: running `colab/train_bmbt_downstream_eval.ipynb`
at real scale - three real GPU training runs (v1, BMBT, BMBT-morph),
longer than tier 3's single run, real compute this environment does not
have. Everything upstream is built, tested, and now verified correct end
to end - the notebook was extended 2026-08-18 to add BMBT-morph as a
third arm (was v1-vs-BMBT only) once BMBT-morph existed as a real
trained artifact; `prepare_lm_tokens.py`/`train_lm.py` needed no code
changes, `bmbt-64k-morph` loads through the same `BMBT.load()`/`--bmbt`
path as `bmbt-64k` since it is the same artifact format
(`bornomala-bmbt/1`), verified by loading it directly before touching
the notebook.

## MorphScore: attempted, Bengali sample too small to be a benchmark

MorphScore (Arnett, Hudspeth & O'Connor, ICML 2025 Tokenization Workshop,
https://arxiv.org/abs/2507.06378) scores how well a tokenizer's boundaries
line up with real morpheme boundaries in Universal Dependencies data, across
70+ languages. It looked like the fastest route to a downstream-quality claim
that isn't fertility, since fertility ties between `bn-bpe-64k` and BMBT (see
the BMBT section above). Run via `scripts/morphscore_eval.py`, results in
`benchmarks/morphscore.json`. The honest conclusion: **it doesn't work for
Bengali, and the reason is structural, not a bug on our side.**

**The data is too thin before any filtering happens.** MorphScore's own
README lists a 70-language headline table with an explicit cutoff (>=100
items after filtering) - Bengali is not in that table. Checked why directly
against the HF dataset (`catherinearnett/morphscore`) rather than assumed:
`ben_beng` has only 102 raw rows total. After MorphScore's own filtering
chain (unique wordforms only, stem must equal lemma, single-token and
single-morpheme words excluded by default since they're not a real boundary
decision), the number of items MorphScore can actually SCORE per tokenizer
ranges from 1 to 15 across everyone tested. `bn-bpe-64k` and BMBT both land
at n=2. A tokenizer's real-world Bengali morphological alignment cannot be
concluded from 2 data points, in either direction, regardless of what the
number says.

**Two real methodology bugs were caught before trusting any of these
numbers**, neither hypothetical:

1. MorphScore reconstructs each predicted morpheme boundary by summing
   decoded-token character lengths and comparing against the gold wordform's
   own length. Our tokenizers recompose decomposed consonant+nukta sequences
   (ড়/ঢ়/য়) to their NFC-excluded singleton inside `normalize()` (see point
   in the earlier bugs section above) - so feeding an unnormalized UD
   wordform through our tokenizer and comparing lengths against the
   unnormalized gold text silently misaligns every boundary in a word
   containing those letters. Fixing this by normalizing the wordform/stem/
   lemma/preceding_part/following_part fields before scoring was the right
   call for our own two tokenizers, since that's exactly what their
   `encode()` does internally regardless of what's fed in.

   Applying that SAME fix to every EXTERNAL tokenizer was a second, separate
   bug, caught by checking the raw UD source text directly rather than
   trusting the first fix: every raw `ben_beng` wordform containing this
   letter uses the DECOMPOSED spelling (verified: U+09AF U+09BC, never the
   singleton U+09DF). Our singleton recomposition is not standard Unicode
   NFC - it's a deliberate, documented exception to a real NFC composition
   exclusion, specific to this project's own normalize() - so no external
   tokenizer's vocabulary was ever built expecting it. Force-normalizing
   external tokenizers' input to our singleton made every word containing
   that letter fully out-of-vocabulary for them, an artifact of this
   script's own preprocessing, not a real property of their tokenizers.
   Fixed by scoring our two tokenizers against a normalized copy of the
   dataset and every external tokenizer against the untouched raw copy.

2. MorphScore filters decoded tokens against `tokenizer.special_tokens_map`
   by STRING equality - which also strips a genuine `[UNK]` token whenever a
   whole word is out-of-vocabulary for a given tokenizer. A fully-OOV word
   then decodes to zero non-special tokens, and MorphScore's own
   macro-average divides by that zero: a real `ZeroDivisionError` inside
   MorphScore's own code, first mistaken for "these two tokenizers are
   unavailable" before being traced to its actual cause. Fixed in
   `scripts/morphscore_eval.py` by detecting fully-OOV wordforms per
   tokenizer beforehand and excluding just those from that tokenizer's
   scoring run, logged and disclosed (`রিক্সায়` for mBERT, 13 words
   containing ড়/য় for BanglaBERT), rather than silently swallowed as
   "unavailable."

**Result, all real, all with the small-N caveat that governs everything
here** (`benchmarks/morphscore.json` has the full per-model breakdown
including precision, micro/macro F1, and token-char-ratio):

| Tokenizer | recall | n |
|---|---|---|
| bn-bpe-64k / BMBT (tied) | 0.500 | 2 |
| XLM-RoBERTa | 0.667 | 12 |
| Krutrim | 0.467 | 15 |
| mBERT | 0.500 | 14 |
| IndicBERTv2 | 0.500 | 4 |
| BrahmicTokenizer-131K | 0.429 | 14 |
| Sarvam-1 | 0.125 | 8 |
| SUTRA | 0.000 | 15 |
| BanglaT5 | 1.000 | 1 |
| BanglaBERT | 0.000 | 1 |

Every row here is a genuinely computed number, not fabricated or estimated -
but n=1 through n=15 means none of them, including ours, should be read as a
real ranking. BanglaT5's "1.000" is one single word scored correctly, not a
claim BanglaT5 is the best morphological tokenizer for Bengali. This section
exists to record that MorphScore was tried, why it doesn't produce a usable
number here, and the two real bugs the attempt caught - not to promote a
result. Not added to either README, the paper, or the website: nothing here
clears the bar of being a real finding worth propagating, per the precedent
set by the BMBT-Hybrid and unigram-vs-BPE experiments above.

**UPDATE 2026-08-18: two banglish tier-3 fixes attempted, one real gain measured, one honest zero.**
The 30k-step bigger config was not retrained this pass (measured at ~0.1
steps/s on CPU, ~83 hours for a full run - not viable on CPU alone), so
this pass targeted what CPU-only, no-retrain work could actually move:

- **Real fix, measured**: `bntok/banglish_synth.py`'s reverse phonetic table
  had the same class of bug already fixed once for য (ya-phala) - ব as a
  chained (non-initial) conjunct consonant (স্ব, শ্ব, ...) was rendering the
  literal "b" sound instead of the real labial glide "w" real spellings use
  (Dakshina dev split: "swamijir"/"biswash", not "sbamijir"/"bisbaas").
  Fixed the same way (`_BA_GLIDE_LATIN`, `i > 0` special case). Measured with
  a new `scripts/validate_banglish_synth.py` (reproduces the methodology
  behind the table above using `artifacts/banglish-translit-data/dev.tsv` as
  the real-spelling reference instead of re-downloading Dakshina): word-level
  hit rate **77.0% -> 77.8%** on a 2,500-word sample, 15 seeds each. Small
  but real and directly traced to a concrete mismatch, same standard as the
  three fixes already in the table above. Regression tests in
  `tests/test_banglish_synth.py`.
- **Real code, zero measured effect**: added a structural re-rank to
  `TranslitTransformer.beam_decode` (`bntok/banglish_tier3.py`) - among the
  beam's surviving candidates, prefer the one with fewest orphan-diacritic
  chunks (a matra/modifier/virama that `bntok.akshara.aksharas()` could not
  attach to a base consonant/vowel), tie-broken by the original score. Wired
  on by default in `load_tier3_fn`'s `tier3_fn` (free: re-orders candidates
  already computed, no new model calls). **Measured on 600 real dev-split
  words against the surviving `step-20000.pt` checkpoint: 0 words where this
  changed the output, 0 orphan-diacritic violations found in any of the
  600 no-rerank outputs to begin with.** Root-caused, not left a mystery: the
  checkpoint's actual failure mode (e.g. the "shopno -> শপন" case already
  documented above, correct is স্বপ্ন) is *confidently wrong but
  structurally well-formed* - it picks a valid, complete, wrong conjunct
  shape, not a malformed one - so a validity-only re-rank has nothing to
  correct here. Kept as real, harmless defensive code (a genuinely malformed
  decode is still possible on other inputs/checkpoints, and this catches it
  for free), but it is not the fix for the diagnosed root cause. That
  diagnosis, unchanged by this session: undersized/partly-synthetic training
  data (379,711 pairs, ~75% of the table's own synthetic feeder verified
  accurate) plus training fully from scratch with no pretraining, with no
  akshara/conjunct-aware structural prior in the generation step itself
  (character-level output, no bias toward valid virama-chains) - the built,
  unrun bigger config (24.9M params, 30k steps) remains the next real lever.

## How to report a new issue

Open an issue on the repository with the exact input, the command, and the
versions. Reproducibility is the point.
