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
   (`_is_clean_bengali_line`), as a formal/book-register proxy. Those PDFs are
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

## How to report a new issue

Open an issue on the repository with the exact input, the command, and the
versions. Reproducibility is the point.
