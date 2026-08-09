# Changelog: Track A Bengali tokenizer (`bntok`)

All notable changes to the Track A tokenizer are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Track A2: corpus dedup and quality filtering (`bntok/dedup.py`), Gate
  G3.** Exact dedup, MinHash-LSH near dedup (`datasketch`), and a
  rule-based quality filter. Measured on real data across four sources
  (Bengali Wikipedia, Sangraha, AI4Bharat IndicCorp v2 at 2M lines,
  CC-100 at 1M lines as the genuine raw-web proxy): survival ranges from
  63.2%/79.6% (CC-100, the real outlier) to 97.9%/99.3% (IndicCorp v2),
  comfortably clearing Gate G3's threshold on every source measured. Full
  writeup: `docs/track-a2-corpus-survival.md`. Now wired into
  `build_configured_corpus` as an opt-in step (`dedup=True`; CLI `--dedup`
  on `train`/`bmbt-train`/`hybrid-train`), applied per-source before
  weighting so it does not interfere with `weighted_corpus`'s deliberate
  cycling of thin sources. Default off - does not change either shipped
  artifact (`bn-bpe-64k`/`bmbt-64k`), which has not been retrained with
  it. 18 new tests (`tests/test_dedup.py`).
- **BMBT (Bornomala's Bengali Tokenizer, v2 roadmap step 5, partial):
  `bntok/bmbt.py`.** The recommended, primary tokenizer as of this
  release. Grammar (the akshara finite-state parser, reused unchanged)
  plus a real featural decomposition (`featurize()`: onset consonants,
  which carry a nukta, vowel, modifiers, ZWJ/ZWNJ flags) plus a
  statistical BPE layer over akshara atoms - the same architecture as v1
  with the atomic unit swapped from grapheme cluster to akshara.
  **Morphology is explicitly not built** - deferred, not abandoned.
  `bmbt.py` is fully self-contained: it imports nothing from `atoms.py` or
  `tokenizer.py`, so v1 (`bn-bpe-64k`) is completely unaffected -
  `tests/test_tokenizer.py` passes unmodified. New CLI: `bmbt-train`,
  `bmbt-encode`, `bmbt-evaluate`, `bmbt-featurize`. 24 new tests
  (`tests/test_bmbt.py`), including a property test proving `featurize()`
  reconstructs the original akshara text exactly (lossless, not just
  "looks right on examples").
  **Trained on the identical corpus as `bn-bpe-64k` (`configs/bpe-64k.json`,
  vocab 64000) and measured the same way, reported exactly as it came
  out**: BMBT *ties* v1, it does not beat it. On Wikipedia held-out the
  two are identical down to the raw token count (17,245 tokens, 11,316
  words) despite genuinely different vocabularies (12,233 atoms for v1,
  12,199 for BMBT). On the other three registers, tiny real differences
  appear in both directions (BMBT needs marginally fewer tokens,
  marginally more fragmented clusters - both within 0.02% either way).
  This confirms, in practice, `FORMAL_SPEC.md`'s own proof that a
  grammar-constrained BPE cannot beat an unconstrained one on raw token
  count. What BMBT adds regardless is the featural output, at no fertility
  cost. Full numbers and a CC-100 corpus ablation (trained again with
  CC-100 added: very slightly worse Wikipedia fertility, very slightly
  better general-web fertility, a wash): `benchmarks/bengali-comparison.md`.
  New docs: `docs/bmbt-architecture.md`.
- **Real bug found and fixed while training BMBT: `stream_cc100` never
  actually worked.** It had been wired in and documented as "available"
  in an earlier change but never exercised end to end; it called
  `datasets.load_dataset("cc100", ...)`, which fails outright on any
  current `datasets` install (the `cc100` Hub repo still ships a legacy
  loader script, unsupported by current `datasets`). Fixed by bypassing
  `datasets` entirely (list and read the repo's auto-converted parquet
  shards via `huggingface_hub` + `pandas`, the same mechanism
  `stream_wikisource`/`stream_xlsum` already use). A second, unrelated
  stall surfaced while fixing this: `huggingface_hub.hf_hub_download`
  (0.36.2) hangs indefinitely on this specific repository's files -
  verified directly (a plain HTTP GET to the identical URL succeeds
  immediately) rather than assumed; `_download_cc100_shard` bypasses
  `hf_hub_download` for CC-100 only, via plain `requests` streaming to a
  local cache. Every other `stream_*` function is unaffected. See
  `docs/known-issues.md` point 15.
- **Two more real external baselines wired into `scripts/compare.py`'s
  `HF_MODELS`: SUTRA (`TWO/sutra-mlt256-v2`) and Krutrim
  (`krutrim-ai-labs/Krutrim-2-instruct`)**, both verified loadable before
  adding. On the Wikipedia held-out set: SUTRA fertility 2.218 (3rd best,
  after ours and IndicBERTv2), Krutrim 3.207 (worst of the field). The two
  other baselines the v2 design doc's roadmap names, IndicSuperTokenizer
  and BengaliBPE, have no usable public release (checked directly, not
  assumed) and are reported as unavailable rather than faked with a
  similarly-named but unverified stand-in. Root `README.md`,
  `bengali-tokenizer/README.md`, and the project website's comparison
  tables updated to match; also fixed a stale `0.0004` conjunct-fragmentation
  figure for our own row in `bengali-tokenizer/README.md` (should have been
  `0.0001`, a pre-existing error unrelated to this change). Full detail:
  `benchmarks/bengali-comparison.md`.
- **`docs/design/`: v2 design record, not yet built.** [*Reading Bengali on
  Its Own Terms*](docs/design/reading-bengali-on-its-own-terms.md), a
  literature-grounded position paper (surveying ~30 works) arguing that
  retraining BPE on Bengali fixes the vocabulary but keeps the English-shaped
  frame, and proposing instead a grammar-first tokenizer: a finite-state
  parser for the akshara (orthographic syllable) as the primary segmenter,
  emitting featural tokens (onset/vowel/modifier), with statistics demoted to
  a fallback for loanwords and noise. Companion
  [`docs/design/FORMAL_SPEC.md`](docs/design/FORMAL_SPEC.md) states
  losslessness, totality, linear time, and constrained optimality as formal
  predicates, proves three by construction, and specifies the property-based
  fuzzer contract for the fourth. This supersedes nothing shipped; `bn-bpe-64k`
  remains the current artifact until v2 is built and measured against the same
  benchmarks.
- **v2 roadmap steps 1-3 built: `bntok/substrate.py` and `bntok/akshara.py`.**
  `substrate.py` is the single source of truth for the Bengali Unicode
  inventory (consonants, vowels, matras, modifiers, plus matra visual-position
  metadata for future featural encoding); `graphemes.py` now imports from it
  rather than defining its own private copies (pure refactor, zero behaviour
  change). Also documents Bengali-block letters not in everyday modern use
  (archaic VOCALIC RR/LL vowels and their matras; RA WITH MIDDLE/LOWER
  DIAGONAL, rare in standard Bengali, used in Assamese) instead of leaving
  them to fall through unnamed.
  `akshara.py` adds `aksharas(text) -> list[Akshara]`: a finite-state parser
  for the akshara grammar (`Consonant Nukta* (Virama ZWJ? Consonant Nukta*)*
  (Virama ZWNJ?)? Matra? Modifier* | Vowel Modifier*`), refined from the
  design doc's simplified grammar after checking it against `regex`'s own
  `\X` behaviour (ZWJ continues a conjunct, ZWNJ terminates it explicitly;
  Nukta can repeat). Lossless (pure segmentation, never rewrites), total
  (never raises for any `str` input), deterministic, and linear (fixed
  integer offsets, no per-step re-slicing). New CLI: `python -m bntok akshara
  --text "..."`. 95 tests total (was 15): `tests/test_substrate.py` (a
  regression tripwire for the moved Unicode ranges) and
  `tests/test_akshara.py` (a property/round-trip suite covering all 7 fuzzer
  input classes from `FORMAL_SPEC.md` section 7.1, including the design doc's
  own named hard words স্ত্রী/ক্ষ্ম/আকাঙ্ক্ষা/ঋত্বিক). See
  `docs/known-issues.md`'s "Roadmap: a proposed v2" section for the two
  documented divergences from `\X` this uncovered, and what step 4/5 (measured
  benchmarking; featural encoding and morphology) still need before anything
  here changes what `bn-bpe-64k` ships or claims.
- **v2 roadmap step 4 measured (Wikipedia held-out only): `scripts/compare.py`
  gains `measure_akshara()`.** Runs `bntok.akshara.aksharas()` over the same
  828-line held-out set `bn-bpe-64k` is benchmarked on, reported as its own
  section (not folded into the fertility-sorted table, since the parser has
  no vocabulary or merges yet, so its chunk count is a different kind of
  number, not a like-for-like token count). Headline:
  fertility 4.527 (vs `bn-bpe-64k`'s 1.524, expected pre-compression), and
  conjunct fragmentation **0.0000 exactly**, better than `bn-bpe-64k`'s own
  0.0001. Full writeup: `benchmarks/bengali-comparison.md`'s "v2 roadmap step
  4" section.
  Running this measurement against real text (not only synthetic tests)
  found 3 real bugs in `akshara.py`'s grammar, now fixed: an independent
  vowel followed by a virama does not chain into a further consonant the
  way a consonant does; a Modifier (not Matra or Nukta) blocks
  conjunct-chain continuation; and ZWJ/ZWNJ are not tied to a fixed
  position relative to the virama the way the first version assumed. The
  grammar is now a single unified tail-scan (`_scan_tail` in `akshara.py`)
  rather than the rigid ordered sequence from the previous entry above; see
  `docs/known-issues.md` points 11-13 for the full account, including the
  intermediate 0.0012 fragmentation number this replaced. Test suite grew
  from 95 to 103 (6 new regression tests for these cases, plus 2 more for
  ZWJ/ZWNJ position).
- **v2 roadmap step 4 extended to the other three held-out registers**
  (literary/formal, general web, news - same held-out sets as `bn-bpe-64k`'s
  own per-register results). Conjunct fragmentation measured **exactly
  0.0000 on general web and news**; literary/formal measured 0.000005
  (13 out of 2,868,557 clusters). Investigated rather than left unexplained:
  all 13 are a Bengali consonant directly followed by a Devanagari combining
  mark (U+093E/U+093F/U+0902), almost certainly OCR/font-mapping noise in
  the scanned 19th/20th-century literature this register draws from. This
  is a deliberate scope boundary, not a bug - `akshara.py`'s Matra/Modifier
  sets are Bengali-block only by design, so it correctly does not absorb a
  foreign-script mark into a Bengali consonant's chunk; widening the
  grammar to do so would blur what a Bengali akshara grammar means and mask
  real encoding noise. See `docs/known-issues.md` point 14 and
  `benchmarks/bengali-comparison.md`'s step 4 section for the full 4-register
  table and account.
- **`stream_cc100` in `corpus.py`**: streams CC-100 Bengali (a large,
  2018-vintage CommonCrawl general-web corpus, same source used by the
  `nawaz0x1/Bengali-BPE-Tokenizer` baseline). Wired into
  `build_configured_corpus` as an available source, `cc100_general_web`, not
  part of the shipped `configs/bpe-64k.json` weights until it is
  retrained-and-benchmarked in. See `docs/known-issues.md` point 10.
- **arXiv preprint** (`paper/`): self-contained LaTeX source, submission guide,
  version 0.1 (preliminary, Wikipedia; to be updated with larger datasets).
- **Hugging Face release** (`huggingface/`): upload-ready model card and
  tokenizer files, with a publishing guide.
- **Literary-weighted corpus, for real** (`corpus.py`): `stream_sangraha`,
  `stream_wikisource`, `stream_xlsum`, an OCR-noise filter for scanned literary
  text, and `build_configured_corpus`/`build_register_held_out` to assemble and
  hold out a real 1.5M-line mix (Wikisource, AI4Bharat Sangraha, Wikipedia,
  XL-Sum). `government_administrative` and `code_mixed_bn_en` have no clean
  public dataset and are dropped with weights renormalised over the rest.
  `bntok train --corpus-config` wires it into the CLI. Full writeup:
  `docs/known-issues.md` points 6 and 8.
- **`docs/bengali-script-reference.md`**: full Bengali Unicode block inventory
  and grapheme-cluster/conjunct rules, verified against Python's `unicodedata`
  directly rather than recalled from memory.
- Register-specific held-out comparison: `scripts/compare.py --register
  {literary_formal,general_web,news}`.

### Changed
- **New official artifact: 64k vocabulary.** An ablation across 32k/48k/64k on
  the literary-weighted corpus showed fertility recovering monotonically with
  vocabulary size; 64k is the smallest size that beats IndicBERTv2 on fertility,
  STRR, *and* conjunct fragmentation across every held-out register tested
  (Wikipedia, literary/formal, general web, news). `artifacts/bn-bpe-64k/`
  replaces `artifacts/bn-bpe-32k/`; `configs/bpe-64k.json` replaces
  `configs/bpe-32k.json`; the CLI's default `--vocab-size` is now 64000.
  Full numbers: `benchmarks/bengali-comparison.md`, `docs/known-issues.md`
  point 7.
- `graphemes.py`: `GUARANTEED_CODEPOINTS` now includes Bengali's shared (non-
  Bengali-block) sentence punctuation, danda and double danda (U+0964/U+0965),
  which were previously covered only by corpus frequency, not by guarantee.
- `normalize.py`: re-composes RRA/RHA/YYA (ড়/ঢ়/য়) to their single dedicated
  codepoint after NFC, a minor efficiency improvement (NFC already unified both
  spellings, just onto the decomposed form — see `docs/known-issues.md`
  point 7 and `docs/bengali-script-reference.md` §3 for what this does and does
  not fix).

### Fixed
- `scripts/compare.py`'s conjunct-fragmentation counter for our own tokenizer
  had two real measurement bugs, both caught via a hard assertion added
  specifically to catch this class of mistake: an off-by-one from the
  Metaspace word-boundary marker's leading space, and `encode_tokens()`
  returning the literal string `<unk>` in place of missing text for
  out-of-coverage foreign-script codepoints. Full account:
  `docs/known-issues.md` point 8.
- `IndicBERTv2` comparison baseline had regressed to the gated `ai4bharat/
  indic-bert` (v1); restored to the ungated `ai4bharat/IndicBERTv2-MLM-only`
  the benchmark doc always claimed.

## [0.1.0] - 2026-07-23

First working version of the Project Bornomala Track A tokenizer. CPU only.

### Added
- **Grapheme-cluster-aware core** that never splits a conjunct across a token
  boundary (conjunct fragmentation rate = 0 by construction):
  - `normalize.py`: NFC normalisation (UAX #15) and a documented, preserve-by-
    default ZWJ / ZWNJ policy, with canonical khanda-ta handling.
  - `graphemes.py`: UAX #29 grapheme-cluster segmentation and Bengali structural
    predicates (conjunct, reph, ya-phala, ra-phala, nukta), plus guaranteed
    coverage sets (Bengali block and ASCII).
  - `atoms.py`: a reversible grapheme-cluster to Private-Use-Area atom map with a
    codepoint decomposition fallback for rare or unseen clusters.
  - `tokenizer.py`: `BengaliTokenizer`, training BPE or Unigram over atoms with a
    Metaspace word-boundary marker; encode, decode, save, load, and a round-trip
    self-check. Every atom is forced into the base vocabulary so covered text
    always round-trips.
- **Evaluation** (`evaluate.py`): fertility, STRR, bytes per token, grapheme
  clusters per token, conjunct fragmentation rate, and round-trip fidelity.
- **Shaping validation** (`shaping.py`): HarfBuzz coverage and cluster-
  correspondence check (Requirement A-1, Gate G1), with system-font auto-detect.
- **Corpus loading** (`corpus.py`): local files and directories, optional Bengali
  Wikipedia streaming, and a literary-weighted source combiner.
- **CLI** (`python -m bntok`): `gate-g1`, `train`, `evaluate`, `encode`.
- **Robustness**: a typed error hierarchy (`errors.py`); every entry point
  validates inputs and fails with a clear message, never a silent corruption or a
  raw traceback (mitigates risk R1).
- **Configs** for the whitepaper ablation grid (BPE and Unigram; 16k, 32k, 48k,
  64k; web-natural vs literary-weighted).
- **Tests**: grapheme integrity, zero fragmentation, round-trip (Bengali and
  code-mixed), save/load, and the error paths.
- **Benchmark**: a measured cross-tokenizer Bengali comparison
  (`scripts/compare.py`, `benchmarks/`) against AI4Bharat IndicBERTv2, Sarvam-1,
  XLM-RoBERTa, GPT-4o, mBERT, and DeepSeek-V3 on held-out Bengali. The Bornomala
  tokenizer leads on fertility (1.39), single-token retention, bytes/token, and
  conjunct fragmentation (0.0006 vs 4 to 30 percent for the others).
- **Artifact**: a trained BPE 32k tokenizer checked in at `artifacts/bn-bpe-32k/`.
- **Docs**: architecture reference with diagrams (`docs/architecture.md`).

### Notes
- Induction is CPU-bound and runs anywhere. Large corpus assembly and the full
  ablation grid run on the training machine.
- This is the tokenizer itself (Track A). It is distinct from the
  MotherTongueIndex subproject, which benchmarks tokenizers.

[0.1.0]: https://github.com/konkomaji/bornomala/tree/main/bengali-tokenizer
