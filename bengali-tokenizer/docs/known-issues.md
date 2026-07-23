# Known issues, limitations, and the bugs we hit

Transparency is a core value of Project Bornomala. This document records the real
limitations of the current tokenizer, the caveats on the comparison, and the bugs
found and fixed during development. Nothing here is hidden.

## Current limitations

1. **Rare-cluster fragmentation is near zero, not exactly zero.** Grapheme
   clusters seen often enough in the corpus (at or above the atom-frequency
   threshold) get their own atom and can never be split: fragmentation is exactly
   0 for them. Clusters below that threshold decompose into their codepoints, and
   a token boundary can fall between those codepoints. On held-out Bengali this
   affects about 0.0006 of clusters (32 in roughly 248,000). It is a deliberate
   efficiency trade, and it is measured and reported, not hidden. Setting
   `min_atom_freq=1` drives it to exactly 0 for all seen clusters at the cost of a
   larger atom set.

2. **Decode normalises surrounding whitespace.** The word-boundary scheme
   (Metaspace) does not carry exact leading or trailing spaces, so decode returns
   content-exact text with surrounding spaces trimmed. Internal single spaces are
   preserved. This matches how subword tokenizers normally behave; use
   `content_roundtrip_ok` for whitespace-insensitive fidelity checks.

3. **The checked-in artifact is a demonstrator, not the final tokenizer.** It is
   trained on 12,000 Bengali Wikipedia articles. Wikipedia is not weighted toward
   literary and formal register, which is where Bengali conjunct density is
   highest. The final tokenizer is trained on the literary-weighted corpus on the
   training machine; the shipped artifact proves the method and the metrics.

4. **Exotic codepoints outside Bengali and ASCII may not round-trip.** The
   guaranteed coverage set is the Bengali block plus ASCII, which covers Bengali
   and code-mixed English. Emoji or other scripts can map to the unknown atom and
   be dropped on decode. That is out of scope for a Bengali-first tokenizer, but
   it is a real limit, so it is stated.

5. **Unigram support depends on the tokenizers version.** BPE is the default and
   the tested path. Unigram training uses the same atom scheme but relies on the
   installed `tokenizers` version accepting `initial_alphabet` for the Unigram
   trainer.

## Caveats on the comparison

- **IndicBERT v1 is gated.** AI4Bharat's `ai4bharat/indic-bert` now requires
  authentication, so the comparison uses the ungated `ai4bharat/IndicBERTv2-MLM-only`.
  Both are AI4Bharat tokenizers.
- **GPT-4o fragmentation is unmeasured.** tiktoken exposes no character offsets,
  so conjunct fragmentation cannot be computed for it. It is reported as n/a
  rather than guessed.
- **Vocabulary sizes differ.** Our tokenizer is 32k; Sarvam-1 is about 68k;
  the encoders are smaller. Fertility (tokens per word) is a fair, vocab-agnostic
  per-word measure, but a smaller vocabulary beating a larger one is itself part
  of the result, and the difference is noted rather than hidden.
- **The held-out set is Wikipedia.** It is unseen relative to training, which is
  the important control, but it is not literary or dialectal. A literary-register
  and a dialect evaluation set are planned and will be more representative of the
  hardest Bengali.

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

## How to report a new issue

Open an issue on the repository with the exact input, the command, and the
versions. Reproducibility is the point.
