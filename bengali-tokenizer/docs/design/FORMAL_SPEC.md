# Formal Specification

### The Bornomala Bengali Tokenizer: Mathematical Contract

**Author:** Konko Maji ([@konkomaji](https://github.com/konkomaji), work.konkomaji@gmail.com)
**Programme:** Bornomala — Bengali-first Language Technology Research
**Companion to:** *Reading Bengali on Its Own Terms* (this directory)
**Status:** Formal spec, v1
**Date:** 24 July 2026

---

## 0. Purpose

This document pins the tokenizer down as mathematics. It states four properties, lossless, reliable, fast, optimized, as formal predicates; proves the ones that follow by construction; and specifies the property-based test contract for the one that must be checked rather than proved. Every claim here is meant to be executable: an assertion a fuzzer can pass or fail, not an adjective.

The single governing rule from which most of this follows:

> **The tokenizer segments; it never rewrites. Normalization lives in a residual channel. A byte-level fallback guarantees totality. A regular grammar guarantees linear time. Statistics optimize only inside the constraint box.**

---

## 1. Objects and signatures

Let `Σ` be the set of bytes, `|Σ| = 256`. Any input text is a finite byte string:

```
s ∈ Σ*
```

The tokenizer is a pair of total functions over a token set `T`:

```
encode : Σ* → T*
decode : T* → Σ*
```

A **token** is not a flat integer. It is a factored record:

```
t = ( logical_id , residual )        t ∈ T
    logical_id ∈ L                   (normalized, model-facing unit)
    residual   ∈ R                   (restoration delta; R contains the empty delta ε)
```

`L` is the model vocabulary (what the language model actually sees and predicts). `R` is the set of restoration deltas, with distinguished empty element `ε`.

Two projections:

```
π_bytes : T* → Σ*     surface reconstruction (used by decode)
π_model : T* → L*     model view (logical_ids only)
```

---

## 2. Property I — Losslessness

### 2.1 Definition

```
(LOSSLESS)      ∀ s ∈ Σ* :  decode(encode(s)) = s
```

Equivalently, `decode ∘ encode = id_{Σ*}`, i.e. `encode` is **injective** and `decode` is a left inverse of it.

### 2.2 The segmentation lemma (why it is nearly free)

Call `encode` a **segmentation** if for every s it returns tokens whose surface bytes, concatenated in order, reproduce s:

```
(SEG)   π_bytes(encode(s))  =  c₁ · c₂ · … · cₖ  =  s        (· = concatenation)
```

**Lemma 1.** If `encode` satisfies (SEG) and `decode := π_bytes`, then (LOSSLESS) holds.

*Proof.* `decode(encode(s)) = π_bytes(encode(s)) = c₁·…·cₖ = s` by (SEG). ∎

So losslessness is discharged the moment the tokenizer is a pure segmentation: it only cuts, never edits, reorders, inserts, or deletes bytes. This is the founding rule as a theorem.

### 2.3 The normalization obstruction

We want normalization for efficiency: collapse distinct byte sequences that denote the same visual/logical unit into one `logical_id`, so the model sees one token, not many.

```
normalize : Σ* → Σ*        (canonicalization)
```

But normalization is **many-to-one**, hence not injective:

```
∃ x ≠ y :  normalize(x) = normalize(y)          e.g. variant encodings of ড়
```

If a token carried only `logical_id = normalize(chunk)`, then `encode` would inherit this collision and (LOSSLESS) would fail.

### 2.4 The residual channel (the resolution)

Factor the token so that the pair preserves all information the surface had. For a chunk `c` with canonical form `ĉ = normalize(c)`:

```
logical_id(c) = enc_L(ĉ)                    (normalized, model-facing)
residual(c)   = diff(c, ĉ)                  (delta from canonical back to surface)
```

Require a restoration operator `restore` such that:

```
(RESTORE)   ∀ c :  restore( ĉ , diff(c, ĉ) )  =  c
            and    diff(c, ĉ) = ε   ⇔   c is already canonical
```

Then define surface reconstruction per token and extend by concatenation:

```
π_bytes( (logical_id, residual) ) = restore( dec_L(logical_id) , residual )
```

**Theorem 1 (Lossless under normalization).** With tokens `(logical_id, residual)` defined as above and `decode := π_bytes`, (LOSSLESS) holds even though `normalize` is many-to-one.

*Proof.* For each chunk c, `π_bytes(encode-of-c) = restore(dec_L(enc_L(ĉ)), diff(c,ĉ)) = restore(ĉ, diff(c,ĉ)) = c` by (RESTORE) and `dec_L ∘ enc_L = id_L`. Concatenating over chunks and applying Lemma 1 gives `decode(encode(s)) = s`. ∎

**Interpretation.** Normalization does not destroy information; it *relocates* it. The entropy that canonicalization removes from `logical_id` is exactly the entropy stored in `residual`. Because canonical text has `residual = ε`, the model view is compact for the common case, while the decoder remains exact for every case.

**Corollary (model-view efficiency).** `π_model(encode(s))` may be strictly shorter/flatter in variant-space than `π_bytes`, since many surfaces share one `logical_id`. Efficiency and losslessness are not in conflict; they occupy different channels.

---

## 3. Property II — Reliability (totality + determinism)

### 3.1 Totality

```
(TOTAL)     ∀ s ∈ Σ* :  encode(s) is defined      (no exception, no divergence)
```

The domain is **all** byte strings: valid Bengali, English, mixed Banglish, emoji, corrupted or truncated UTF-8, the empty string, a lone dangling hasanta.

### 3.2 Layered coverage

`encode` tries layers in order and takes the first that matches at the current position:

```
Layer A:  akshara grammar        (valid Bengali orthographic syllables; Section 5)
Layer B:  morpheme table         (root + suffix decomposition)
Layer C:  byte-level fallback    (alphabet = Σ; 256 always-valid tokens)
```

**Lemma 2 (Coverage).** At any position i in s, at least one layer emits a non-empty chunk.

*Proof.* Layer C maps the single byte `s[i]` to a byte-token, which is always defined since its alphabet is Σ. ∎

**Theorem 2 (Totality).** `encode` is total on Σ*.

*Proof.* By Lemma 2 the left-to-right scan can always advance by at least one byte, so it terminates after ≤ |s| steps with a complete partition; no position is ever stuck. ∎

Note Layer C also preserves (SEG): a byte-token's surface is that exact byte, so fallback never breaks losslessness. Reliability and losslessness are compatible by design.

### 3.3 Determinism

```
(DET)   ∀ s :  encode(s) evaluated twice yields identical output
```

Guaranteed by: (i) fixed layer priority A → B → C; (ii) a fixed tie-break rule within a layer (specified: **longest match**, then lowest `logical_id`); (iii) no randomness, no wall-clock, no hash-order dependence. The finite-state recognizer is deterministic (a DFA).

---

## 4. Property III — Speed (complexity bounds)

```
(TIME)    time( encode(s) )   = O(n)          n = |s|
(LOOK)    lookahead            = O(1)          bounded by  M = max akshara length (small constant)
(MEM)     working memory       = O(1)          streaming; emit tokens online
```

**Justification.** The akshara language of Section 5 is **regular** (Kleene-star over `Virama Consonant`, finite modifiers), so it is recognized by a DFA. A DFA scan is one pass, no backtracking, constant work per byte, hence O(n). Bengali clusters are bounded in practice (rarely > 4 consonants), so the lookahead window M is a small constant and does not grow with n.

**Constraint on Layer B.** The morpheme layer must also be finite-state (a finite suffix/affix automaton), not an unbounded search. If Layer B were, say, exponential-time analysis, (TIME) would break and the documented latency objection (IndicSuperTokenizer) would return. **Requirement:** every layer is O(n); the pipeline is a composition of finite-state transducers, closed under composition, so the whole `encode` is O(n).

---

## 5. Property IV — Optimality (constrained objective)

Optimality is **not** unconstrained token minimization. It is:

```
minimize      Fertility(encode) := E_{s ~ 𝒟} [ |π_model(encode(s))| / words(s) ]

subject to    (LOSSLESS)                             decode(encode(s)) = s
              (TOTAL)                                encode total on Σ*
              (TIME)                                 O(n)
              (STRUCT)  no token straddles an akshara or morpheme boundary
```

`𝒟` is the target corpus distribution. The statistical fallback (a Bengali-trained BPE inside Layer B/C) performs the minimization, **but only over the feasible set carved out by the hard constraints.** This inverts standard BPE, which minimizes with no constraints and shreds structure.

### 5.1 The fertility tension, formally

Let `F*` be the fertility of unconstrained BPE and `F_c` the fertility of the constrained tokenizer.

**Proposition.** `F_c ≥ F*`.

*Reason.* (STRUCT) removes merges from the candidate set; a minimization over a subset cannot beat minimization over the superset. ∎

This is not a defeat; it is the cost to be justified. The programme's empirical claim is a **multi-objective** one:

```
Value(encode) = quality_per_token(encode) − λ · Fertility(encode)
```

The bet is that featural akshara tokens raise `quality_per_token` (generalization across shared onsets, reused morphemes) by more than the constraint raises fertility. **This must be measured, never assumed** (see the paper, Sections 11–12).

---

## 6. The token algebra (summary of operators)

```
enc_L   : Σ*(canonical) → L          logical encode         dec_L ∘ enc_L = id_L
dec_L   : L → Σ*(canonical)          logical decode
normalize : Σ* → Σ*                  canonicalization (many-to-one)
diff    : (surface, canonical) → R   restoration delta       diff(c,ĉ)=ε ⇔ c canonical
restore : (canonical, R) → Σ*        inverse of diff         restore(ĉ, diff(c,ĉ)) = c

encode(s)          = segment s into chunks (Layers A→B→C), map each c ↦ (enc_L(normalize(c)), diff(c, normalize(c)))
decode(t₁…tₖ)      = π_bytes(t₁) · … · π_bytes(tₖ)
π_bytes(l,r)       = restore(dec_L(l), r)
π_model(t₁…tₖ)     = logical_id(t₁) … logical_id(tₖ)
```

Invariants the algebra must satisfy (checked, see Section 7):

```
INV-1  dec_L(enc_L(x)) = x                    for canonical x
INV-2  restore(ĉ, diff(c,ĉ)) = c              for all c
INV-3  π_bytes(encode(s)) = s                 (SEG) ⇒ (LOSSLESS)
INV-4  every position of s covered            (Lemma 2)
```

---

## 7. Verification contract (how each property is checked, not just claimed)

Most of this is enforced by a **round-trip fuzzer** in CI, not by hand proof.

| Property | Check | Method | Gate |
| --- | --- | --- | --- |
| LOSSLESS | `decode(encode(s)) == s` | property-based fuzzing over generated s | must pass 100% |
| TOTAL | `encode(s)` never raises/hangs | same fuzzer, exception + timeout guard | must pass 100% |
| DET | `encode(s) == encode(s)` across runs/threads/seeds | repeat + cross-process | must pass 100% |
| TIME | tokens-vs-time slope is linear | microbenchmark on increasing n and cluster depth | slope ≈ 1, no superlinear knee |
| STRUCT | no token crosses an akshara/morpheme boundary | assert on segmentation vs reference syllabifier | zero violations |
| OPTIMAL | fertility vs baselines | benchmark on held-out corpus | report vs Sarvam 2.05, SUTRA, BengaliBPE |

### 7.1 Fuzzer input classes (must all round-trip)

```
1. Valid Bengali          — sampled words, deep conjuncts (স্ত্রী, ক্ষ্ম, আকাঙ্ক্ষা, ঋত্বিক)
2. Variant encodings      — nukta/composed forms, ya-phala/ref orderings (exercise residual ≠ ε)
3. Mixed script / Banglish— "আমি busy আছি", URLs, digits ০-৯ and 0-9
4. Adversarial Unicode    — dangling hasanta, lone matra, ZWJ/ZWNJ, combining overflow
5. Non-Bengali            — English, emoji, CJK (exercise Layer C fallback)
6. Corrupted bytes        — truncated/invalid UTF-8, random Σ* (exercise totality)
7. Degenerate            — empty string, single byte, whitespace only
```

A single failing case in classes 1–7 is a **build-breaking defect**, not a warning. Losslessness is binary.

### 7.2 What is proved vs what is fuzzed

- **Proved by construction:** TOTAL (Thm 2), TIME (regularity ⇒ DFA), SEG ⇒ LOSSLESS (Lemma 1), lossless-under-normalization (Thm 1) — *contingent on* INV-1 and INV-2 holding.
- **Fuzzed (because it depends on correct `diff`/`restore`/normalize implementations):** INV-1, INV-2, and therefore end-to-end LOSSLESS. The theorem reduces trust to two small invariants; the fuzzer guards those two.

---

## 8. Open formal questions

1. **Minimal residual encoding.** What is the information-theoretically minimal `R` that satisfies (RESTORE) for the actual variant set of Bengali? Smaller R ⇒ cheaper losslessness.
2. **Layer-B finite-state morphology.** Can the full noun+verb+sandhi morphology be expressed as a bounded finite-state transducer without losing (TIME)? Sandhi ambiguity is the risk.
3. **Constrained-optimal vocabulary.** Given (STRUCT), what vocabulary `L` minimizes `F_c`? This is a constrained vocabulary-allocation problem (cf. Arnett et al., 2025).
4. **Determinism of morpheme segmentation under ambiguity.** When multiple valid morpheme splits exist, the tie-break must be fixed and documented to preserve (DET); which tie-break minimizes fertility?

---

*© 2026 Konko Maji / Bornomala. Formal spec v1; companion to* Reading Bengali on Its Own Terms.
