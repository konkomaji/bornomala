# BMBT morphology: the second half of v2 roadmap step 5

`docs/bmbt-architecture.md` describes BMBT as grammar plus featural
decomposition plus a statistical BPE layer, and records that morphology was
deliberately not built. This document covers the layer that was missing, the
measurement that constrains what it can achieve, and what it costs.

## Why a rule-based layer

`bntok/morphology.py` is a rule-based suffix analyser, not a statistical
segmenter. Morfessor or an unsupervised alternative would have been less work.
It was rejected on the same grounds the akshara parser was written by hand
rather than delegated to `\X`: BMBT's claim is that it reads Bengali by the
language's own rules instead of inferring them from counts, and a morphology
layer learned from frequency would abandon that claim at exactly the point
where it matters most.

Bengali inflection is overwhelmingly suffixal, so right-to-left longest-match
stripping is a defensible first approximation. The inventory covers nominal
case, plural, classifiers and determiners, the common verb-conjugation
endings, productive derivational suffixes, and the emphatic clitics.

## What it does not cover, stated plainly

- **Prefixes** (`অ-`, `নি-`, `বি-`, `সু-`, `প্রতি-`, `উপ-`, `অনু-`). Largely
  tatsama borrowings from Sanskrit; stripping them safely needs a stem lexicon
  this project does not have.
- **Sandhi**, where the stem itself changes shape at the boundary. `করা + ছি`
  surfaces as `করছি` with a lost vowel. The layer finds `-ছি` and leaves the
  altered stem as it is rather than trying to restore an underlying form.
- **Compounding** (সমাস). Needs a lexicon.
- **Lexical ambiguity.** `-এ` is a locative case marker and also the final
  akshara of many uninflected stems. Nothing here can separate those without a
  lexicon or context.

The guards are deliberately asymmetric. Over-stripping is treated as the worse
error, because it asserts a morpheme boundary that does not exist, and
morphological-alignment scoring counts a false split against the tokenizer.

## Two bugs the inventory had, found by running it on real forms

Neither was caught by unit tests, because the tests would have been written
from the same wrong assumptions. Both surfaced on the first pass over real
Bengali words, which is the same way the akshara parser's three bugs surfaced.

1. **A two-akshara minimum stem is not safe, it is wrong.** The floor was set
   to 2 on the stated assumption that it "keeps every common monosyllabic verb
   root intact". False: `কর` is two aksharas, but `যা` and `দে` are **one**
   each, because a consonant plus its matra is a single akshara rather than
   two. The floor refused to strip `যাবেন` into `যা` + `বেন` and produced
   `যাব` + `েন` instead, inventing a boundary in the middle of the
   future-tense marker. `বাংলায়` failed the same way. Floor lowered to 1.

2. **Lowering the floor then broke noun stems, and the fix was grammatical
   rather than numeric.** With a floor of 1, `ছেলেরা` segmented as `ছে` +
   `লে`[verb] + `রা`[plural]. `লে` is a real past-tense ending and `ছেলে`
   really does end with those codepoints, so no length threshold separates
   them. The fix is a suffix ordering constraint: ranks run outermost to
   innermost (clitic, case, plural, classifier, then derivational or verb) and
   may never decrease moving inward, and a verb ending admits nothing nominal
   outside it. `করছিও` is well formed; `করছিরা` is not. A finite verb cannot
   be pluralised, so the parse is rejected on grammatical grounds.

Both are regression tests in `tests/test_morphology.py`.

## The constraint that decides the design

BMBT's atoms are aksharas and it may never split one. It can therefore only
place a boundary where an akshara boundary already exists. **How often does a
Bengali morpheme boundary actually fall there?**

Measured over 80,000 held-out Bengali Wikipedia words, 39,928 morpheme
boundaries:

| | Count | Share |
|---|--:|--:|
| Boundary lands **on** an akshara boundary | 24,904 | **62.4%** |
| Boundary lands **inside** an akshara | 15,024 | **37.6%** |

By suffix class:

| Suffix class | Reachable | Unreachable | Reachable share |
|---|--:|--:|--:|
| Clitic | 1,255 | 0 | 100.0% |
| Classifier | 2,034 | 193 | 91.3% |
| Plural | 2,516 | 289 | 89.7% |
| Verb | 6,489 | 1,630 | 79.9% |
| Derivational | 2,225 | 2,029 | 52.3% |
| **Case** | 10,385 | 10,883 | **48.8%** |

The unreachable boundaries are overwhelmingly one codepoint away (9,636 of
15,024 sit exactly one codepoint right of the nearest akshara boundary), and
the cause is a single orthographic fact: **a matra binds orthographically to
the consonant before it while belonging morphologically to the suffix after
it.** `বিশ্বের` breaks morphologically at `বিশ্ব|ের`, but its aksharas are
`বি|শ্বে|র`, so the seam falls inside `শ্বে`. Case markers are the most
frequent suffix class in Bengali and the worst affected, at 48.8%.

### What follows from it

- **The alignment ceiling is about 62%, and it is a property of the script,
  not of this implementation.** No amount of work on the suffix inventory
  raises it while the akshara guarantee holds.
- **v1 has the identical ceiling.** Grapheme-cluster boundaries and akshara
  boundaries are near-isomorphic on well-formed Bengali, so `bn-bpe-64k` is
  subject to exactly the same limit.
- **Byte-level tokenizers that break conjuncts can reach those boundaries.**
  GPT-4o's o200k, BrahmicTokenizer-131K and the rest are under no constraint
  here. They may therefore score *better* on morphological alignment while
  fragmenting a fifth to a third of Bengali conjuncts. These two quality axes
  are in genuine tension, and any morphological-alignment result this project
  publishes has to report both rather than the flattering one.

### Boundaries that cannot be placed correctly are skipped

Snapping an unreachable boundary to the nearest akshara boundary was
considered and rejected. It would assert a morpheme seam one codepoint away
from the real one, and a boundary in the wrong place is worse than no boundary
at all, for the model and for alignment scoring alike. `_morph_barriers` emits
a barrier only where a morpheme boundary and an akshara boundary already
coincide.

## How the constraint reaches BPE

Training inserts a reserved marker codepoint at each qualifying morpheme seam
in the atom string. The pre-tokenizer becomes a sequence of the existing
Metaspace word-boundary handler and a `Split` on that marker with
`behavior="removed"`, so BPE never sees a pair spanning a seam and can never
learn a merge across one.

The marker is never in the vocabulary and never becomes a token. It is
discarded before training and before encoding, so decode is untouched and
round-trip holds exactly, which the tests assert directly.

## Expected cost, recorded before measurement

`docs/design/FORMAL_SPEC.md` proves a BPE constrained to respect additional
boundaries cannot beat an unconstrained one on raw token count. Morpheme
boundaries are additional boundaries. **Morphology is therefore expected to
make fertility slightly worse, not better.** This is written down here in
advance so the result cannot be presented as a surprise in either direction.

What it is meant to buy is different in kind: boundaries that fall where
Bengali's morphemes fall, so a rare inflected form shares its stem with every
other form of the same word rather than being an unrelated token. That is a
generalisation property, invisible to fertility, and measurable only by
morphological alignment and, eventually, by downstream modelling quality.

## Coverage on real text

Over 60,000 held-out Wikipedia words, the layer segments 45.1% of them and
finds 29,852 suffixes: case 15,730, verb 6,182, derivational 3,360, plural
2,104, classifier 1,534, clitic 942. Case dominance is expected for Bengali.

`coverage()` is exposed so this stays checkable rather than assumed: a layer
that almost never fires is adding nothing, and one that fires on nearly every
word is over-stripping. Both failure modes are visible in that report.

## Known limitation kept visible

`কলকাতা` is wrongly segmented as `কলকা` + `তা`[derivational]. Kolkata is a
proper noun, not a nominalised stem, and only a stem lexicon or context can
tell the difference. `tests/test_morphology.py` asserts the current behaviour
directly, so the limitation appears in the suite rather than only in prose,
and a future stem lexicon will show up there as a deliberate change rather
than a surprise.

## Reproduce

```bash
python -m bntok bmbt-train --corpus-config configs/bpe-64k.json --morphology --out artifacts/bmbt-64k-morph
```
