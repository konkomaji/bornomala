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

## The constraint that decides the design, and how it was removed

BMBT's atoms are aksharas and it may never split one. It can therefore only
place a boundary where an akshara boundary already exists. **How often does a
Bengali morpheme boundary actually fall there?**

Measured over 80,000 held-out Bengali Wikipedia words, 39,928 morpheme
boundaries: **62.4% land on an akshara boundary, 37.6% land inside one.**
By suffix class, reachable: clitic 100%, classifier 91.3%, plural 89.7%,
verb 79.9%, derivational 52.3%, **case 48.8%**. Case markers are Bengali's
most frequent suffix class and the worst affected.

That looked like a hard ceiling. It was not. Taking the unreachable set apart
by *what kind of split it would require* showed it is two different things:

| | Count | Share of unreachable |
|---|--:|--:|
| **Onset/rime seam** (`শ্বে` -> `শ্ব` + `ে`) | 12,132 | 80.8% |
| **Inside a conjunct** (`ষ্ট্র` -> `ষ্ট্` + `র`) | 2,892 | 19.2% |

### The intra-conjunct cases were not unreachable, they were wrong

Every one inspected was the analyser over-stripping, not a real seam it could
not place: `জাতিরাষ্ট্র`, `একমাত্র` and `স্তোত্র` split before a stem-final
`র`, and `বিশ্বে` was read as stem plus a future-tense `বে` rather than
`বিশ্ব` plus the locative `ে`.

A Bengali morpheme does not begin in the middle of a conjunct.
`morphology.cuts_inside_conjunct` now refuses any analysis requiring such a
cut. That removed 2,288 proposed boundaries, all false positives, which
**raises precision and empties the "blocked" category at the same time**.

### The onset/rime cases needed a distinction this project had been eliding

Two guarantees had been treated as one:

- **Conjunct integrity** - never sever a virama-joined consonant cluster.
  This is the guarantee that matters, and it is absolute.
- **Akshara atomicity** - never split a consonant cluster from its matra.
  This is an *implementation choice*, strictly stronger than the guarantee
  requires, and it was costing 30.4% of all morpheme boundaries.

`শ্ব` + `ে` yields a valid consonant cluster and a valid vowel sign, both
units Bengali literacy teaches by name. `ক্` + `ষ` yields a fragment
corresponding to nothing. Those are not the same operation, and only the
second is the thing this tokenizer exists to prevent.

So BMBT now factors an akshara at the onset/rime seam **when, and only when, a
morpheme boundary falls there**.

### Result

| | Reachable | Blocked by a conjunct |
|---|--:|--:|
| Akshara atoms, no conjunct guard | 62.4% | 7.2% |
| Conjunct guard only | 66.5% | **0** |
| **Conjunct guard + onset/rime factoring** | **100.0%** | **0** |

Verified directly on 1,468,236 codepoints of held-out Wikipedia: 99,945
morpheme seams forced, **zero conjunct-integrity violations**, and 31,923
akshara splits (3.349% of all grapheme clusters) which are the deliberate,
measured cost.

### The cost, stated rather than buried

A morphology-enabled artifact has a **nonzero grapheme-cluster fragmentation
rate by design** (3.349%), entirely at morpheme seams. Its **conjunct
fragmentation rate remains exactly zero**.

`evaluate.py` now reports both. The long-standing field
`conjunct_fragmentation_rate` in fact counts any split grapheme cluster, not
only severed conjuncts; the name predates this distinction and is kept so that
every already-published number stays comparable. The new
`conjunct_broken_rate` is the strict measure. Reporting only the first would
misread this trade as a regression; reporting only the second would hide a
real cost.

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

## Trained (session 9, 2026-08-18)

`artifacts/bmbt-64k-morph` now exists: full 64,000 vocab, same corpus mix as
`bn-bpe-64k`/`bmbt-64k` (1,500,000 weighted lines). On the training run's own
2,000-text held-out sample:

| | Value |
|---|--:|
| Legacy fragmentation (`conjunct_fragmentation_rate`) | 3.399% |
| Destructive rate | 0.004% (5 of 121,136 splittable clusters) |
| Any-split rate | 8.620% |
| Fertility | 1.704 |

The legacy-fragmentation number **cross-validates the advance prediction
above almost exactly** (3.349%, measured by running the morphology layer
directly on held-out text before any artifact existed) - a real trained BPE
model, independently, landed within 0.05 points of a prediction made before
it was trained. Destructive rate confirms the design intent directly: onset/
rime splits (10,434) outnumber destructive ones (5) by roughly 2000:1 - the
layer is doing what it was built to do, not incidentally damaging conjuncts
it happens to also split.

Not yet run through the standard 4-register `compare.py` benchmark (only
this training-time sample so far) - a natural next measurement. Not yet
published to Hugging Face.
