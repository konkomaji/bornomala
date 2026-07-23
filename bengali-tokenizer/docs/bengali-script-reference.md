# Bengali script reference: every letter, why it matters to `bntok`

This document exists so that every design decision in `bntok` can be traced back
to a concrete fact about the Bengali script, not a guess. Every codepoint fact
below was checked programmatically against Python's own Unicode character
database (`unicodedata`, Unicode Character Database version 13.0.0), not
recalled from memory or scraped from a secondary source — commands are included
so anyone can re-verify. Where something surprised us during development, that
is written up honestly, including the two mistakes we made and fixed while
writing this document (see "Two things we got wrong" at the end).

## 1. The Bengali Unicode block, completely enumerated

Bengali occupies `U+0980`–`U+09FF`, 128 code points, of which 96 are assigned
(verified by walking the full range with `unicodedata.name()`):

```python
import unicodedata
for cp in range(0x0980, 0x0A00):
    ch = chr(cp)
    try:
        print(f"U+{cp:04X}", unicodedata.name(ch))
    except ValueError:
        pass  # unassigned
```

### 1.1 Independent vowels (স্বরবর্ণ) — used at the start of a word or syllable

| Codepoint | Glyph | Name |
|---|---|---|
| U+0985 | অ | A |
| U+0986 | আ | AA |
| U+0987 | ই | I |
| U+0988 | ঈ | II |
| U+0989 | উ | U |
| U+098A | ঊ | UU |
| U+098B | ঋ | VOCALIC R |
| U+098C | ঌ | VOCALIC L |
| U+098F | এ | E |
| U+0990 | ঐ | AI |
| U+0993 | ও | O |
| U+0994 | ঔ | AU |

Two ranges (`U+098D`–`U+098E`, `U+0991`–`U+0992`) are unassigned gaps — Bengali
does not fill every slot Devanagari-derived scripts leave room for, because it
merged some Sanskrit vowel distinctions.

### 1.2 Dependent vowel signs / matras (স্বরচিহ্ন, কার) — attach to a consonant

| Codepoint | Glyph | Name |
|---|---|---|
| U+09BE | া | AA |
| U+09BF | ি | I |
| U+09C0 | ী | II |
| U+09C1 | ু | U |
| U+09C2 | ূ | UU |
| U+09C3 | ৃ | VOCALIC R |
| U+09C4 | ৄ | VOCALIC RR |
| U+09C7 | ে | E |
| U+09C8 | ৈ | AI |
| U+09CB | ো | O |
| U+09CC | ৌ | AU |
| U+09D7 | ৗ | AU LENGTH MARK (used with U+09C7 to spell some AU forms) |

The `ি` (I) and `ে`/`ৈ` (E/AI) signs are the well-known "left matras": logically
they follow the consonant they attach to, but they render to its **left**. This
is a rendering/reordering fact, not an encoding fact — the codepoint order in
memory is always consonant-then-vowel-sign, and `bntok` never has to reason
about visual position because `graphemes.py` segments UAX #29 clusters, which
are defined on logical order.

`U+09CB` (O) and `U+09CC` (AU) each have a canonical decomposition (to E + AA,
and E + AU-length-mark) and, unlike the three letters in §3, standard NFC
recomposes them correctly — verified:

```python
>>> import unicodedata
>>> unicodedata.normalize("NFC", "ো") == "ো"   # E + AA -> O
True
```

### 1.3 Consonants (ব্যঞ্জনবর্ণ)

The main block, `U+0995`–`U+09B9`, in the traditional varga (place-of-articulation)
order — velar, palatal, retroflex, dental, labial, then the semivowels and
sibilants:

ক খ গ ঘ ঙ · চ ছ জ ঝ ঞ · ট ঠ ড ঢ ণ · ত থ দ ধ ন · প ফ ব ভ ম · য র ল · শ ষ স হ

Three gaps inside this range are unassigned (`U+09A9`, `U+09B1`, `U+09B3`–`U+09B5`) —
these are consonant slots ISCII/Devanagari's layout reserves that Bengali's
36-consonant inventory does not need.

Three more consonants live outside the main run, each with a **nukta**
(diacritic dot, U+09BC) fused onto a retroflex letter to represent a flapped
sound Bengali innovated beyond the shared Indic base: `ড়` (RRA, U+09DC), `ঢ়`
(RHA, U+09DD), `য়` (YYA, U+09DF). These three deserve their own section — see
§3, because their encoding has a genuine, easy-to-miss subtlety.

A fourth outlier, `ৎ` (KHANDA TA, U+09CE), is a special "bare" form of ত (TA)
used word-finally without an inherent vowel or explicit virama — spelled with
its own dedicated codepoint rather than TA + VIRAMA, because Bengali treats it
as a distinct, non-conjunct-forming letter shape. `normalize.py` also
canonicalises one legacy spelling of it (ত + ্ + ZWJ, sometimes produced by
older input methods) to this dedicated codepoint — see `_LEGACY_KHANDA_TA`.

Bengali/Assamese also share this block for two more consonant-like letters used
mainly in Assamese: `ৰ` (RA WITH MIDDLE DIAGONAL, U+09F0) and `ৱ` (RA WITH LOWER
DIAGONAL / WA, U+09F1). `bntok`'s guaranteed-coverage set spans the whole
Bengali block, so Assamese text round-trips too, even though the project is
Bengali-first.

### 1.4 Signs

| Codepoint | Glyph | Name | Use |
|---|---|---|---|
| U+0981 | ঁ | CANDRABINDU | nasalisation of a vowel |
| U+0982 | ং | ANUSVARA | nasal consonant sound (usually word-final/pre-consonant) |
| U+0983 | ঃ | VISARGA | rare aspirated breath sound, mostly in Sanskrit loanwords |
| U+09BC | ় | NUKTA | modifies a base consonant (forms ড়/ঢ়/য়, see §3) |
| U+09BD | ঽ | AVAGRAHA | marks vowel elision in Sanskritic verse, rare |
| U+09CD | ্ | VIRAMA (হসন্ত) | suppresses the inherent vowel; the conjunct-forming glue, see §2 |
| U+09FE | ৾ | SANDHI MARK | Unicode 11.0 addition for Sanskrit-style elision marking |
| U+09FC | ৼ | VEDIC ANUSVARA | Unicode addition for Vedic/philological Bengali texts, very rare |
| U+09FD | ৽ | ABBREVIATION SIGN | marks a contraction, e.g. in dates/titles |

### 1.5 Digits and historical currency

`U+09E6`–`U+09EF` are the Bengali digits ০১২৩৪৫৬৭৮৯. `bntok` treats them as
ordinary Bengali-block codepoints — they always get an atom.

`U+09F2`–`U+09FB` are a full pre-decimal Bengali currency notation: rupee
marks, and a set of "currency numerator/denominator" glyphs (`৴ ৵ ৶ ৷ ৸ ৹`) used
to typeset annas, paisa, and ganda fractions of a rupee before India's 1957
decimalisation, plus `৺` (ISSHAR, an honorific mark once placed before deities'
or deceased elders' names) and `৻` (GANDA MARK). These look exotic, but they are
not a museum curiosity for this project: the pre-1950 digitised literature in
our induction corpus (Sangraha's `pdf`-typed documents, see
`docs/known-issues.md` point 6) genuinely contains some of these, since old
Bengali books quoted prices and honorifics this way. One of them, `৷`
(U+09F7, CURRENCY NUMERATOR FOUR), has a second life: because it visually
resembles a single vertical stroke, it was widely reused on typewriters and in
some fonts as a stand-in for the real sentence-ending danda when that
character wasn't available — `corpus.py`'s sentence splitter accounts for this
(see §4).

## 2. Grapheme clusters: what a Bengali reader actually sees as "one letter"

A conjunct (যুক্তাক্ষর) forms when a virama (্, U+09CD) glues two or more
consonants together with no vowel between them — logically "consonant + virama +
consonant [+ virama + consonant ...]", optionally followed by a vowel sign. What
renders is not a mechanical stacking: Bengali has genuine ligature glyphs for
common conjuncts (ক্ষ "kṣa", জ্ঞ "jña" — historically read almost as their own
letters), and three special reordering/attachment rules:

* **Reph**: RA + VIRAMA at the *start* of a cluster (র্ক, "rka") renders as a
  hook above the following consonant, not as RA in its normal position.
  `graphemes.py: has_reph`.
* **Ra-phala**: VIRAMA + RA where RA is *not* the first consonant (ক্র, "kra")
  renders as a diagonal stroke below-right of the base. `graphemes.py:
  has_ra_phala`.
* **Ya-phala**: VIRAMA + YA (ক্য, "kya") renders as a diagonal stroke to the
  right of the base, and also marks vowel length in some conjuncts.
  `graphemes.py: has_ya_phala`.

A full cluster can therefore run from 1 codepoint (a bare vowel or consonant, no
vowel sign, e.g. simple "ক") to around 8 (a multi-consonant conjunct plus a
vowel sign plus a sign, e.g. "ক্ষ্ণ্যৈ"-style extremes in Sanskritic text) while
still being one visually and cognitively atomic unit. This is exactly why
`bntok` never operates on codepoints: `graphemes.py` uses UAX #29 extended
grapheme clustering (`regex`'s `\X`) to get this right without hand-coding every
reordering rule, and `atoms.py` maps each whole cluster to one Private-Use-Area
symbol before any subword merge ever runs, so a merge boundary structurally
cannot land inside a cluster (`docs/architecture.md`).

## 3. Two valid spellings of the same letter: RRA, RHA, YYA

`ড়` (RRA), `ঢ়` (RHA), and `য়` (YYA) can each be written two ways:

1. **The precomposed singleton** — one dedicated codepoint: U+09DC, U+09DD,
   U+09DF respectively.
2. **The decomposed sequence** — the visually-related base consonant (ড, ঢ, য)
   followed by U+09BC NUKTA.

Both are common in real text; which one a given source uses depends on the
input method, font, or (for the OCR'd literary material in our corpus) the
digitisation tool. Unicode does register a canonical decomposition from each
singleton to its base+nukta pair — but permanently excludes all three from
NFC's re-composition step (this is Unicode's frozen "composition exclusion"
policy, not a bug in this codebase or in Python). The practical, verified
consequence:

```python
>>> import unicodedata
>>> s = chr(0x09DC)                      # the singleton ড়
>>> d = chr(0x09A1) + chr(0x09BC)        # DDA + NUKTA, decomposed
>>> unicodedata.normalize("NFC", s) == unicodedata.normalize("NFC", d)
True                                      # NFC DOES unify them...
>>> unicodedata.normalize("NFC", s)
'ড়'                            # ...but onto the DECOMPOSED form
```

So there is no cross-source encoding-inconsistency bug here — plain NFC already
makes both spellings converge to the same output. `normalize.py` adds one more
step purely for efficiency: it re-composes these three back to their single
dedicated codepoint, so the tokenizer spends one atom instead of two on three
letters that are common in everyday Bengali (বড়, "big"; পড়া, "reading"; গড়ে,
"on average"). Skipping this step would still be correct, just marginally less
efficient — see the "Two things we got wrong" note below for how this was
originally mis-described.

## 4. Bengali's sentence punctuation is not in the Bengali block at all

The দাঁড়ি (danda, sentence-final punctuation, "।") and দ্বিদাঁড়ি (double danda,
verse/paragraph-final, "॥") are `U+0964` and `U+0965` — codepoints in the
**Devanagari** block, reused unchanged across nearly every Brahmic script
including Bengali, rather than re-encoded per script. `bntok`'s guaranteed
round-trip set is built from the Bengali block plus ASCII; danda and double
danda were originally *not* in that set (`graphemes.py: GUARANTEED_CODEPOINTS`),
meaning the single most common punctuation mark in real Bengali text only got
an atom by corpus frequency, not by structural guarantee. In practice this
never caused a visible failure (danda appears far too often in any real corpus
to miss the frequency threshold), but it was a real gap in what the codebase
claimed versus what it did, and it is now fixed: `graphemes.py` explicitly adds
`SHARED_INDIC_PUNCTUATION = {DANDA, DOUBLE_DANDA}` to the guarantee.
`corpus.py`'s sentence splitter (`_split_lines`) also already accounted for the
informal `৷` (U+09F7, the currency-numerator character reused as a danda
substitute, §1.5) alongside the real danda and double danda.

## 5. Where this leaves the tokenizer's honesty ledger

Two things we got wrong while writing this document, corrected in the same
session, both worth stating plainly rather than quietly fixing:

1. We first assumed the RRA/RHA/YYA composition-exclusion issue meant our
   corpus sources could be *inconsistently encoded against each other*,
   fragmenting vocabulary. That was wrong: plain NFC already unifies both
   spellings (onto the decomposed form). The fix we shipped (re-composing to
   the singleton) is a real, small efficiency improvement, not a correctness
   fix for a bug that never existed — see §3.
2. Our first attempt at that fix was itself a no-op: typing the Bengali
   literal characters directly into the source file let an intermediate
   tool/editor layer silently re-decompose them before they were saved, so the
   dictionary's "singleton" values were actually still decomposed sequences.
   The fix now builds `_NFC_EXCLUSIONS` from explicit `chr(0x09DC)`-style
   codepoints instead of literal characters, specifically to make this class
   of mistake impossible to reintroduce (`normalize.py`).

Everything else in this document — the full block inventory, the danda gap, the
grapheme-cluster and reph/phala rules — was cross-checked against
`unicodedata` or the existing, tested code in `graphemes.py` before being
written down, in keeping with this project's rule that no number or claim is
stated without the data behind it (`docs/known-issues.md`).
