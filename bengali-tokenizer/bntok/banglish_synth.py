r"""
Synthetic Banglish (romanized Bengali) training-pair generator.

Part of the Banglish work: our tokenizer (v1 and BMBT, tied) measures dead
last, 17th of 17, on real romanized Bengali held-out text
(scripts/compare.py --register banglish; see docs/known-issues.md). The
planned fix does not compete with BanglaBERT/BanglaT5/IndicBERTv2 on raw
Latin-script BPE - it transliterates Banglish to real Bengali script first,
then hands the result to the tokenizer that already wins by a wide margin on
Bengali script. This module generates the (noisy_latin, canonical_bengali)
training pairs that transliteration model needs, from scratch: no pretrained
model, no borrowed transliteration output.

Two honest, separate sources feed the training set, and this module produces
only the first:

1. Spelling-variant synthetic pairs (this module): real Bengali text, already
   in this repo's own training corpus, rendered into Latin script by a
   reverse phonetic table with randomized variant selection. This captures
   the DOMINANT real source of Banglish variance - the same Bengali word
   spelled several different ways in Latin ("achi"/"asi"/"achhi") - at
   whatever scale the source corpus allows, cheaply, because it is pure
   substitution over BMBT's own real featural decomposition
   (bntok.bmbt.featurize), not a new parser.
2. Real noisy examples (NOT built here): mined directly from
   scripts/compare.py's own `banglish` held-out register (real CC-100
   bn_rom text, genuine human typing noise including digit-substitution
   slang like "kor6o"). This module's reverse table does NOT invent
   slang/leetspeak rules - digit-for-sound substitution patterns are not
   something we can verify as systematic without real examples, so
   inventing rules for them here would be exactly the kind of fabricated
   pattern this project's transparency standard exists to prevent. Real
   noisy pairs come from real text, separately, not from this generator.

The reverse phonetic table (BENGALI_TO_LATIN below) follows the same
spelling conventions the Avro Phonetic input method popularised for Bengali
typing (SUST OmicronLab; the de facto standard casual romanization most
Bengali internet users already learned by habit, not one this project
invented). It is a heuristic approximation, not an authoritative linguistic
mapping: many Bengali words have more valid casual spellings than any fixed
table can enumerate, and the inherent vowel (the implicit "a"/"o" sound a
bare consonant carries with no matra) is genuinely ambiguous - sometimes
written, usually dropped, and its choice is context-dependent in ways this
table does not model. This is stated plainly, the same honesty standard as
`is_clean_bengali_line`'s and `is_clean_banglish_line`'s own docstrings:
this reduces the problem, it does not solve it perfectly, and "perfect" is
not a real target for any informal-text transliteration system.
"""

from __future__ import annotations

import random

from .akshara import Akshara
from .bmbt import AksharaFeatures, featurize
from .errors import NormalizationError

# --- reverse phonetic table -------------------------------------------------
# Ordered by variant list: first entry is the most common casual spelling,
# later entries are real alternates seen in practice. Random selection is
# weighted toward earlier entries (see _pick), not uniform, so the generated
# corpus is dominated by common spellings the way real usage is, with real
# variance in the tail rather than every spelling equally likely.

_CONSONANT_LATIN: dict[str, list[str]] = {
    "ক": ["k"], "খ": ["kh"], "গ": ["g"], "ঘ": ["gh"], "ঙ": ["ng"],
    "চ": ["ch", "c"], "ছ": ["chh", "ch"], "জ": ["j"], "ঝ": ["jh"], "ঞ": ["ng", "n"],
    # Real casual typing barely distinguishes retroflex ট/ড from dental ত/দ
    # (checked against Dakshina: 0/15 sampled real spellings used the
    # uppercase Avro convention) - lowercase dominant, uppercase kept only
    # as a rare tail variant rather than removed outright.
    "ট": ["t", "T"], "ঠ": ["th"], "ড": ["d", "D"], "ঢ": ["dh"], "ণ": ["n"],
    "ত": ["t"], "থ": ["th"], "দ": ["d"], "ধ": ["dh"], "ন": ["n"],
    "প": ["p"], "ফ": ["ph", "f"],
    # ব is handled specially in render_akshara_latin, not read from this
    # table directly, when it is a chained (non-initial) conjunct
    # consonant: as the second member of স্ব/শ্ব/দ্ব-style conjuncts it is
    # phonetically a labial glide, real spellings render it "w", not "b"
    # (checked against Dakshina dev split: "swamijir"/"shamijir", not
    # "sbamijir"; "biswash"/"bishwash", not "bisbaas"). Word-initial/
    # standalone ব keeps its own sound, "b", from this table.
    "ব": ["b"], "ভ": ["bh", "v"], "ম": ["m"],
    # য is handled specially in render_akshara_latin, not read from this
    # table directly: word-initial/onset-first it is "j" (its own sound),
    # but as a chained conjunct consonant (ya-phala, e.g. ত্য) it is
    # phonetically a glide and real spellings render it "y", not "j"/"z"
    # (checked against Dakshina: "protyahar", not "protjahar"). See
    # render_akshara_latin.
    "য": ["j", "z"],
    "র": ["r"], "ল": ["l"],
    "শ": ["sh", "s"], "ষ": ["sh", "s"], "স": ["s"], "হ": ["h"],
    "ড়": ["r", "rh"], "ঢ়": ["rh"], "য়": ["y"],
    "ৎ": ["t"],
    chr(0x09F0): ["r"], chr(0x09F1): ["r"],  # rare Assamese-shared RA variants
}
_YA_GLIDE_LATIN = ["y", "j"]  # য as a non-initial (ya-phala) onset consonant
_BA_GLIDE_LATIN = ["w", "b"]  # ব as a non-initial (conjunct-chained) onset consonant

_INDEPENDENT_VOWEL_LATIN: dict[str, list[str]] = {
    "অ": ["o", "a"], "আ": ["a", "aa"], "ই": ["i"], "ঈ": ["i", "ee"],
    "উ": ["u"], "ঊ": ["u", "oo"], "ঋ": ["ri"],
    "এ": ["e"], "ঐ": ["oi", "oy"], "ও": ["o"], "ঔ": ["ou", "ow"],
    chr(0x09E0): ["ri"], chr(0x09E1): ["li"],  # obsolete vocalic RR/LL
}

_MATRA_LATIN: dict[str, list[str]] = {
    "া": ["a", "aa"], "ি": ["i"], "ী": ["i", "ee", "ii"],
    "ু": ["u"], "ূ": ["u", "oo"], "ৃ": ["ri"],
    "ে": ["e"], "ৈ": ["oi", "oy"], "ো": ["o"], "ৌ": ["ou", "ow"],
    "ৗ": ["u"],  # AU length mark, rare standalone; coarse fallback
    chr(0x09E2): ["ri"], chr(0x09E3): ["li"],
}

_MODIFIER_LATIN: dict[str, list[str]] = {
    "ঁ": ["n", ""],   # chandrabindu: often written as trailing n, or dropped
    "ং": ["ng"],       # anusvara
    "ঃ": ["h"],        # visarga
    chr(0x09FE): [""],  # sandhi mark: no real Latin rendering
}

# Consonant-only akshara, no explicit matra: the inherent vowel. Genuinely
# ambiguous (see module docstring), but checked against Dakshina's real
# lexicon rather than assumed, in two directions:
# - Word-medial: the "o" spelling dominates real usage ("shomorthoner",
#   "purokoushal") - dropping the vowel entirely, this table's first cut,
#   was the single biggest source of mismatch (a validation run against
#   3000 sampled Dakshina words caught this; see docs/known-issues.md).
# - Word-final: the OPPOSITE preference holds - real spellings usually drop
#   it ("protyahar" not "protyaharo", "eider" not "eidero"), matching how
#   the inherent vowel is typically silent word-finally in modern spoken
#   Bengali. render_word_latin passes which case applies per akshara.
_INHERENT_VOWEL_MEDIAL = ["o", "a", ""]
_INHERENT_VOWEL_FINAL = ["", "o", "a"]

_ALL_TABLES = (_CONSONANT_LATIN, _INDEPENDENT_VOWEL_LATIN, _MATRA_LATIN, _MODIFIER_LATIN)


def _pick(rng: random.Random, variants: list[str]) -> str:
    """Weighted pick favouring earlier (more common) variants.

    Weight halves with each position (1, 1/2, 1/4, ...), so a 2-variant list
    picks the first about 2/3 of the time, matching the observation that the
    canonical Avro spelling dominates real usage but is not the only one
    seen. Not derived from any measured frequency table - a documented,
    reasonable default, not a claimed-precise distribution.
    """
    if len(variants) == 1:
        return variants[0]
    weights = [1.0 / (2 ** i) for i in range(len(variants))]
    return rng.choices(variants, weights=weights, k=1)[0]


def render_akshara_latin(feat: AksharaFeatures, rng: random.Random, is_word_final: bool = False) -> str:
    """Render one BMBT-featurized akshara as one randomly-chosen Latin spelling.

    Unmapped codepoints (should not occur for any well-formed "akshara"-kind
    chunk, since featurize_akshara only ever places CONSONANTS/VOWELS/MATRAS/
    MODIFIERS members into onset/vowel/modifiers) fall back to the empty
    string rather than raising, so a single unexpected codepoint degrades one
    syllable's spelling instead of failing the whole line - acceptable for
    bulk synthetic-data generation, where a review pass on the output corpus
    is expected, unlike the tokenizer's own strict round-trip guarantee.
    """
    parts: list[str] = []
    for i, c in enumerate(feat.onset):
        if c == "য" and i > 0:
            # ya-phala: a chained (non-initial) onset consonant is a glide,
            # not the word-initial "j" sound. See _YA_GLIDE_LATIN.
            parts.append(_pick(rng, _YA_GLIDE_LATIN))
        elif c == "ব" and i > 0:
            # ba-phala: same pattern as ya-phala above, a chained onset
            # consonant is a labial glide ("w"), not the standalone "b"
            # sound. See _BA_GLIDE_LATIN.
            parts.append(_pick(rng, _BA_GLIDE_LATIN))
        else:
            parts.append(_pick(rng, _CONSONANT_LATIN.get(c, [""])))
    if feat.vowel is not None:
        table = _INDEPENDENT_VOWEL_LATIN if feat.vowel in _INDEPENDENT_VOWEL_LATIN else _MATRA_LATIN
        parts.append(_pick(rng, table.get(feat.vowel, [""])))
    elif feat.onset:
        table = _INHERENT_VOWEL_FINAL if is_word_final else _INHERENT_VOWEL_MEDIAL
        parts.append(_pick(rng, table))
    for m in feat.modifiers:
        parts.append(_pick(rng, _MODIFIER_LATIN.get(m, [""])))
    return "".join(parts)


def render_word_latin(text: str, rng: random.Random) -> str:
    """Render one normalized Bengali string as one randomly-sampled Latin
    (Banglish) spelling, akshara by akshara, via bntok.bmbt.featurize.

    Non-akshara chunks (ASCII, punctuation, whitespace, foreign scripts -
    featurize() returns these as plain Akshara objects, unchanged) pass
    through as-is: real Banglish text is itself code-mixed with real
    English/digits/punctuation, so leaving those chunks alone rather than
    inventing a rendering for them matches real usage.
    """
    chunks = featurize(text)
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, Akshara):
            parts.append(chunk.text)
        else:
            # Word-final: the last akshara before a following "other" chunk
            # that is whitespace/punctuation (word boundary), or before the
            # end of the text entirely. A following "other" chunk that is
            # itself alphanumeric (code-mixed text glued without a
            # separator) does not count as a word boundary.
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            is_final = nxt is None or (isinstance(nxt, Akshara) and not nxt.text.isalnum())
            parts.append(render_akshara_latin(chunk, rng, is_word_final=is_final))
    return "".join(parts)


def generate_pairs(lines: list[str], variants_per_line: int = 3,
                    seed: int = 0) -> list[tuple[str, str]]:
    """Generate (noisy_latin, canonical_bengali) pairs from real Bengali lines.

    `variants_per_line` independent Latin renderings are sampled per input
    line (different random spelling choices each time), so N input lines
    produce up to N * variants_per_line pairs, all sharing the same
    canonical Bengali target - the model sees genuine one-to-many spelling
    variance for the same underlying text, not one fixed mapping.

    Raises:
      NormalizationError: propagated from featurize() if a line is not a str.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for line in lines:
        if not isinstance(line, str):
            raise NormalizationError(f"expected str, got {type(line).__name__}")
        seen: set[str] = set()
        for _ in range(variants_per_line):
            latin = render_word_latin(line, rng)
            if latin and latin not in seen:
                seen.add(latin)
                out.append((latin, line))
    return out
