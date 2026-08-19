r"""
BMBT: Bornomala's Bengali Tokenizer (v2 design step 5, partial).

Where the shipped v1 tokenizer (tokenizer.py, `BengaliTokenizer`) discovers
Bengali's structure statistically (BPE merges over UAX #29 grapheme-cluster
atoms), BMBT parses it first from Bengali's own generative grammar and
compresses on top of that:

    text
      -> normalize (NFC + ZWJ/ZWNJ policy)            normalize.py
      -> akshara parse (the finite-state virama grammar) akshara.py
      -> atoms (one symbol per akshara/other chunk, reversible)  AksharaAtomMap
      -> subword model (BPE or Unigram) over atoms     Hugging Face tokenizers
      -> token ids

This is the same "atoms, then compress" architecture as v1, with one change:
the atomic unit is the akshara `aksharas()` finds (a whole conjunct chain,
parsed by grammar), not a UAX #29 grapheme cluster (found by `regex`'s
generic Unicode algorithm). Since akshara boundaries are already nearly
identical to grapheme-cluster boundaries on well-formed Bengali text (the
v2 design's own step-4 measurement), this is NOT expected to dramatically
change fertility, and FORMAL_SPEC.md's own OPTIMAL section proves a
constrained BPE (never split an akshara) cannot beat an unconstrained one on
raw token count. Whatever the measured comparison against `bn-bpe-64k`
actually shows is reported in benchmarks/bengali-comparison.md, honestly,
whichever way it goes.

What BMBT adds regardless of the fertility outcome is `featurize()`: for
each akshara, its actual structural decomposition (onset consonants, which
carry a nukta, the vowel, trailing modifiers, whether a ZWJ/ZWNJ occurred) -
a real, tested, usable output of the tokenizer itself, not an
embedding-layer afterthought.

Morphology (v2 design step 5's other half) is now built, opt-in via
`morphology=True`. `morphology.py` finds the suffix chain, and training
inserts a merge barrier at each morpheme seam so BPE cannot learn a token
spanning one. Two things about it are stated up front rather than discovered
in the results:

  * it is expected to make fertility slightly WORSE, because FORMAL_SPEC.md's
    OPTIMAL proof applies to any additional constraint, and morpheme
    boundaries are additional constraints. What it buys is boundaries that
    fall where Bengali's morphemes fall, which fertility cannot see;
  * it reaches 100% of Bengali morpheme boundaries, but only after separating
    two guarantees this module had been treating as one. CONJUNCT INTEGRITY
    (never sever a virama-joined consonant cluster) is the guarantee that
    matters and stays absolute. AKSHARA ATOMICITY (never part a consonant
    cluster from its matra) was an implementation choice strictly stronger
    than that, and it was costing 30.4% of all morpheme boundaries, because a
    matra binds orthographically to the consonant before it while belonging
    morphologically to the suffix after it.

    Under akshara atomicity only 62.4% of seams were placeable. Refusing
    analyses that would cut inside a conjunct (`morphology.cuts_inside_conjunct`)
    removed the rest as false positives rather than as unreachable seams, and
    factoring an akshara at the onset/rime seam where a morpheme boundary falls
    there covers the remainder: 100% reachable, zero conjunct-integrity
    violations, verified on 1.47M codepoints of held-out text.

    The cost is real and reported, not hidden: a morphology-enabled artifact
    splits 3.349% of grapheme clusters by design, all at morpheme seams, while
    its conjunct fragmentation stays exactly zero. `evaluate.py` reports both
    numbers separately. Full measurement: `docs/bmbt-morphology.md`.

This module is deliberately self-contained: it imports nothing from
atoms.py or tokenizer.py (v1's implementation), only from errors.py,
normalize.py, akshara.py, substrate.py, and the `tokenizers` library. A
future change to either file can never silently change BMBT's behaviour,
and vice versa - v1 (`bn-bpe-64k`) is unaffected by anything here.

The artifact is portable the same way v1's is: a directory with
tokenizer.json (the atom-space subword model), atoms.json (the
akshara-atom map), and config.json (algorithm, vocab size, normalisation
policy, and `"format": "bornomala-bmbt/1"` so a loader can tell this apart
from a v1 `"bornomala-track-a/1"` directory without guessing from the
directory name).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .akshara import Akshara, akshara_bounds, akshara_bounds_batch, aksharas
from .errors import (
    BnTokError,
    ConfigError,
    DecodeError,
    EmptyCorpusError,
    EncodeError,
    LoadError,
    TrainingError,
    VocabSizeError,
    require,
)
from .morphology import morph_bounds
from .normalize import normalize
from .substrate import (
    CONSONANTS,
    GUARANTEED_CODEPOINTS,
    MATRAS,
    MODIFIERS,
    NUKTA,
    VIRAMA,
    VOWELS,
    ZWJ,
    ZWNJ,
)

UNK_TOKEN = "<unk>"
_SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

# Lines segmented per vectorized call during atom-map training. Large enough
# that numpy's per-call setup is amortised, small enough that a streamed
# corpus is never fully materialised in memory.
_TRAIN_BATCH_LINES = 4096


class FeaturizeError(BnTokError):
    """`featurize_akshara` was called on a chunk with no akshara-grammar structure."""


# --- the akshara atom map ---------------------------------------------------

# Private Use Area planes 15 and 16, same ranges atoms.py uses for v1 - huge,
# no realistic risk of exhaustion from one artifact, let alone two. A
# distinct UNK atom (one codepoint below v1's) keeps the two artifacts'
# atom spaces provably non-colliding by construction, not just "probably
# fine because they're separate files": this module never imports v1's
# atoms.py, so there is no shared state to collide, but the distinct
# constant costs nothing and removes the question entirely.
_PUA_RANGES = [(0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
_UNK_ATOM = chr(0xF8FE)

# Morpheme-boundary marker. Never a token and never part of the vocabulary: the
# pre-tokenizer splits on it and discards it, so it exists only to stop BPE
# merging across a morpheme seam. Because it is removed before training and
# before encoding, decode is unaffected and round-trip is unchanged.
_MORPH_MARKER = chr(0xF8FD)


def _morph_seams(nfc_text: str) -> frozenset[int]:
    """Morpheme seam offsets in `nfc_text`, word by word.

    Shared by `AksharaAtomMap.build` and `AksharaAtomMap._morph_barriers` so the
    atom inventory is built over exactly the pieces encoding will later produce.
    """
    seams: set[int] = set()
    base = 0
    for word in nfc_text.split(" "):
        if word:
            for cut in morph_bounds(word):
                seams.add(base + cut)
        base += len(word) + 1
    return frozenset(seams)


def _pua_generator():
    for lo, hi in _PUA_RANGES:
        for cp in range(lo, hi + 1):
            yield chr(cp)


class AksharaAtomMap:
    """A reversible map between akshara/other chunk text and single atom codepoints.

    Structurally parallel to `atoms.AtomMap` (same two-tier coverage scheme,
    same PUA-based atom assignment, same JSON persistence shape), but built
    over `aksharas()` chunks instead of `grapheme_clusters()`. "Akshara" and
    "other" kind chunks are treated identically here: `aksharas()` already
    guarantees every "other" chunk is exactly one grapheme cluster (never a
    partial one, never a run of several), so there is nothing kind-specific
    an atom map needs to do - `.kind` only matters later, in `featurize()`.

    The JSON key is still named `cluster_to_atom` for mechanical
    copy-paste-verifiability against atoms.py's format; here "cluster"
    means "akshara-or-grapheme chunk text", not a UAX #29 cluster
    specifically.
    """

    def __init__(self, cluster_to_atom: dict[str, str], min_freq: int = 1,
                 morphology: bool = False):
        self.cluster_to_atom = cluster_to_atom
        self.morphology = morphology
        self.atom_to_cluster = {a: c for c, a in cluster_to_atom.items()}
        self.min_freq = min_freq
        self.atom_to_cluster[_UNK_ATOM] = ""  # UNK decodes to nothing (lossy, rare)

    # ---- build ----
    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 2,
        guarantee: Iterable[str] | None = None,
        morphology: bool = False,
    ) -> AksharaAtomMap:
        """Build an akshara atom map from a corpus of NFC-normalised strings.

        Every codepoint in `guarantee` is given an atom even if absent from
        the corpus, so round-trip is guaranteed for that character set
        (typically the whole Bengali block plus ASCII). Non-string lines
        are skipped defensively. Raises EmptyCorpusError if no usable text
        is found, and ConfigError if the atom budget is exhausted.
        """
        if min_freq < 1:
            raise ConfigError(f"min_freq must be >= 1, got {min_freq}")

        chunk_freq: Counter[str] = Counter()
        codepoints: set[str] = set()
        seen_any = False

        def _absorb(batch: list[str]) -> bool:
            """Count every chunk in `batch`. Returns whether anything counted."""
            any_text = False
            for line, bounds in zip(batch, akshara_bounds_batch(batch)):
                if morphology:
                    seams = _morph_seams(line)
                    if seams:
                        bounds = sorted(set(bounds) | seams)
                start = 0
                for end in bounds:
                    text = line[start:end]
                    start = end
                    if text.isspace():
                        continue  # whitespace stays literal (carries word boundaries)
                    any_text = True
                    chunk_freq[text] += 1
                    if len(text) > 1:
                        codepoints.update(text)
                    else:
                        codepoints.add(text)
            return any_text

        # Buffered rather than line-at-a-time: the vectorized backend's array
        # setup dominates on single short lines, so segmenting a block at once
        # is what actually makes it faster than the scalar scan. `corpus` may
        # be a one-shot stream, so it is consumed exactly once.
        batch: list[str] = []
        for line in corpus:
            if not isinstance(line, str) or not line:
                continue
            batch.append(line)
            if len(batch) >= _TRAIN_BATCH_LINES:
                seen_any = _absorb(batch) or seen_any
                batch.clear()
        if batch:
            seen_any = _absorb(batch) or seen_any

        if not seen_any:
            raise EmptyCorpusError("akshara atom map: corpus contained no usable text")

        for cp in (guarantee or ()):
            if isinstance(cp, str) and len(cp) == 1 and not cp.isspace():
                codepoints.add(cp)

        gen = _pua_generator()
        cluster_to_atom: dict[str, str] = {}

        def _next_atom() -> str:
            try:
                return next(gen)
            except StopIteration:  # pragma: no cover - needs >131k distinct atoms
                raise ConfigError(
                    "akshara atom budget exhausted (>131000 distinct chunks); "
                    "raise min_freq to reduce the atom set"
                )

        # 1) every individual codepoint gets an atom (guarantees the fallback).
        for cp in sorted(codepoints):
            cluster_to_atom[cp] = _next_atom()
        # 2) frequent multi-codepoint chunks get their own atom.
        for text, freq in chunk_freq.most_common():
            if len(text) > 1 and freq >= min_freq and text not in cluster_to_atom:
                cluster_to_atom[text] = _next_atom()

        return cls(cluster_to_atom, min_freq=min_freq, morphology=morphology)

    # ---- encode / decode ----
    def encode(self, nfc_text: str) -> str:
        """Map NFC text to a string of atoms (one atom per known chunk,
        else one atom per constituent codepoint, else UNK)."""
        out = []
        # `akshara_bounds` rather than `aksharas`: this loop only ever needs
        # each chunk's surface text, never its kind or offsets, so there is
        # no reason to pay for an Akshara object per chunk. Same scan, same
        # boundaries - see akshara.py's `_scan`.
        get = self.cluster_to_atom.get
        barriers = self._morph_barriers(nfc_text) if self.morphology else frozenset()
        # A barrier may fall strictly INSIDE an akshara, at the seam between a
        # consonant cluster and its matra. Merging the barrier offsets into the
        # boundary list splits the akshara there, which is the whole point of
        # the factoring: see `_morph_barriers` for why that is safe and
        # `docs/bmbt-morphology.md` for what it buys.
        bounds = akshara_bounds(nfc_text)
        if barriers:
            bounds = sorted(set(bounds) | barriers)
        start = 0
        for end in bounds:
            text = nfc_text[start:end]
            if start in barriers:
                out.append(_MORPH_MARKER)
            start = end
            if text.isspace():
                out.append(text)  # keep whitespace literal for the word-boundary marker
                continue
            a = get(text)
            if a is not None:
                out.append(a)
            else:
                for cp in text:
                    out.append(get(cp, _UNK_ATOM))
        return "".join(out)

    def _morph_barriers(self, nfc_text: str) -> frozenset[int]:
        """Offsets in `nfc_text` where a token boundary is allowed to be forced.

        Every morpheme seam qualifies. Two facts make that safe, and both were
        measured rather than assumed (`docs/bmbt-morphology.md`):

          * `morphology.cuts_inside_conjunct` already refuses any analysis whose
            seam would fall inside a virama-joined cluster, so no barrier can
            ever split a conjunct. On 80,000 held-out Wikipedia words this
            removed 2,288 proposed boundaries, every one of them a false
            positive (`রাষ্ট্র`, `মাত্র` and `স্তোত্র` split before a
            stem-final `র`; `বিশ্বে` read as a future-tense `বে`);
          * of the seams that remain, the ones that do not coincide with an
            akshara boundary are now ALL of a single kind: the onset/rime seam,
            between a consonant cluster and its matra (`শ্বে` -> `শ্ব` + `ে`).

        Splitting there breaks the akshara but not the conjunct, and those are
        different operations. `শ্ব` + `ে` yields a valid consonant cluster and a
        valid vowel sign, both units Bengali literacy teaches by name. `ক্` +
        `ষ` yields a fragment corresponding to nothing. BMBT's real guarantee is
        conjunct integrity; akshara atomicity was an implementation choice
        strictly stronger than that guarantee needs, and it was costing 30.4% of
        all morpheme boundaries. Relaxing it exactly here takes morphological
        reachability from 66.5% to 100%, with conjunct fragmentation still zero.

        The consequence is stated rather than hidden: grapheme-cluster
        fragmentation is no longer zero for a morphology-enabled artifact, by
        design and only at morpheme seams. Conjunct fragmentation, the metric
        this project has always actually meant, is unchanged at zero.
        """
        return _morph_seams(nfc_text)

    def decode(self, atom_text: str) -> str:
        """Map a string of atoms back to text.

        Known atoms map to their chunk; the reserved UNK atom maps to
        nothing; any other character (spaces, newlines, the Metaspace
        marker once already converted) passes through unchanged.
        """
        out = []
        for a in atom_text:
            if a == _UNK_ATOM:
                continue
            out.append(self.atom_to_cluster.get(a, a))
        return "".join(out)

    # ---- persistence ----
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"min_freq": self.min_freq, "morphology": self.morphology,
                 "cluster_to_atom": self.cluster_to_atom},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> AksharaAtomMap:
        if not os.path.exists(path):
            raise LoadError(f"akshara atom map not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(d["cluster_to_atom"], min_freq=d.get("min_freq", 1),
                       morphology=d.get("morphology", False))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise LoadError(f"corrupt akshara atom map {path}: {e}") from e

    def __len__(self) -> int:
        return len(self.cluster_to_atom)

    @property
    def num_cluster_atoms(self) -> int:
        return sum(1 for c in self.cluster_to_atom if len(c) > 1)


# --- BMBT: train / encode / decode ------------------------------------------


class BMBT:
    """BMBT: Bornomala's Bengali Tokenizer (v2, grammar + featural + statistical).

    Mirrors `BengaliTokenizer`'s public shape (train/encode/decode/save/load/
    roundtrip_ok/content_roundtrip_ok) deliberately, so evaluate.py's
    `evaluate()` function works on a trained BMBT completely unmodified, and
    scripts/compare.py's tokenizer-measurement functions need only a
    mechanical class-name swap to work with BMBT too.
    """

    def __init__(self, tok: Tokenizer, atoms: AksharaAtomMap, config: dict):
        self._tok = tok
        self._atoms = atoms
        self.config = config

    # ---------------- training ----------------
    @classmethod
    def train(
        cls,
        corpus: list[str],
        algo: str = "bpe",
        vocab_size: int = 64000,
        min_atom_freq: int = 2,
        zwnj_policy: str = "preserve",
        morphology: bool = False,
    ) -> BMBT:
        """Induce a BMBT tokenizer from a corpus (a list of raw strings).

        Raises:
          ConfigError: invalid algo, vocab_size, or corpus type.
          EmptyCorpusError: corpus has no usable text.
          VocabSizeError: vocab_size cannot hold even the atom alphabet.
          TrainingError: the underlying subword trainer failed.
        """
        require(algo in ("bpe", "unigram"), f"algo must be 'bpe' or 'unigram', got {algo!r}")
        require(isinstance(vocab_size, int) and vocab_size >= 256,
                f"vocab_size must be an int >= 256, got {vocab_size!r}", VocabSizeError)
        require(hasattr(corpus, "__iter__") and not isinstance(corpus, (str, bytes)),
                "corpus must be an iterable of strings (not a single string)")

        norm = []
        for line in corpus:
            if not isinstance(line, str):
                continue
            s = normalize(line, zwnj_policy=zwnj_policy)
            if s.strip():
                norm.append(s)
        if not norm:
            raise EmptyCorpusError("corpus contained no usable text after normalisation")

        atoms = AksharaAtomMap.build(norm, min_freq=min_atom_freq,
                                     guarantee=GUARANTEED_CODEPOINTS, morphology=morphology)

        floor = len(atoms) + len(_SPECIALS)
        if vocab_size < floor:
            raise VocabSizeError(
                f"vocab_size={vocab_size} is below the akshara atom alphabet size "
                f"({floor}); raise vocab_size or raise min_atom_freq to shrink the alphabet"
            )

        atomised = [atoms.encode(line) for line in norm]
        initial_alphabet = list(atoms.cluster_to_atom.values())

        if algo == "bpe":
            tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size, special_tokens=_SPECIALS,
                initial_alphabet=initial_alphabet, show_progress=False,
            )
        else:
            tok = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size, special_tokens=_SPECIALS, unk_token=UNK_TOKEN,
                initial_alphabet=initial_alphabet, show_progress=False,
            )

        if morphology:
            # Split on the morpheme marker and discard it, so BPE can never
            # learn a merge that spans a morpheme seam. The marker is not in
            # the vocabulary and never becomes a token, so decode is untouched
            # and round-trip still holds exactly.
            tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Metaspace(replacement="▁"),
                pre_tokenizers.Split(_MORPH_MARKER, behavior="removed"),
            ])
        else:
            tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
        tok.decoder = decoders.Metaspace(replacement="▁")

        try:
            tok.train_from_iterator(atomised, trainer=trainer, length=len(atomised))
        except Exception as e:  # the Rust trainer raises assorted exception types
            raise TrainingError(f"{algo} training failed: {type(e).__name__}: {e}") from e

        config = {
            "algo": algo,
            "morphology": morphology,
            "vocab_size": vocab_size,
            "actual_vocab_size": tok.get_vocab_size(),
            "min_atom_freq": min_atom_freq,
            "zwnj_policy": zwnj_policy,
            "num_atoms": len(atoms),
            "num_cluster_atoms": atoms.num_cluster_atoms,
            "format": "bornomala-bmbt/1",
        }
        return cls(tok, atoms, config)

    # ---------------- encode / decode ----------------
    def _atomise(self, text: str) -> str:
        if not isinstance(text, str):
            raise EncodeError(f"expected str, got {type(text).__name__}")
        return self._atoms.encode(normalize(text, zwnj_policy=self.config["zwnj_policy"]))

    def encode(self, text: str) -> list[int]:
        try:
            return self._tok.encode(self._atomise(text)).ids
        except EncodeError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise EncodeError(f"encode failed: {type(e).__name__}: {e}") from e

    def encode_tokens(self, text: str) -> list[str]:
        """Human-readable token surfaces (each token mapped back to atoms' chunks)."""
        toks = self._tok.encode(self._atomise(text)).tokens
        out = []
        for t in toks:
            readable = self._atoms.decode(t.replace("▁", " "))
            out.append(readable if readable else t)
        return out

    def decode(self, ids: list[int]) -> str:
        if not hasattr(ids, "__iter__"):
            raise DecodeError(f"expected an iterable of ids, got {type(ids).__name__}")
        try:
            atomstr = self._tok.decode(list(ids))
            text = self._atoms.decode(atomstr)
            return text.strip(" ")
        except Exception as e:  # pragma: no cover - defensive
            raise DecodeError(f"decode failed: {type(e).__name__}: {e}") from e

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # ---------------- featural output ----------------
    def featurize(self, text: str) -> list:
        """See the module-level `featurize()`; exposed as a convenience method
        so it is usable both with and without a trained tokenizer."""
        return featurize(text)

    # ---------------- persistence ----------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._tok.save(os.path.join(directory, "tokenizer.json"))
        self._atoms.save(os.path.join(directory, "atoms.json"))
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> BMBT:
        if not os.path.isdir(directory):
            raise LoadError(f"not a tokenizer directory: {directory}")
        tpath = os.path.join(directory, "tokenizer.json")
        cpath = os.path.join(directory, "config.json")
        for p in (tpath, cpath):
            if not os.path.exists(p):
                raise LoadError(f"missing file in tokenizer directory: {p}")
        try:
            tok = Tokenizer.from_file(tpath)
            atoms = AksharaAtomMap.load(os.path.join(directory, "atoms.json"))
            with open(cpath, encoding="utf-8") as f:
                config = json.load(f)
        except LoadError:
            raise
        except Exception as e:
            raise LoadError(f"could not load tokenizer from {directory}: {e}") from e
        return cls(tok, atoms, config)

    # ---------------- self-check ----------------
    def roundtrip_ok(self, text: str) -> bool:
        norm = normalize(text, zwnj_policy=self.config["zwnj_policy"])
        return self.decode(self.encode(text)) == norm.strip(" ")

    def content_roundtrip_ok(self, text: str) -> bool:
        norm = normalize(text, zwnj_policy=self.config["zwnj_policy"])

        def strip_ws(s: str) -> str:
            return "".join(s.split())

        return strip_ws(self.decode(self.encode(text))) == strip_ws(norm)


# --- featural decomposition --------------------------------------------------


@dataclass(frozen=True)
class AksharaFeatures:
    """The structural decomposition of one akshara: onset, vowel, modifiers.

    This is BMBT's real, tested output beyond an opaque token id: for any
    "akshara"-kind chunk `aksharas()` finds, its onset consonants (in chain
    order), which of them carry a nukta, its vowel (an independent vowel or
    a dependent matra, or None for the inherent "a"), and its trailing
    modifiers - plus whether a ZWJ or ZWNJ occurred in the chunk, since
    `akshara.py`'s own grammar treats those as first-class chain-control
    signals, not incidental noise to discard.
    """

    text: str
    onset: list[str]
    nuktas: list[bool]
    vowel: str | None
    modifiers: list[str]
    has_zwj: bool
    has_zwnj: bool


def featurize_akshara(chunk: Akshara) -> AksharaFeatures:
    """Decompose one akshara chunk into its onset/vowel/modifier structure.

    A strictly smaller problem than akshara.py's own boundary-finding scan:
    the chunk's extent is already known, so this only classifies each
    codepoint within it. Reuses the same mixed-run tolerance `_scan_tail`
    had to learn empirically (nukta/matra/modifier/ZWJ/ZWNJ can appear in
    any order around a chaining virama in real text) rather than assuming a
    stricter ordering that would just re-break on the same real-world cases
    akshara.py already fixed.

    Raises:
      FeaturizeError: if `chunk.kind != "akshara"` (an "other" chunk has no
        akshara-grammar structure to extract).
    """
    if chunk.kind != "akshara":
        raise FeaturizeError(
            f"featurize_akshara: chunk.kind == {chunk.kind!r} has no akshara-grammar "
            f"structure (text={chunk.text!r})"
        )

    text = chunk.text
    n = len(text)
    pos = 0
    onset: list[str] = []
    nuktas: list[bool] = []
    vowel: str | None = None
    modifiers: list[str] = []
    has_zwj = False
    has_zwnj = False

    if text[0] in VOWELS:
        vowel = text[0]
        pos = 1
    else:
        # text[0] in CONSONANTS, guaranteed by kind == "akshara" (akshara.py's
        # own grammar: Consonant Tail | Vowel Tail).
        onset.append(text[0])
        nuktas.append(False)
        pos = 1
        while True:
            advanced = False
            # absorb a mixed run of {Nukta, ZWJ, ZWNJ} attached to the current
            # onset consonant, tracking flags without assuming a fixed order.
            while pos < n and (text[pos] == NUKTA or text[pos] == ZWJ or text[pos] == ZWNJ):
                if text[pos] == NUKTA:
                    nuktas[-1] = True
                elif text[pos] == ZWJ:
                    has_zwj = True
                else:
                    has_zwnj = True
                pos += 1
                advanced = True
            if pos < n and text[pos] == VIRAMA:
                lookahead = pos + 1
                if lookahead < n and text[lookahead] in CONSONANTS:
                    onset.append(text[lookahead])
                    nuktas.append(False)
                    pos = lookahead + 1
                    advanced = True
                    continue
            if not advanced:
                break

    # trailing decoration: matra (vowel, consonant-branch only) and modifiers,
    # in any order/repetition - akshara.py's own _scan_tail absorbs these the
    # same order-agnostic way.
    while pos < n:
        ch = text[pos]
        if ch in MATRAS:
            vowel = ch  # last one wins on malformed repeated-matra input
            pos += 1
        elif ch in MODIFIERS:
            modifiers.append(ch)
            pos += 1
        elif ch == NUKTA:
            pos += 1  # stray trailing nukta on malformed input; already lossless via chunk.text
        elif ch == ZWJ:
            has_zwj = True
            pos += 1
        elif ch == ZWNJ:
            has_zwnj = True
            pos += 1
        elif ch == VIRAMA:
            pos += 1  # dangling terminal virama already absorbed by aksharas(); nothing to classify
        else:  # pragma: no cover - defensive; aksharas() guarantees no other codepoint appears here
            pos += 1

    return AksharaFeatures(
        text=text, onset=onset, nuktas=nuktas, vowel=vowel,
        modifiers=modifiers, has_zwj=has_zwj, has_zwnj=has_zwnj,
    )


def featurize(text: str) -> list:
    """Segment and featurize `text`: normalize, parse into aksharas, and
    decompose every "akshara"-kind chunk. "Other"-kind chunks (foreign
    scripts, ASCII, punctuation, whitespace) are returned unchanged as plain
    `Akshara` objects - a non-Bengali chunk doing its documented job is not
    a failure, so this never raises for that reason.

    Raises:
      NormalizationError: if `text` is not a `str` (from `normalize()`).
    """
    chunks = aksharas(normalize(text))
    return [featurize_akshara(c) if c.kind == "akshara" else c for c in chunks]
