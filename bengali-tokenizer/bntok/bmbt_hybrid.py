r"""
BMBT-Hybrid: a frequency-adaptive atom scheme built on top of BMBT's own
akshara grammar (v2 follow-up experiment; not yet the shipped BMBT).

BMBT (bmbt.py) maps every akshara `aksharas()` finds to ONE atom before BPE
runs. Measured against v1 (`bn-bpe-64k`) at full production scale, that
ties exactly -- matching `docs/design/FORMAL_SPEC.md`'s own proof that a
BPE constrained to respect akshara boundaries cannot beat an unconstrained
one on raw token count.

This module explores an orthogonal lever the proof doesn't touch: WHAT the
atomic unit is, not how BPE merges it. Two ideas, both measured, not
assumed:

  1. FACTORED atoms. A rare akshara (say, an uncommon onset conjunct paired
     with an uncommon vowel) spends its whole corpus frequency on one
     opaque, rarely-repeated atom. Splitting it into an onset atom (the
     consonant chain) and a tail atom (matra + modifiers) lets that same
     onset accumulate frequency across every vowel it occurs with, and
     that same tail accumulate frequency across every onset it attaches
     to -- something only possible because the akshara grammar already
     knows where that internal boundary is; v1 has no such boundary to
     split on, its atoms are opaque grapheme-cluster strings.

  2. HYBRID allocation. Factoring the very highest-frequency syllables
     (common roots like a bare "কর") is pure overhead: fusing costs one
     atom, factoring costs two, and a syllable that frequent doesn't need
     the cross-combination reuse factoring buys for the long tail. So only
     the top `k_fused` most frequent akshara chunks are kept fused (one
     atom, like BMBT); everything else is factored.

Measured (small-scale proof-of-concept, 8k vocab / 3k Wikipedia articles):
fused-only 1.9587 -> always-factored 1.8700 -> hybrid k=200 1.8454. At full
production scale (configs/bpe-64k.json, 64k vocab, all four registers),
hybrid genuinely beat v1/BMBT on fertility (1.2-3.8% lower) -- but paid for
it with 20-70x worse conjunct fragmentation (0.18-0.70% vs v1/BMBT's flat
0.01%): whenever the learned BPE merges didn't happen to re-fuse a factored
onset+tail pair, the akshara boundary survived as a real split in the final
output.

There is no "bias merge priority toward re-fusion" knob in
`tokenizers.trainers.BpeTrainer` -- it is a pure frequency-driven learner.
A first attempt at a fix baked extra low-priority MERGE RULES into the
trained model for every factored (and per-codepoint-fallback) atom
sequence, predicting the exact intermediate symbols a left-to-right fold
would produce. That turned out unsound: real BPE always applies whichever
adjacent pair has the globally lowest rank first, not left-to-right, so a
high-priority learned merge can fuse a MIDDLE pair before the predicted
chain reaches it, producing an intermediate symbol the predicted chain
never accounted for -- confirmed by a real failing case
(`ঙ্ক্ষ` partially merging to `ঙ্` + `ক্` + `ষ`, two of three still split)
before this module shipped, not discovered later.

The actual fix (`_reserve_guaranteed_chunk_vocab` + the encode-time guard
in `BMBTHybrid.encode`) sidesteps merge-order prediction entirely: for
every distinct akshara chunk seen during training that maps to more than
one atom, a dedicated vocab id is reserved for its FULL atom span directly
-- no merge rule needed to reach it. At encode time, real BPE runs exactly
as trained (so genuine cross-chunk compression merges are still used
wherever they occur); afterward, any chunk whose span ends up split across
more than one final token has those tokens replaced with its single
reserved id. This is the same forbidden-cut-point technique verified by
direct measurement against `artifacts/bmbt-64k-hybrid` before this module
existed (0.0000 fragmentation on 3 of 4 registers, 0.000005 on
literary_formal -- the exact same cross-script OCR-noise residual already
documented for v1/BMBT in `docs/known-issues.md` point 14, not a new
failure mode), while fertility improved FURTHER versus the unguarded
hybrid scheme (merging a forced pair back together is a pure token-count
win): 1.2-3.8% better than v1/BMBT on every register, with equal-or-better
fragmentation. See `docs/known-issues.md` and
`benchmarks/bengali-comparison.md` for the full account. A chunk never
seen during training has no reservation and can, rarely, still show the
same "unseen cluster decomposes" residual v1/BMBT already accept -- not a
regression, the same documented limitation extended to a new atom scheme.

Self-contained the same way bmbt.py is: imports only from akshara.py,
substrate.py, normalize.py, errors.py, bmbt.py's featural decomposition
(reused rather than duplicated a third time), and the `tokenizers`
library. Nothing here can affect v1 or BMBT's own artifacts.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .akshara import aksharas
from .bmbt import AksharaFeatures, FeaturizeError, featurize, featurize_akshara  # noqa: F401 (re-exported)
from .errors import (
    ConfigError,
    DecodeError,
    EmptyCorpusError,
    EncodeError,
    LoadError,
    TrainingError,
    VocabSizeError,
    require,
)
from .normalize import normalize
from .substrate import (
    CONSONANTS,
    GUARANTEED_CODEPOINTS,
    NUKTA,
    VIRAMA,
    VOWELS,
    ZWJ,
    ZWNJ,
)

UNK_TOKEN = "<unk>"
_SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
METASPACE = "▁"

# Same PUA ranges as atoms.py/bmbt.py. A distinct UNK atom (one below
# BMBT's 0xF8FE, two below v1's 0xF8FF) keeps all three artifacts'
# atom spaces provably non-colliding by construction.
_PUA_RANGES = [(0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
_UNK_ATOM = chr(0xF8FD)

# The measured winner of the k-sweep (200/800/2000 tested at small scale;
# 200 won there and is what the full-scale run and safety-net measurement
# above both used). Exposed as a default, not hardcoded, so a future sweep
# at full scale can override it without editing this module.
DEFAULT_K_FUSED = 200


def _pua_generator():
    for lo, hi in _PUA_RANGES:
        for cp in range(lo, hi + 1):
            yield chr(cp)


def _onset_tail_split(text: str) -> tuple[str, str]:
    """Split one akshara's text into (onset, tail): the consonant chain
    (virama-joined, nukta/ZWJ/ZWNJ absorbed) and everything after it
    (matra/modifiers). Mirrors bmbt.featurize_akshara's own onset-scan
    exactly, but returns the raw substring split point instead of a
    structured decomposition -- a strictly smaller problem, since we only
    need where the cut falls, not what's on each side of it.

    Vowel-initial aksharas are not split (returned whole as `onset`, empty
    tail): there is no onset to factor out, and splitting off an empty
    tail would be pure overhead for no benefit.
    """
    n = len(text)
    if text[0] in VOWELS:
        return text, ""
    pos = 1
    while True:
        advanced = False
        while pos < n and text[pos] in (NUKTA, ZWJ, ZWNJ):
            pos += 1
            advanced = True
        if pos < n and text[pos] == VIRAMA:
            look = pos + 1
            if look < n and text[look] in CONSONANTS:
                pos = look + 1
                advanced = True
                continue
        if not advanced:
            break
    return text[:pos], text[pos:]


class HybridAtomMap:
    """A reversible map between akshara/other chunk text and atoms, where a
    chunk is either FUSED (one atom, if its whole text is in `fused_set`)
    or FACTORED (two atoms: onset, then tail -- only for "akshara"-kind
    chunks with a non-empty tail; everything else is fused by definition,
    including "other" chunks, which is the same behaviour AksharaAtomMap
    already has for those).
    """

    def __init__(self, piece_to_atom: dict[str, str], fused_set, min_freq: int = 1):
        self.piece_to_atom = piece_to_atom
        self.atom_to_piece = {a: p for p, a in piece_to_atom.items()}
        self.fused_set = frozenset(fused_set)
        self.min_freq = min_freq
        self.atom_to_piece[_UNK_ATOM] = ""

    # ---- chunk -> piece(s) ----
    def _pieces_for_chunk(self, chunk) -> list[str]:
        text = chunk.text
        if chunk.kind != "akshara" or text in self.fused_set:
            return [text]
        onset, tail = _onset_tail_split(text)
        return [onset, tail] if tail else [onset]

    def _atoms_for_piece(self, piece: str) -> list[str]:
        a = self.piece_to_atom.get(piece)
        if a is not None:
            return [a]
        return [self.piece_to_atom.get(cp, _UNK_ATOM) for cp in piece]

    # ---- build ----
    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        k_fused: int = DEFAULT_K_FUSED,
        min_freq: int = 2,
        guarantee: Iterable[str] | None = None,
    ) -> HybridAtomMap:
        """Build a hybrid atom map from a corpus of NFC-normalised strings.

        Two passes: first rank whole akshara chunks by raw corpus
        frequency and keep the top `k_fused` as the fused set; second,
        build the piece (fused-or-factored) frequency table and assign
        atoms exactly like AksharaAtomMap.build does (every codepoint
        guaranteed, every frequent multi-codepoint piece its own atom).

        Raises EmptyCorpusError if no usable text is found, ConfigError
        if k_fused is negative or the atom budget is exhausted.
        """
        if min_freq < 1:
            raise ConfigError(f"min_freq must be >= 1, got {min_freq}")
        if k_fused < 0:
            raise ConfigError(f"k_fused must be >= 0, got {k_fused}")

        lines = list(corpus)
        akshara_freq: Counter[str] = Counter()
        seen_any = False
        for line in lines:
            if not isinstance(line, str) or not line:
                continue
            for chunk in aksharas(line):
                if chunk.kind == "akshara" and not chunk.text.isspace():
                    akshara_freq[chunk.text] += 1
                    seen_any = True
        if not seen_any:
            raise EmptyCorpusError("hybrid atom map: corpus contained no usable akshara text")

        fused_set = frozenset(t for t, _ in akshara_freq.most_common(k_fused))

        chunk_freq: Counter[str] = Counter()
        codepoints: set[str] = set()
        self_probe = cls({}, fused_set)
        for line in lines:
            if not isinstance(line, str) or not line:
                continue
            for chunk in aksharas(line):
                text = chunk.text
                if text.isspace():
                    continue
                for piece in self_probe._pieces_for_chunk(chunk):
                    chunk_freq[piece] += 1
                    if len(piece) > 1:
                        codepoints.update(piece)
                    else:
                        codepoints.add(piece)

        for cp in (guarantee or ()):
            if isinstance(cp, str) and len(cp) == 1 and not cp.isspace():
                codepoints.add(cp)

        gen = _pua_generator()
        piece_to_atom: dict[str, str] = {}

        def _next_atom() -> str:
            try:
                return next(gen)
            except StopIteration:  # pragma: no cover - needs >131k distinct atoms
                raise ConfigError(
                    "hybrid atom budget exhausted (>131000 distinct pieces); "
                    "raise min_freq to reduce the piece set"
                )

        for cp in sorted(codepoints):
            piece_to_atom[cp] = _next_atom()
        for text, freq in chunk_freq.most_common():
            if len(text) > 1 and freq >= min_freq and text not in piece_to_atom:
                piece_to_atom[text] = _next_atom()

        return cls(piece_to_atom, fused_set, min_freq=min_freq)

    # ---- encode / decode ----
    def encode(self, nfc_text: str, collect: set | None = None) -> str:
        """Map NFC text to a string of atoms. If `collect` is given, every
        chunk that ends up spanning MORE than one atom has its full,
        ordered atom tuple added to it -- not just the onset/tail seam,
        since a rare onset or tail piece can itself fall below
        `min_atom_freq` and decompose further, per codepoint (the same
        residual-decomposition path v1/BMBT's own atom scheme has). Used
        during training to know exactly which chunk spans need a
        reserved whole-chunk vocab entry (see `BMBTHybrid.train`)."""
        out = []
        for chunk in aksharas(nfc_text):
            text = chunk.text
            if text.isspace():
                out.append(text)
                continue
            pieces = self._pieces_for_chunk(chunk)
            chunk_atoms = []
            for p in pieces:
                chunk_atoms.extend(self._atoms_for_piece(p))
            out.extend(chunk_atoms)
            if collect is not None and len(chunk_atoms) > 1:
                collect.add(tuple(chunk_atoms))
        return "".join(out)

    def encode_with_spans(self, nfc_text: str) -> tuple[str, list[tuple[int, int]]]:
        """Like `encode`, but also returns each akshara/other chunk's
        [start, end) span in the returned atom string's own coordinates
        (codepoint-indexed) -- what the encode-time guard needs to tell a
        genuinely fragmented chunk apart from a real, safe cross-chunk
        merge."""
        out = []
        spans = []
        pos = 0
        for chunk in aksharas(nfc_text):
            text = chunk.text
            if text.isspace():
                out.append(text)
                pos += len(text)
                continue
            start = pos
            for p in self._pieces_for_chunk(chunk):
                for a in self._atoms_for_piece(p):
                    out.append(a)
                    pos += len(a)
            spans.append((start, pos))
        return "".join(out), spans

    def decode(self, atom_text: str) -> str:
        out = []
        for a in atom_text:
            out.append(self.atom_to_piece.get(a, a))
        return "".join(out)

    # ---- persistence ----
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_freq": self.min_freq,
                    "fused_set": sorted(self.fused_set),
                    "piece_to_atom": self.piece_to_atom,
                },
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> HybridAtomMap:
        if not os.path.exists(path):
            raise LoadError(f"hybrid atom map not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(d["piece_to_atom"], d["fused_set"], min_freq=d.get("min_freq", 1))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise LoadError(f"corrupt hybrid atom map {path}: {e}") from e

    def __len__(self) -> int:
        return len(self.piece_to_atom)

    @property
    def num_cluster_atoms(self) -> int:
        return sum(1 for c in self.piece_to_atom if len(c) > 1)


def _reserve_guaranteed_chunk_vocab(tok: Tokenizer, atom_chains: set[tuple[str, ...]]) -> tuple[Tokenizer, int]:
    """Reserve a dedicated vocab id for every multi-atom chunk span seen
    during training -- no merge rule involved, so this cannot depend on,
    or be defeated by, whatever order the learned merges happen to apply
    in (see the module docstring for why an earlier merge-chain-based
    attempt at this was unsound). `BMBTHybrid.encode`'s guard looks these
    ids up directly whenever a chunk's span isn't already exactly one
    token after real BPE encoding.

    Both the bare form and the Metaspace-prefixed form are reserved: a
    guard-substituted chunk might be the first syllable of a (non-first)
    word, and the tokenizer's own Metaspace decoder only restores that
    word's leading space if the token string it decodes actually carries
    the "▁" marker -- a bare reservation alone would silently swallow the
    space before that word when decoding ids back to text.
    """
    d = json.loads(tok.to_str())
    vocab = d["model"]["vocab"]
    next_id = max(vocab.values()) + 1 if vocab else 0
    added = 0
    for chain in sorted(atom_chains):
        fused = "".join(chain)
        for candidate in (fused, METASPACE + fused):
            if candidate not in vocab:
                vocab[candidate] = next_id
                next_id += 1
                added += 1
    d["model"]["vocab"] = vocab
    return Tokenizer.from_str(json.dumps(d)), added


class BMBTHybrid:
    """BMBT-Hybrid: BMBT's akshara grammar plus a frequency-adaptive atom
    scheme (top-`k_fused` akshara chunks fused, the rest onset/tail
    factored) plus a reserved-vocab encode-time guard that guarantees no
    factored akshara boundary survives as a real token split.

    Mirrors `BMBT`'s public shape (train/encode/decode/save/load/
    roundtrip_ok/content_roundtrip_ok/featurize) deliberately, so
    `evaluate.py`'s `evaluate()` function and `scripts/compare.py`'s
    measurement functions work on a trained BMBTHybrid with only a
    mechanical class-name swap.
    """

    def __init__(self, tok: Tokenizer, atoms: HybridAtomMap, config: dict):
        self._tok = tok
        self._atoms = atoms
        self.config = config

    # ---------------- training ----------------
    @classmethod
    def train(
        cls,
        corpus: list[str],
        vocab_size: int = 64000,
        min_atom_freq: int = 2,
        zwnj_policy: str = "preserve",
        k_fused: int = DEFAULT_K_FUSED,
    ) -> BMBTHybrid:
        """Induce a BMBTHybrid tokenizer from a corpus (a list of raw strings).

        Only BPE is supported (Unigram measured worse for this atom scheme
        at full scale -- see benchmarks/bengali-comparison.md -- so it is
        not exposed as a choice here rather than silently defaulting to a
        known-worse option).

        Raises:
          ConfigError: invalid vocab_size, k_fused, or corpus type.
          EmptyCorpusError: corpus has no usable text.
          VocabSizeError: vocab_size cannot hold even the atom alphabet.
          TrainingError: the underlying subword trainer failed.
        """
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

        atoms = HybridAtomMap.build(norm, k_fused=k_fused, min_freq=min_atom_freq,
                                     guarantee=GUARANTEED_CODEPOINTS)

        floor = len(atoms) + len(_SPECIALS)
        if vocab_size < floor:
            raise VocabSizeError(
                f"vocab_size={vocab_size} is below the hybrid atom alphabet size "
                f"({floor}); raise vocab_size or raise min_atom_freq to shrink the alphabet"
            )

        atom_chains: set[tuple[str, ...]] = set()
        atomised = [atoms.encode(line, collect=atom_chains) for line in norm]
        initial_alphabet = list(atoms.piece_to_atom.values())

        tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=_SPECIALS,
            initial_alphabet=initial_alphabet, show_progress=False,
        )
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=METASPACE)
        tok.decoder = decoders.Metaspace(replacement=METASPACE)

        try:
            tok.train_from_iterator(atomised, trainer=trainer, length=len(atomised))
        except Exception as e:  # the Rust trainer raises assorted exception types
            raise TrainingError(f"bpe training failed: {type(e).__name__}: {e}") from e

        tok, n_reserved = _reserve_guaranteed_chunk_vocab(tok, atom_chains)

        config = {
            "algo": "bpe",
            "vocab_size": vocab_size,
            "actual_vocab_size": tok.get_vocab_size(),
            "min_atom_freq": min_atom_freq,
            "zwnj_policy": zwnj_policy,
            "num_atoms": len(atoms),
            "num_cluster_atoms": atoms.num_cluster_atoms,
            "k_fused": k_fused,
            "num_fused_akshara": len(atoms.fused_set),
            "num_reserved_chunk_vocab": n_reserved,
            "format": "bornomala-bmbt-hybrid/1",
        }
        return cls(tok, atoms, config)

    # ---------------- guarded encode ----------------
    def _atomise(self, text: str) -> str:
        if not isinstance(text, str):
            raise EncodeError(f"expected str, got {type(text).__name__}")
        return self._atoms.encode(normalize(text, zwnj_policy=self.config["zwnj_policy"]))

    def _encode_guarded(self, nfc_text: str) -> tuple[list[int], list[str]]:
        """Real BPE encode of the whole (already-normalised) text in ONE
        call -- letting the tokenizer's own configured Metaspace
        pre_tokenizer handle multi-word splitting/prefixing exactly like
        BMBT/v1 do, rather than re-implementing it per word -- then a
        guard pass using a forbidden-cut-point sweep: any atom-string
        position strictly INSIDE an akshara/other chunk's span is an
        illegal token boundary (it would split that chunk), so it is
        dropped from the set of real token boundaries before
        reconstructing final tokens.

        This can only ever merge tokens together, never split one further,
        and by construction every surviving run is a union of one or more
        WHOLE chunks -- never a partial one -- regardless of what order
        the learned merges happened to apply in (proof: if a run's edge
        were interior to some chunk, that position is in the forbidden
        set and would already have been removed, extending the run past
        it; induction on that argument rules out a partial-chunk edge
        anywhere).

        A run that exactly matches one of the tokenizer's OWN real token
        spans (no forbidden position needed removing near it) reuses that
        token's real id and surface directly -- necessary, not just an
        optimisation: Metaspace bundles a word's leading space INTO its
        first real token's offset span (confirmed by direct inspection: a
        token for " def" covers offsets that include the space), but the
        underlying vocab STRING for that token uses "▁", not a literal
        space, so slicing the literal space out of `atom_str` and doing a
        fresh vocab lookup on it can never match. Reusing the real id
        sidesteps this entirely.

        A run that does NOT match a real token span (created by merging
        away a forbidden position) resolves to its reserved whole-chunk
        vocab id (see `_reserve_guaranteed_chunk_vocab`) -- these merges
        only ever occur strictly inside a chunk's own span, which never
        contains whitespace, so no space-vs-▁ mismatch is possible there.
        The Metaspace-prefixed reservation is used if the run starts a
        (non-first) word, so the decoder still restores that word's
        leading space.

        A run can still fail a whole-span vocab lookup even though no
        chunk inside it was split -- this happens whenever a real BPE
        merge independently fuses one chunk's OUTER edge with a neighbour
        (e.g. a factored chunk's tail atom fusing with the next chunk's
        first atom into its own trained token) on both sides of a
        removed forbidden position: the two flanking real spans collapse
        into one bigger run whose literal text (which can include a
        literal space) was never itself a trained vocab entry. Measured
        as NOT rare at full production scale (unlike the small-scale
        proof-of-concept this guard was first verified against) -- see
        docs/known-issues.md. The fix: such a run is a union of complete
        protected UNITS (each either a whole chunk span or a single
        literal-whitespace character -- `units` below, reconstructed
        directly from `chunk_spans` since aksharas() emits chunk/space in
        strict left-to-right order with no gaps), so it is re-resolved
        unit by unit -- never split further than that -- instead of
        being decomposed per raw codepoint. Each unit's own lookup is
        exactly the same real-span / vocab / reserved-id chain used at
        the top level, just scoped to that one unit, so a factored
        chunk's onset+tail pair is looked up (and found, via its
        reservation) as one piece regardless of what its neighbours did.
        Only a chunk that was never seen during training at all falls
        further to per-atom lookup, and only within that one chunk --
        the same "rare cluster decomposes" residual v1/BMBT already
        document for genuinely unseen input, not a new failure mode.
        """
        atom_str, chunk_spans = self._atoms.encode_with_spans(nfc_text)
        if not atom_str:
            return [], []
        vocab = self._tok.get_vocab()
        enc = self._tok.encode(atom_str)

        # Reconstruct real-token boundaries via cumulative RAW token
        # length, NOT enc.offsets: when Metaspace's "▁" marker isn't
        # merged with the word's first character, a real token can be
        # JUST "▁" with no atom content at all, and HF reports an offset
        # for it that overlaps the very next (real-content) token's own
        # offset -- confirmed by direct inspection. The correction
        # length-tracking needs: Metaspace's prepend_scheme adds an
        # unconditional leading "▁" with NO counterpart in atom_str, but
        # ONLY when atom_str doesn't already start with real whitespace --
        # if it does, that leading "▁" (or run of them) already corresponds
        # 1:1 to real space characters in atom_str (confirmed: 4 leading
        # spaces produce 4 separate "▁" tokens, not a 5th synthetic one),
        # so no correction applies in that case. Every subsequent word's
        # own "▁" always has a real space in atom_str that it's replacing.
        first_token_is_synthetic = bool(atom_str) and not atom_str[0].isspace()
        real_span_to_token = {}
        boundaries = {0, len(atom_str)}
        pos = 0
        for idx, (tid, tsurf) in enumerate(zip(enc.ids, enc.tokens)):
            length = len(tsurf) - 1 if (idx == 0 and first_token_is_synthetic) else len(tsurf)
            start, end = pos, pos + length
            real_span_to_token[(start, end)] = (tid, tsurf)
            boundaries.add(start)
            boundaries.add(end)
            pos = end

        forbidden = set()
        for (cs, ce) in chunk_spans:
            for p in range(cs + 1, ce):
                forbidden.add(p)

        surviving = sorted(b for b in boundaries if b not in forbidden)

        # Full ordered list of protected atomic UNITS covering [0, len)
        # with no gaps: each is either a whole chunk span (from
        # chunk_spans) or a single literal-whitespace position. Any
        # position not inside a chunk span is guaranteed whitespace --
        # encode_with_spans (via aksharas()) emits chunk text and literal
        # space runs in strict left-to-right order, 1:1 with atom_str.
        units: list[tuple[int, int]] = []
        sorted_chunks = sorted(chunk_spans)
        ci = 0
        p = 0
        n = len(atom_str)
        while p < n:
            if ci < len(sorted_chunks) and sorted_chunks[ci][0] == p:
                units.append(sorted_chunks[ci])
                p = sorted_chunks[ci][1]
                ci += 1
            else:
                units.append((p, p + 1))
                p += 1

        ids: list[int] = []
        surfaces: list[str] = []
        # Tracks whether the run just processed already fully accounted for
        # a preceding space (a real "▁"-only token, or a literal-whitespace
        # fallback run) -- needed because a real standalone "▁" token and a
        # reservation's OWN "▁"-prefixed form can both independently exist
        # for the very same logical word boundary; using both double-counts
        # the space (confirmed: produced a real, measured double-space bug
        # before this tracking was added). Threaded through both the
        # outer run loop and the per-unit fallback loop, since either can
        # consume a space.
        prev_run_consumed_space = True
        metaspace_id = vocab.get(METASPACE)

        def emit_content(word_initial: bool, text: str) -> bool:
            """Append `text`'s vocab id (any lookup granularity: whole
            chunk, whole run, or a single fallback atom) plus its surface.
            Returns False if `text` has no vocab entry at all (caller's
            problem to split further or drop it).

            If `text` is word-initial but only its BARE (non-▁-prefixed)
            form is in vocab, the standalone "▁" token id is emitted
            first -- decode() reconstructs purely from `ids` via each id's
            OWN vocab string, recovering a space ONLY from a token whose
            string carries the METASPACE marker, so a bare-form fallback
            at a word boundary would otherwise silently drop the space
            (confirmed: this is exactly what caused "123ঠিক" instead of
            "123 ঠিক" -- a real bug, not a hypothetical, surfaced by the
            per-unit fallback now taking this path more often than the
            original per-codepoint decomposition did).
            """
            nonlocal prev_run_consumed_space
            fid = vocab.get(METASPACE + text) if word_initial else None
            if fid is not None:
                ids.append(fid)
                surfaces.append(text)
                prev_run_consumed_space = False
                return True
            bid = vocab.get(text)
            if bid is None:
                return False
            if word_initial and metaspace_id is not None:
                ids.append(metaspace_id)
                surfaces.append(METASPACE)
            ids.append(bid)
            surfaces.append(text)
            prev_run_consumed_space = False
            return True

        def resolve_unit(us: int, ue: int) -> None:
            nonlocal prev_run_consumed_space
            utext = atom_str[us:ue]
            if utext.isspace():
                # No id emitted here (a literal space has no vocab entry
                # of its own -- Metaspace consumes it into "▁" markers,
                # never trains a bare ' ' token), so this space is NOT
                # yet captured in `ids`. The next content unit must still
                # attempt a ▁-prefixed lookup (word-initial) to carry it;
                # only a REAL standalone "▁" token (below) already
                # captures a space in `ids` and should suppress that.
                surfaces.append(utext)
                prev_run_consumed_space = False
                return
            ureal = real_span_to_token.get((us, ue))
            if ureal is not None:
                tid, tsurf = ureal
                ids.append(tid)
                surfaces.append(tsurf)
                prev_run_consumed_space = (tsurf == METASPACE)
                return
            u_word_initial = us == 0 or (atom_str[us - 1].isspace() and not prev_run_consumed_space)
            if emit_content(u_word_initial, utext):
                return
            # pragma: no cover - genuinely unseen chunk, see docstring
            next_word_initial = u_word_initial
            for cp in utext:
                emit_content(next_word_initial, cp)
                next_word_initial = False

        unit_idx = 0
        for i in range(len(surviving) - 1):
            a, b = surviving[i], surviving[i + 1]
            if b <= a:
                continue

            real = real_span_to_token.get((a, b))
            if real is not None:
                # unmodified run: reuse the tokenizer's own id/surface
                # directly, sidestepping the space-vs-▁ mismatch entirely.
                # `tsurf` stays in raw atom-space (possibly ▁-prefixed) here,
                # same as every other branch -- encode_tokens() decodes all
                # surfaces uniformly in one place, not per-branch.
                tid, tsurf = real
                ids.append(tid)
                surfaces.append(tsurf)
                prev_run_consumed_space = (tsurf == METASPACE)
                while unit_idx < len(units) and units[unit_idx][0] < b:
                    unit_idx += 1
                continue

            span_text = atom_str[a:b]
            if span_text.isspace():
                # See resolve_unit's identical branch: no id emitted, so
                # this space is not yet captured in `ids` -- do not
                # suppress the next content's word-initial lookup.
                surfaces.append(span_text)
                prev_run_consumed_space = False
                while unit_idx < len(units) and units[unit_idx][0] < b:
                    unit_idx += 1
                continue

            word_initial = a == 0 or (atom_str[a - 1].isspace() and not prev_run_consumed_space)
            if emit_content(word_initial, span_text):
                while unit_idx < len(units) and units[unit_idx][0] < b:
                    unit_idx += 1
                continue

            # Whole run has no vocab entry of its own: resolve it as a
            # union of its own protected units instead of decomposing
            # per raw codepoint (see docstring).
            while unit_idx < len(units) and units[unit_idx][1] <= a:
                unit_idx += 1
            while unit_idx < len(units) and units[unit_idx][0] < b:
                us, ue = units[unit_idx]
                resolve_unit(us, ue)
                unit_idx += 1
        return ids, surfaces

    def encode(self, text: str) -> list[int]:
        try:
            nfc = normalize(text, zwnj_policy=self.config["zwnj_policy"])
        except Exception as e:
            raise EncodeError(f"encode failed: {type(e).__name__}: {e}") from e
        try:
            ids, _ = self._encode_guarded(nfc)
            return ids
        except EncodeError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise EncodeError(f"encode failed: {type(e).__name__}: {e}") from e

    def encode_tokens(self, text: str) -> list[str]:
        """Human-readable token surfaces (each token mapped back to atoms' pieces)."""
        nfc = normalize(text, zwnj_policy=self.config["zwnj_policy"])
        _, surfaces = self._encode_guarded(nfc)
        out = []
        for s in surfaces:
            readable = self._atoms.decode(s.replace(METASPACE, " "))
            out.append(readable if readable else s)
        return out

    def decode(self, ids: list[int]) -> str:
        if not hasattr(ids, "__iter__"):
            raise DecodeError(f"expected an iterable of ids, got {type(ids).__name__}")
        try:
            # Deliberately NOT self._tok.decode(ids): HF's own Metaspace
            # decoder inserts an extra space wherever the id sequence
            # contains an isolated "▁"-only token (a real, unmerged token
            # this encoder produces whenever Metaspace's marker never got
            # fused with a word's first character) -- confirmed by direct
            # comparison, it duplicates the space in that case. A direct
            # id->string reverse lookup, followed by one single ▁->" "
            # pass done ourselves, has no such double-counting.
            id_to_str = self._id_to_str()
            atomstr = "".join(id_to_str.get(i, "") for i in ids).replace(METASPACE, " ")
            text = self._atoms.decode(atomstr)
            return text.strip(" ")
        except Exception as e:  # pragma: no cover - defensive
            raise DecodeError(f"decode failed: {type(e).__name__}: {e}") from e

    def _id_to_str(self) -> dict[int, str]:
        cache = getattr(self, "_id_to_str_cache", None)
        if cache is None:
            cache = {v: k for k, v in self._tok.get_vocab().items()}
            self._id_to_str_cache = cache
        return cache

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # ---------------- featural output ----------------
    def featurize(self, text: str) -> list:
        """See bmbt.featurize(); exposed as a convenience method so it is
        usable both with and without a trained tokenizer. Identical to
        BMBT's own featurize() -- the akshara grammar (and therefore the
        featural decomposition) is unchanged, only the atom/vocab scheme
        built on top of it differs."""
        return featurize(text)

    # ---------------- persistence ----------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._tok.save(os.path.join(directory, "tokenizer.json"))
        self._atoms.save(os.path.join(directory, "atoms.json"))
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> BMBTHybrid:
        if not os.path.isdir(directory):
            raise LoadError(f"not a tokenizer directory: {directory}")
        tpath = os.path.join(directory, "tokenizer.json")
        cpath = os.path.join(directory, "config.json")
        for p in (tpath, cpath):
            if not os.path.exists(p):
                raise LoadError(f"missing file in tokenizer directory: {p}")
        try:
            tok = Tokenizer.from_file(tpath)
            atoms = HybridAtomMap.load(os.path.join(directory, "atoms.json"))
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
