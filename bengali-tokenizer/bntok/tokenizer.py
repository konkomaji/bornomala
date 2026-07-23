r"""
BengaliTokenizer: the Track A tokenizer (Project Bornomala).

A Bengali-first subword tokenizer that, by construction, never splits a grapheme
cluster across token boundaries. It composes the pieces in this package:

    text
      -> normalize (NFC + ZWJ/ZWNJ policy)            normalize.py
      -> grapheme clusters (UAX #29)                  graphemes.py
      -> atoms (one symbol per cluster, reversible)   atoms.py
      -> subword model (BPE or Unigram) over atoms    Hugging Face tokenizers
      -> token ids

Because the subword model sees atoms (whole grapheme clusters), every learned
token is a sequence of whole clusters. Conjunct fragmentation is structurally
impossible (spec section 9.2). Word boundaries are carried by a Metaspace marker,
the standard SentencePiece scheme, so decode restores the exact NFC text.

The artifact is portable as a directory: tokenizer.json (the atom-space subword
model), atoms.json (the cluster<->atom map), and config.json (algorithm, vocab
size, normalisation policy). Load the directory to encode and decode.
"""

from __future__ import annotations

import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .atoms import AtomMap
from .errors import (
    DecodeError,
    EmptyCorpusError,
    EncodeError,
    LoadError,
    TrainingError,
    VocabSizeError,
    require,
)
from .graphemes import GUARANTEED_CODEPOINTS
from .normalize import normalize

UNK = "<unk>"
_SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]


class BengaliTokenizer:
    def __init__(self, tok: Tokenizer, atoms: AtomMap, config: dict):
        self._tok = tok
        self._atoms = atoms
        self.config = config

    # ---------------- training ----------------
    @classmethod
    def train(
        cls,
        corpus: list[str],
        algo: str = "bpe",
        vocab_size: int = 32000,
        min_atom_freq: int = 2,
        zwnj_policy: str = "preserve",
    ) -> BengaliTokenizer:
        """Induce a tokenizer from a corpus (a list of raw strings).

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

        # 1) normalise, then build the atom map from normalised text. Bad lines
        #    are skipped rather than aborting the whole run.
        norm = []
        skipped = 0
        for line in corpus:
            if not isinstance(line, str):
                skipped += 1
                continue
            s = normalize(line, zwnj_policy=zwnj_policy)
            if s.strip():
                norm.append(s)
        if not norm:
            raise EmptyCorpusError("corpus contained no usable text after normalisation")

        # Guarantee an atom for the whole Bengali block and ASCII, so any Bengali
        # or code-mixed text round-trips regardless of what the corpus contained.
        atoms = AtomMap.build(norm, min_freq=min_atom_freq, guarantee=GUARANTEED_CODEPOINTS)

        # The vocabulary must at least hold the atom alphabet plus specials.
        floor = len(atoms) + len(_SPECIALS)
        if vocab_size < floor:
            raise VocabSizeError(
                f"vocab_size={vocab_size} is below the atom alphabet size ({floor}); "
                f"raise vocab_size or raise min_atom_freq to shrink the alphabet"
            )

        # 2) remap corpus to atom space (whitespace passes through literally so the
        #    Metaspace pre-tokenizer can mark word boundaries).
        atomised = [atoms.encode(line) for line in norm]

        # Force every atom into the base vocabulary via initial_alphabet, so every
        # covered cluster/codepoint is a guaranteed single token and round-trips.
        initial_alphabet = list(atoms.cluster_to_atom.values())

        # 3) atom-space subword model.
        if algo == "bpe":
            tok = Tokenizer(models.BPE(unk_token=UNK))
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size, special_tokens=_SPECIALS,
                initial_alphabet=initial_alphabet, show_progress=False,
            )
        else:
            tok = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size, special_tokens=_SPECIALS, unk_token=UNK,
                initial_alphabet=initial_alphabet, show_progress=False,
            )

        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
        tok.decoder = decoders.Metaspace(replacement="▁")

        try:
            tok.train_from_iterator(atomised, trainer=trainer, length=len(atomised))
        except Exception as e:  # the Rust trainer raises assorted exception types
            raise TrainingError(f"{algo} training failed: {type(e).__name__}: {e}") from e

        config = {
            "algo": algo,
            "vocab_size": vocab_size,
            "actual_vocab_size": tok.get_vocab_size(),
            "min_atom_freq": min_atom_freq,
            "zwnj_policy": zwnj_policy,
            "num_atoms": len(atoms),
            "num_cluster_atoms": atoms.num_cluster_atoms,
            "format": "bornomala-track-a/1",
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
        """Human-readable token surfaces (each token mapped back to clusters)."""
        toks = self._tok.encode(self._atomise(text)).tokens
        out = []
        for t in toks:
            # a token is a run of atoms possibly prefixed by the Metaspace marker.
            readable = self._atoms.decode(t.replace("▁", " "))
            out.append(readable if readable else t)
        return out

    def decode(self, ids: list[int]) -> str:
        if not hasattr(ids, "__iter__"):
            raise DecodeError(f"expected an iterable of ids, got {type(ids).__name__}")
        try:
            atomstr = self._tok.decode(list(ids))    # Metaspace restores spaces
            text = self._atoms.decode(atomstr)
            # Metaspace introduces a boundary space at the start (its word marker)
            # and can leave a trailing one. Surrounding whitespace is not carried
            # by subword tokenizers; trim it so decode returns exact content.
            return text.strip(" ")
        except Exception as e:  # pragma: no cover - defensive
            raise DecodeError(f"decode failed: {type(e).__name__}: {e}") from e

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # ---------------- persistence ----------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._tok.save(os.path.join(directory, "tokenizer.json"))
        self._atoms.save(os.path.join(directory, "atoms.json"))
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> BengaliTokenizer:
        if not os.path.isdir(directory):
            raise LoadError(f"not a tokenizer directory: {directory}")
        tpath = os.path.join(directory, "tokenizer.json")
        cpath = os.path.join(directory, "config.json")
        for p in (tpath, cpath):
            if not os.path.exists(p):
                raise LoadError(f"missing file in tokenizer directory: {p}")
        try:
            tok = Tokenizer.from_file(tpath)
            atoms = AtomMap.load(os.path.join(directory, "atoms.json"))
            with open(cpath, encoding="utf-8") as f:
                config = json.load(f)
        except LoadError:
            raise
        except Exception as e:
            raise LoadError(f"could not load tokenizer from {directory}: {e}") from e
        return cls(tok, atoms, config)

    # ---------------- self-check ----------------
    def roundtrip_ok(self, text: str) -> bool:
        """True if encode then decode reproduces the normalised input.

        This is the integrity guarantee: with the atom scheme it should always
        hold for text whose codepoints were seen at training time.
        """
        norm = normalize(text, zwnj_policy=self.config["zwnj_policy"])
        return self.decode(self.encode(text)) == norm.strip(" ")

    def content_roundtrip_ok(self, text: str) -> bool:
        """Round-trip ignoring all whitespace differences (content fidelity only)."""
        norm = normalize(text, zwnj_policy=self.config["zwnj_policy"])

        def strip_ws(s: str) -> str:
            return "".join(s.split())

        return strip_ws(self.decode(self.encode(text))) == strip_ws(norm)
