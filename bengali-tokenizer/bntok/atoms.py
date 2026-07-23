r"""
Atom mapping: the mechanism that guarantees conjunct integrity (Track A core).

Standard BPE and Unigram operate on characters (codepoints). On Bengali that is
wrong: a merge can end in the middle of a conjunct, splitting a grapheme cluster
across two tokens. That is the single failure mode this tokenizer exists to
avoid (spec section 9.2, requirement: conjunct fragmentation rate = 0).

The fix is to change the atomic unit the subword model sees. We segment text into
grapheme clusters (graphemes.py), then remap each cluster to a single "atom"
codepoint drawn from the Unicode Private Use Area. The subword model is then
trained over atoms: every merge combines whole clusters, so no token can ever
split a cluster. Conjunct fragmentation becomes structurally impossible, not
merely rare.

Coverage and fallback (spec section 10.4, UNK-decomposition fallback):
  * Frequent grapheme clusters (freq >= min_freq in the induction corpus) each
    get their own atom.
  * Every individual codepoint that appears also gets an atom. So any cluster
    that is not itself frequent still maps, codepoint by codepoint, to atoms.
    Rare clusters therefore decompose rather than becoming unknown.
  * A single reserved UNK atom covers codepoints unseen at training time.

The map is fully reversible, so decode reconstructs the exact NFC text. The map
is saved next to the tokenizer (atoms.json) and reloaded for encode / decode, so
the pair (tokenizer.json, atoms.json) is a portable artifact.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable

from .graphemes import grapheme_clusters
from .errors import EmptyCorpusError, LoadError, ConfigError

# Private Use Area planes 15 and 16: about 131,000 usable codepoints. Far more
# than the few thousand grapheme clusters Bengali needs, with room for code-mixed
# Latin and digits.
_PUA_RANGES = [(0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
_UNK = ""  # reserved atom for codepoints unseen at build time


def _pua_generator():
    for lo, hi in _PUA_RANGES:
        for cp in range(lo, hi + 1):
            yield chr(cp)


class AtomMap:
    """A reversible map between grapheme clusters and single atom codepoints."""

    def __init__(self, cluster_to_atom: dict[str, str], min_freq: int = 1):
        self.cluster_to_atom = cluster_to_atom
        self.atom_to_cluster = {a: c for c, a in cluster_to_atom.items()}
        self.min_freq = min_freq
        self.atom_to_cluster[_UNK] = ""  # UNK decodes to nothing (lossy, rare)

    # ---- build ----
    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 2,
        guarantee: Iterable[str] | None = None,
    ) -> "AtomMap":
        """Build an atom map from a corpus of NFC-normalised strings.

        Every codepoint in `guarantee` is given an atom even if absent from the
        corpus, so round-trip is guaranteed for that character set (typically the
        whole Bengali block plus ASCII). Non-string lines are skipped defensively.
        Raises EmptyCorpusError if no usable text is found, and ConfigError if the
        atom budget is exhausted.
        """
        if min_freq < 1:
            raise ConfigError(f"min_freq must be >= 1, got {min_freq}")

        cluster_freq: Counter[str] = Counter()
        codepoints: set[str] = set()
        seen_any = False
        for line in corpus:
            if not isinstance(line, str) or not line:
                continue
            for g in grapheme_clusters(line):
                if g.isspace():
                    continue  # whitespace stays literal (carries word boundaries)
                seen_any = True
                cluster_freq[g] += 1
                if len(g) > 1:
                    codepoints.update(g)
                else:
                    codepoints.add(g)

        if not seen_any:
            raise EmptyCorpusError("atom map: corpus contained no usable text")

        # Guaranteed coverage: fold in codepoints that must always have an atom.
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
                    "atom budget exhausted (>131000 distinct clusters); "
                    "raise min_freq to reduce the atom set"
                )

        # 1) every individual codepoint gets an atom (guarantees the fallback).
        for cp in sorted(codepoints):
            cluster_to_atom[cp] = _next_atom()
        # 2) frequent multi-codepoint clusters get their own atom.
        for cluster, freq in cluster_freq.most_common():
            if len(cluster) > 1 and freq >= min_freq and cluster not in cluster_to_atom:
                cluster_to_atom[cluster] = _next_atom()

        return cls(cluster_to_atom, min_freq=min_freq)

    # ---- encode / decode ----
    def encode(self, nfc_text: str) -> str:
        """Map NFC text to a string of atoms (one atom per known cluster,
        else one atom per constituent codepoint, else UNK)."""
        out = []
        for g in grapheme_clusters(nfc_text):
            if g.isspace():
                out.append(g)  # keep whitespace literal for the word-boundary marker
                continue
            a = self.cluster_to_atom.get(g)
            if a is not None:
                out.append(a)
            else:
                for cp in g:
                    out.append(self.cluster_to_atom.get(cp, _UNK))
        return "".join(out)

    def decode(self, atom_text: str) -> str:
        """Map a string of atoms back to text.

        Known atoms map to their cluster; the reserved UNK atom maps to nothing;
        any other character (spaces, newlines, the Metaspace marker once already
        converted) passes through unchanged.
        """
        out = []
        for a in atom_text:
            if a == _UNK:
                continue
            out.append(self.atom_to_cluster.get(a, a))
        return "".join(out)

    # ---- persistence ----
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"min_freq": self.min_freq, "cluster_to_atom": self.cluster_to_atom},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "AtomMap":
        if not os.path.exists(path):
            raise LoadError(f"atom map not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(d["cluster_to_atom"], min_freq=d.get("min_freq", 1))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise LoadError(f"corrupt atom map {path}: {e}") from e

    def __len__(self) -> int:
        return len(self.cluster_to_atom)

    @property
    def num_cluster_atoms(self) -> int:
        return sum(1 for c in self.cluster_to_atom if len(c) > 1)
