"""
bntok: the Project Bornomala Track A Bengali tokenizer.

A Bengali-first subword tokenizer that never splits a grapheme cluster across
token boundaries. It normalises to NFC, segments into UAX #29 grapheme clusters,
remaps each cluster to an atomic symbol, and trains BPE or Unigram over atoms, so
conjunct fragmentation is structurally impossible.

Public API:

    from bntok import BengaliTokenizer, evaluate
    tok = BengaliTokenizer.train(corpus, algo="bpe", vocab_size=32000)
    ids = tok.encode("আমি বাংলায় গান গাই")
    text = tok.decode(ids)
    tok.save("out/tok")
    report = evaluate(tok, held_out_texts)
"""

from . import corpus, errors, shaping
from .akshara import Akshara, aksharas
from .atoms import AtomMap
from .evaluate import Report, evaluate
from .graphemes import grapheme_clusters
from .normalize import normalize
from .tokenizer import BengaliTokenizer

__version__ = "0.1.0"

__all__ = [
    "Akshara",
    "AtomMap",
    "BengaliTokenizer",
    "Report",
    "__version__",
    "aksharas",
    "corpus",
    "errors",
    "evaluate",
    "grapheme_clusters",
    "normalize",
    "shaping",
]
