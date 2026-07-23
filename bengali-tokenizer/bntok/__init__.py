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

from .tokenizer import BengaliTokenizer
from .evaluate import evaluate, Report
from .atoms import AtomMap
from .normalize import normalize
from .graphemes import grapheme_clusters
from . import shaping, corpus, errors

__version__ = "0.1.0"

__all__ = [
    "BengaliTokenizer",
    "evaluate",
    "Report",
    "AtomMap",
    "normalize",
    "grapheme_clusters",
    "shaping",
    "corpus",
    "errors",
    "__version__",
]
