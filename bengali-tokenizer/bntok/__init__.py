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

BMBT (Bornomala's Bengali Tokenizer, v2) parses the akshara grammar directly
instead of discovering structure statistically, and exposes a real featural
decomposition (onset/vowel/modifier) alongside the usual token ids:

    from bntok import BMBT, featurize
    tok = BMBT.train(corpus, algo="bpe", vocab_size=64000)
    ids = tok.encode("আমি বাংলায় গান গাই")
    for f in featurize("স্ত্রী"):
        print(f.onset, f.vowel)
"""

from . import corpus, errors, shaping
from .akshara import Akshara, aksharas
from .atoms import AtomMap
from .bmbt import BMBT, AksharaFeatures, featurize, featurize_akshara
from .evaluate import Report, evaluate
from .graphemes import grapheme_clusters
from .normalize import normalize
from .tokenizer import BengaliTokenizer

__version__ = "0.1.0"

__all__ = [
    "BMBT",
    "Akshara",
    "AksharaFeatures",
    "AtomMap",
    "BengaliTokenizer",
    "Report",
    "__version__",
    "aksharas",
    "corpus",
    "errors",
    "evaluate",
    "featurize",
    "featurize_akshara",
    "grapheme_clusters",
    "normalize",
    "shaping",
]
