"""
MotherTongueIndex - multilingual tokenizer efficiency analyzer.

Runs the real tokenizers of mainstream LLMs against text in any language and
reports how efficiently each one encodes it: token count, fertility
(tokens/word), bytes-per-token, characters-per-token, single-token retention,
and a per-script breakdown.

It is an *understanding* tool, not a cost calculator. It answers "why does this
language cost more?" by showing exactly how each model's tokenizer fragments the
text, and how that fragmentation differs across scripts.

The same engine produces the cross-tokenizer fertility/STRR comparison tables
that Project Bornomala's Track A requires (spec §4.1, §9.4).
"""

from .analyze import analyze, analyze_many
from .metrics import Metrics
from .registry import MODELS, get_model, list_models
from .segment import grapheme_clusters, words

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "Metrics",
    "__version__",
    "analyze",
    "analyze_many",
    "get_model",
    "grapheme_clusters",
    "list_models",
    "words",
]
