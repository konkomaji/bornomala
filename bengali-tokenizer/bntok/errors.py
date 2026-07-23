r"""
Typed error hierarchy for the Track A tokenizer.

Robustness is a first-class requirement here: this tokenizer sits at the root of
the whole Bornomala pipeline, and a silent failure (a mis-shaped conjunct, a
corpus that decoded wrong, a vocab size that quietly clipped) poisons everything
downstream (spec risk R1). So every failure mode has a named exception, every
public entry point validates its inputs, and nothing fails silently.

Guideline for callers:
  * All exceptions raised deliberately by this package derive from BnTokError.
  * Programming errors (a wrong type, an impossible argument) raise a subclass
    immediately, with a message that says what was expected and what was given.
  * Recoverable, data-dependent problems (a bad line in a corpus) are counted and
    reported, not raised, unless they make the operation impossible.
"""

from __future__ import annotations


class BnTokError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(BnTokError):
    """An argument or configuration value is invalid."""


class NormalizationError(BnTokError):
    """Input could not be normalised (for example, not a string)."""


class EmptyCorpusError(BnTokError):
    """A corpus was empty or contained no usable text after cleaning."""


class VocabSizeError(ConfigError):
    """The requested vocabulary size is impossible for this corpus."""


class TrainingError(BnTokError):
    """The underlying subword trainer failed."""


class EncodeError(BnTokError):
    """Encoding failed for a specific input."""


class DecodeError(BnTokError):
    """Decoding failed for a specific token sequence."""


class LoadError(BnTokError):
    """A saved tokenizer directory is missing files or is corrupt."""


class ShapingError(BnTokError):
    """Shaping validation could not run or a check failed hard."""


class RoundTripError(BnTokError):
    """Encode then decode did not reproduce the normalised input."""


def require(condition: bool, message: str, cls: type[BnTokError] = ConfigError) -> None:
    """Raise `cls(message)` unless `condition` holds. A guard-clause helper."""
    if not condition:
        raise cls(message)
