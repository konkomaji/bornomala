r"""
Corpus loading for tokenizer induction (Track A).

The induction corpus is deliberately over-weighted toward literary and formal
Bengali, because a vocabulary induced on web text starves the tatsama stratum
where literary and formal Bengali actually lives (spec section 9.2 step 1). This
module loads text from local files or directories, and optionally streams Bengali
Wikipedia, and lets a config assign a sampling weight per source.

Everything here is defensive: unreadable files are skipped with a count,
decoding uses a permissive error policy, and an empty result raises a clear
EmptyCorpusError rather than letting training fail deep in the trainer.
"""

from __future__ import annotations

import glob
import os

from .errors import EmptyCorpusError, ConfigError


def load_file(path: str) -> list[str]:
    """Load one text file as a list of non-empty lines. Permissive decoding."""
    if not os.path.exists(path):
        raise ConfigError(f"corpus file not found: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def load_paths(paths: list[str]) -> list[str]:
    """Load many files or glob patterns into one list of lines.

    Skips files that cannot be read, and reports how many were skipped via the
    returned list being shorter; raises only if nothing at all was loaded.
    """
    lines: list[str] = []
    skipped = 0
    expanded: list[str] = []
    for p in paths:
        matches = glob.glob(p)
        expanded.extend(matches if matches else [p])
    for p in expanded:
        if not os.path.isfile(p):
            skipped += 1
            continue
        try:
            lines.extend(load_file(p))
        except OSError:
            skipped += 1
    if not lines:
        raise EmptyCorpusError(f"no readable text found in {len(expanded)} path(s)")
    return lines


def load_dir(directory: str, pattern: str = "**/*.txt") -> list[str]:
    """Recursively load text files from a directory."""
    if not os.path.isdir(directory):
        raise ConfigError(f"not a directory: {directory}")
    return load_paths([os.path.join(directory, pattern)])


def stream_wikipedia(lang: str = "bn", limit: int = 5000) -> list[str]:
    """Stream Bengali Wikipedia article text via the `datasets` library.

    Optional: requires `pip install datasets`. Returns up to `limit` articles as
    lines. Raises ConfigError with a clear hint if `datasets` is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ConfigError(
            "streaming Wikipedia needs the datasets library: pip install datasets"
        ) from e
    if limit < 1:
        raise ConfigError(f"limit must be >= 1, got {limit}")

    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    out: list[str] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        text = row.get("text", "")
        if text and text.strip():
            out.extend(p for p in text.split("\n") if p.strip())
    if not out:
        raise EmptyCorpusError(f"Wikipedia stream for '{lang}' yielded no text")
    return out


def weighted_corpus(sources: dict[str, float], loaders: dict[str, list[str]]) -> list[str]:
    """Combine named sources with integer-ish sampling weights.

    `sources` maps a source name to a weight; `loaders` maps the same names to the
    already-loaded lines. A source with weight w contributes its lines repeated
    round(w) times. Weights are relative; this is a simple, transparent scheme.
    """
    if not sources:
        raise ConfigError("no sources given")
    combined: list[str] = []
    for name, weight in sources.items():
        if name not in loaders:
            raise ConfigError(f"source '{name}' has a weight but no loaded text")
        reps = max(1, round(weight))
        for _ in range(reps):
            combined.extend(loaders[name])
    if not combined:
        raise EmptyCorpusError("weighted corpus is empty")
    return combined
