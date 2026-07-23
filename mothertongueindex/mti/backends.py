"""
Tokenizer backends.

A backend turns text into a token list for one family of models. Three exist:

  TiktokenBackend  - OpenAI GPT models via the public `tiktoken` BPE ranks.
                     Exact. No auth, small download, cached locally.

  HFBackend        - any Hugging Face `tokenizers`/`transformers` tokenizer:
                     Llama, Gemma, Qwen, Mistral, DeepSeek, Sarvam, XLM-R,
                     mBERT, BLOOM, IndicSuperTokenizer, etc. Exact. Some repos
                     are gated (Llama, Gemma) and need HF_TOKEN; ungated ones
                     work out of the box.

  EstimateBackend  - models with no public tokenizer (notably Claude). Produces
                     a clearly-labelled *estimate* from a bytes-per-token ratio,
                     never presented as measured (spec rule E4).

Every backend is loaded lazily and fails soft: a tokenizer that cannot be
fetched is reported as unavailable, it does not crash the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

# Quieten Hugging Face hub noise (symlink + unauthenticated-request warnings)
# that otherwise floods multilingual output on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class BackendError(RuntimeError):
    """Raised when a tokenizer cannot be loaded or run."""


@dataclass
class Encoding:
    """Result of encoding one text with one tokenizer."""

    tokens: list[str]          # human-readable token surface strings (best effort)
    n_tokens: int              # authoritative token count
    single_token_words: int    # words encoded as exactly 1 token (for STRR)
    estimated: bool = False
    note: str = ""


# --------------------------------------------------------------------------
# tiktoken (OpenAI)
# --------------------------------------------------------------------------

class TiktokenBackend:
    _cache: ClassVar[dict[str, object]] = {}

    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name

    def _enc(self):
        if self.encoding_name not in self._cache:
            try:
                import tiktoken
            except ImportError as e:  # pragma: no cover
                raise BackendError("tiktoken not installed (pip install tiktoken)") from e
            try:
                self._cache[self.encoding_name] = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                raise BackendError(f"could not load tiktoken '{self.encoding_name}': {e}") from e
        return self._cache[self.encoding_name]

    def encode(self, text: str) -> Encoding:
        from .segment import words as _words

        enc = self._enc()
        ids = enc.encode(text, disallowed_special=())
        # Surface strings: decode each id independently (may show byte artefacts).
        tokens = []
        for i in ids:
            try:
                tokens.append(enc.decode([i]))
            except Exception:  # noqa: BLE001 - a single id may not decode standalone; show a placeholder
                tokens.append("�")

        stw = sum(1 for w in _words(text) if len(enc.encode(w, disallowed_special=())) == 1)
        return Encoding(tokens=tokens, n_tokens=len(ids), single_token_words=stw)


# --------------------------------------------------------------------------
# Hugging Face tokenizers
# --------------------------------------------------------------------------

class HFBackend:
    _cache: ClassVar[dict[str, object]] = {}

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def _tok(self):
        if self.repo_id not in self._cache:
            try:
                from tokenizers import Tokenizer
            except ImportError as e:  # pragma: no cover
                raise BackendError("tokenizers not installed (pip install tokenizers)") from e
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            try:
                # Fast path: pull tokenizer.json directly from the hub.
                self._cache[self.repo_id] = Tokenizer.from_pretrained(self.repo_id, token=token)
            except Exception:  # noqa: BLE001 - fast path can fail for many reasons; fall back below
                # Fallback: transformers can assemble tokenizers that lack a
                # single tokenizer.json (sentencepiece-only repos).
                try:
                    from transformers import AutoTokenizer
                    at = AutoTokenizer.from_pretrained(self.repo_id, token=token, use_fast=True)
                    self._cache[self.repo_id] = ("transformers", at)
                except Exception as e:
                    hint = "" if token else " (set HF_TOKEN for gated repos)"
                    raise BackendError(f"could not load HF tokenizer '{self.repo_id}': {e}{hint}") from e
        return self._cache[self.repo_id]

    def _encode_ids_and_tokens(self, text: str):
        tok = self._tok()
        if isinstance(tok, tuple) and tok[0] == "transformers":
            at = tok[1]
            ids = at.encode(text, add_special_tokens=False)
            toks = at.convert_ids_to_tokens(ids)
            return ids, [self._clean(t) for t in toks], ("tf", at)
        out = tok.encode(text, add_special_tokens=False)
        return out.ids, [self._clean(t) for t in out.tokens], ("fast", tok)

    @staticmethod
    def _clean(t: str) -> str:
        # Normalise the common subword continuation markers to a readable form.
        return t.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")

    def encode(self, text: str) -> Encoding:
        from .segment import words as _words

        ids, tokens, handle = self._encode_ids_and_tokens(text)

        kind, obj = handle
        if kind == "tf":
            def wc(w: str) -> int:
                return len(obj.encode(w, add_special_tokens=False))
        else:
            def wc(w: str) -> int:
                return len(obj.encode(w, add_special_tokens=False).ids)

        stw = sum(1 for w in _words(text) if wc(w) == 1)
        return Encoding(tokens=tokens, n_tokens=len(ids), single_token_words=stw)


# --------------------------------------------------------------------------
# Estimate (no public tokenizer)
# --------------------------------------------------------------------------

class EstimateBackend:
    """Heuristic token-count estimate for models with no public tokenizer.

    Strategy: estimate tokens per *script* from a bytes-per-token prior. Claude's
    tokenizer is not public; Anthropic's own guidance is roughly ~3.5–4 chars per
    token on English and materially worse on Indic scripts. We approximate by
    scoring each grapheme cluster with a per-script byte cost, then dividing total
    bytes by an effective bytes-per-token constant. It is an *estimate* and is
    always labelled as such.
    """

    # Effective bytes-per-token priors by script (rough, English-anchored).
    # Base table reflects a large modern vocabulary (Claude-class ~cl100k-like).
    _BPT: ClassVar[dict[str, float]] = {
        "Latin": 4.0, "Number": 3.0, "Other": 3.0,
        "Cyrillic": 2.2, "Greek": 2.4, "Arabic": 2.2, "Hebrew": 2.4,
        "Han": 1.6, "Hiragana": 1.8, "Katakana": 1.8, "Hangul": 2.0,
        # Indic scripts fragment badly on non-native tokenizers.
        "Bengali": 1.5, "Devanagari": 1.6, "Tamil": 1.4, "Telugu": 1.4,
        "Kannada": 1.4, "Malayalam": 1.3, "Gujarati": 1.6, "Gurmukhi": 1.6,
        "Oriya": 1.4, "Thai": 1.8,
    }

    # Per-model multipliers on the base priors, reflecting known vocabulary size
    # differences. Gemini/Gemma use a very large (256k) multilingual vocabulary,
    # so it packs non-Latin scripts better than a cl100k-class tokenizer.
    _MODEL_SCALE: ClassVar[dict[str, float]] = {
        "claude": 1.0,
        "gemini": 1.35,   # large multilingual vocab -> fewer tokens on non-Latin
        "grok":   1.0,
    }

    def __init__(self, label: str = "estimate"):
        self.label = label
        self._scale = self._MODEL_SCALE.get(label, 1.0)

    def encode(self, text: str) -> Encoding:
        from .segment import grapheme_clusters, script_of

        est = 0.0
        for g in grapheme_clusters(text):
            if g.isspace():
                continue
            b = len(g.encode("utf-8"))
            bpt = self._BPT.get(script_of(g), 3.0) * self._scale
            est += b / bpt
        n = max(1, round(est))
        return Encoding(
            tokens=[],
            n_tokens=n,
            single_token_words=0,
            estimated=True,
            note="heuristic estimate - no public tokenizer",
        )
