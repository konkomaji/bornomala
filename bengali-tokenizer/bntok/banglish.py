r"""
The Banglish pipeline: tiers 0-2 of the tiered-cascade design (see
docs/known-issues.md's Banglish section for the full architecture and why
it is shaped this way - a frequency cascade, not "run a model on
everything", the same grammar/frequency-first-then-statistics philosophy
BMBT already uses for Bengali script itself, one layer up).

  Tier 0: Bengali-script text passes through untouched (already handled by
          the existing tokenizer, zero new cost).
  Tier 1: real-word lookup table (bntok/data/banglish-lookup.tsv, built by
          scripts/build_banglish_lookup.py from Dakshina + synthetic
          gap-fill). O(1) dict hit. Measured on an independent real
          held-out set (scripts/compare.py --register banglish): 79.5%
          coverage - see docs/known-issues.md.
  Tier 2: a from-scratch character-n-gram Naive Bayes classifier (no
          pretrained model, no fine-tuning) deciding English (leave
          unchanged - real code-mixed English inside Banglish sentences is
          common and must not be mangled) vs Banglish (unresolved, falls to
          tier 3) for whatever tier 1 misses.
  Tier 3: NOT YET BUILT. The trained seq2seq transliteration model (scoped
          separately, needs real training compute - see the Colab plan).
          Until it exists, tier-2-classified Banglish words with no tier-1
          hit pass through UNCHANGED, same as today's behaviour. This is an
          honest, documented gap, not a silent failure: `transliterate()`
          reports how many words hit each tier so a caller can see exactly
          how much of a given text this pipeline actually improved.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-zA-Z']+")
_DEFAULT_LOOKUP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "banglish-lookup.tsv"
)


def load_lookup_table(path: str = _DEFAULT_LOOKUP_PATH) -> dict[str, tuple[str, str]]:
    """Load the tier-1 table: latin surface form -> (bengali_word, source).

    `source` is "real" (Dakshina-attested) or "synthetic" (gap-fill from
    bntok.banglish_synth); callers that care about trust level can branch on
    it, `transliterate()` below does not distinguish for substitution
    purposes - both are used, since the synthetic-gap-fill validation
    (docs/known-issues.md) showed real hits and synthetic hits are both
    genuine table entries, just with different provenance. This loader
    drops the 5th (runner-up) column - use `load_lookup_table_full` for
    that; kept separate so the common case (just resolve a word) stays a
    simple 2-tuple.
    """
    return {latin: (bengali, source) for latin, (bengali, _cnt, source, _ru)
            in load_lookup_table_full(path).items()}


def load_lookup_table_full(path: str = _DEFAULT_LOOKUP_PATH) -> dict[str, tuple[str, int, str, str]]:
    """Load the tier-1 table with everything scripts/build_banglish_lookup.py
    writes: latin -> (bengali_word, count, source, runner_up).

    `runner_up` is `"bengali_word:count"` when the Latin spelling was
    attested for more than one real Bengali word (6.4% of real entries -
    see the table-building docstring for why most of these are harmless
    same-word orthographic variants, not genuine ambiguity), or "" when
    there was none. Not consumed by `transliterate()` yet - exposed so a
    future disambiguation pass has what it needs without rebuilding the
    table from source data again.
    """
    table: dict[str, tuple[str, int, str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            latin, bengali, count, source, runner_up = parts
            table[latin] = (bengali, int(count), source, runner_up)
    return table


# --- tier 2: from-scratch character n-gram Naive Bayes -----------------------

@dataclass
class NgramClassifier:
    """A character-bigram+trigram Naive Bayes classifier, trained from
    scratch on two real word lists (no pretrained model). Deliberately not
    a neural model: this tier exists specifically to be cheap (a handful of
    dict lookups and additions per word), since its whole job is triage
    before the expensive tier-3 model, not to be maximally accurate itself.
    """

    log_probs: dict[str, dict[str, float]]  # class -> ngram -> log P(ngram|class)
    log_prior: dict[str, float]
    vocab_size: int
    classes: tuple[str, ...] = ("banglish", "english")

    @staticmethod
    def _ngrams(word: str) -> list[str]:
        w = f"^{word}$"
        grams = [w[i:i + 2] for i in range(len(w) - 1)]
        grams += [w[i:i + 3] for i in range(len(w) - 2)]
        return grams

    @classmethod
    def train(cls, banglish_words: list[str], english_words: list[str]) -> NgramClassifier:
        counts: dict[str, Counter] = {"banglish": Counter(), "english": Counter()}
        totals: dict[str, list[str]] = {"banglish": banglish_words, "english": english_words}
        vocab: set[str] = set()
        for label, words in totals.items():
            for w in words:
                for g in cls._ngrams(w.lower()):
                    counts[label][g] += 1
                    vocab.add(g)
        vocab_size = len(vocab)
        log_probs: dict[str, dict[str, float]] = {}
        for label in ("banglish", "english"):
            total = sum(counts[label].values())
            denom = total + vocab_size  # Laplace (add-1) smoothing
            log_probs[label] = {g: math.log((c + 1) / denom) for g, c in counts[label].items()}
            log_probs[label]["__unseen__"] = math.log(1 / denom)
        # Balanced (uniform) priors, not proportional-to-training-set-size:
        # the real Dakshina word list and the English wordlist differ in
        # size by an artifact of how each was collected (125k vs 10k), not
        # by any known real-world ratio of English-to-Banglish words in
        # code-mixed chat text. A proportional prior pushed the decision
        # boundary hard toward "banglish" and was the main cause of real
        # English words being misclassified (checked directly: "a", "by",
        # "ad", "arizona", "bosnia" - short/ambiguous words - were the
        # majority of the errors before this fix).
        log_prior = {"banglish": math.log(0.5), "english": math.log(0.5)}
        return cls(log_probs=log_probs, log_prior=log_prior, vocab_size=vocab_size)

    def classify(self, word: str) -> str:
        grams = self._ngrams(word.lower())
        best_label, best_score = None, float("-inf")
        for label in self.classes:
            score = self.log_prior[label]
            table = self.log_probs[label]
            unseen = table["__unseen__"]
            for g in grams:
                score += table.get(g, unseen)
            if score > best_score:
                best_label, best_score = label, score
        return best_label

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "log_probs": self.log_probs, "log_prior": self.log_prior,
                "vocab_size": self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path: str) -> NgramClassifier:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(log_probs=data["log_probs"], log_prior=data["log_prior"], vocab_size=data["vocab_size"])


# --- the pipeline --------------------------------------------------------

@dataclass
class TransliterationResult:
    text: str
    tier1_hits: int
    tier2_english: int
    tier3_hits: int         # resolved by tier3_fn, if one was supplied
    tier3_unresolved: int   # classified Banglish, no tier-1 hit, no tier3_fn (or it declined): passed through unchanged
    total_latin_words: int
    cache_growth: int       # new entries tier3_fn's output added to `lookup` this call


def transliterate(text: str, lookup: dict[str, tuple[str, str]], classifier: NgramClassifier,
                   tier3_fn=None) -> TransliterationResult:
    """Run the full cascade over `text`. Bengali-script and non-alphabetic
    spans pass through untouched (tier 0). Each Latin word is tried against
    the tier-1 lookup table first (O(1)); on a miss, the tier-2 classifier
    decides whether to leave it alone (real English) or mark it unresolved
    Banglish, which then falls to `tier3_fn`.

    `tier3_fn`, an optional `str -> str | None` callable, is the pluggable
    slot for the actual seq2seq transliteration model (not built yet - see
    module docstring). When it resolves a word, that result is used AND
    written back into `lookup` in place: the self-growing cache. A novel
    word costs `tier3_fn` once; every later occurrence of the same spelling,
    in this call or a future one sharing the same `lookup` dict, is a tier-1
    hit from then on. `tier3_fn` returning None means "I can't resolve this
    either" - the word passes through unchanged, same as when no `tier3_fn`
    is supplied at all, and is still counted in `tier3_unresolved`.

    The cache-growth mechanism itself is tested (tests/test_banglish.py)
    against a stub `tier3_fn`, since no trained model exists yet to test it
    against for real; the mechanism does not depend on what tier3_fn's real
    implementation turns out to be.
    """
    tier1_hits = tier2_english = tier3_hits = tier3_unresolved = total = 0
    cache_growth = 0

    def repl(m: re.Match) -> str:
        nonlocal tier1_hits, tier2_english, tier3_hits, tier3_unresolved, total, cache_growth
        word = m.group(0)
        total += 1
        key = word.lower()
        entry = lookup.get(key)
        if entry is not None:
            tier1_hits += 1
            bengali, _source = entry
            return bengali
        label = classifier.classify(word)
        if label == "english":
            tier2_english += 1
            return word  # leave real English untouched
        if tier3_fn is not None:
            resolved = tier3_fn(word)
            if resolved:
                tier3_hits += 1
                lookup[key] = (resolved, "tier3")
                cache_growth += 1
                return resolved
        tier3_unresolved += 1
        return word  # Banglish, unresolved: pass through unchanged

    out = _WORD_RE.sub(repl, text)
    return TransliterationResult(
        text=out, tier1_hits=tier1_hits, tier2_english=tier2_english,
        tier3_hits=tier3_hits, tier3_unresolved=tier3_unresolved,
        total_latin_words=total, cache_growth=cache_growth,
    )
