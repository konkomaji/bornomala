"""
Reasoning probe - MEASURED (not derived) capability check across languages.

`mti.capability` *derives* a capability-risk signal from tokenization alone
(effective context loss). That is a prediction, not a measurement. This script
is the measurement: it runs an actual model on the same reasoning items in
English and in a target language, and reports the accuracy gap.

It is deliberately kept OUT of the core package because it needs a model or an
API key and (optionally) a GPU - the Bornomala plan runs this class of work on a
separate rented machine, not on the local CPU box (spec §15). The core tool
stays pure-CPU and dependency-light; this is opt-in.

Design:
  * A small set of language-neutral reasoning items (arithmetic word problems,
    simple logic) is provided in English. You supply faithful translations into
    the target language (or plug in a translation step).
  * The probe queries a model via a pluggable `answer_fn(prompt) -> str` - wire
    it to the OpenAI/Anthropic/HF endpoint of your choice.
  * It scores exact-match accuracy per language and reports:
        accuracy(EN), accuracy(target), gap, and mean tokens/item per language
    so the accuracy gap can be read alongside the tokenization overhead that
    `mti` predicts.

Nothing here fabricates scores. If you do not wire an `answer_fn`, it runs in
dry mode and only reports the tokenization overhead, clearly labelled.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

from mti.analyze import analyze_one

# Minimal seed items. Extend with a real benchmark (e.g. translated GSM8K subset,
# MGSM, or native-authored items - native authoring is preferred; see spec §13.5).
SEED_ITEMS = [
    {"id": "arith1", "question_en": "A shop sells pens at 7 rupees each. How much do 6 pens cost? Answer with a number only.", "answer": "42"},
    {"id": "arith2", "question_en": "There are 3 baskets with 8 apples each. How many apples in total? Answer with a number only.", "answer": "24"},
    {"id": "logic1", "question_en": "If all cats are animals and Tom is a cat, is Tom an animal? Answer yes or no.", "answer": "yes"},
]


@dataclass
class LangScore:
    language: str
    model_id: str
    n_items: int
    accuracy: float | None       # None in dry mode (no answer_fn)
    mean_tokens_per_item: float
    measured: bool


def _mean_tokens(items_text: list[str], model_id: str) -> float:
    if not items_text:
        return 0.0
    total = 0
    for t in items_text:
        r = analyze_one(t, model_id, want_tokens=False, anchor_english=False)
        if r.available and r.metrics:
            total += r.metrics.n_tokens
    return total / len(items_text)


def run(
    items: list[dict],
    language: str,
    model_id: str,
    question_key: str = "question_en",
    answer_fn: Callable[[str], str] | None = None,
) -> LangScore:
    texts = [it[question_key] for it in items]
    mean_tok = _mean_tokens(texts, model_id)

    if answer_fn is None:
        return LangScore(language, model_id, len(items), None, mean_tok, measured=False)

    correct = 0
    for it in items:
        got = answer_fn(it[question_key]).strip().lower()
        want = str(it["answer"]).strip().lower()
        if want in got:
            correct += 1
    return LangScore(language, model_id, len(items), correct / len(items), mean_tok, measured=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measured cross-language reasoning probe.")
    p.add_argument("--model", default="gpt-4o", help="model id from mti registry (for token accounting)")
    p.add_argument("--items", help="JSON file of items; defaults to the built-in seed set")
    p.add_argument("--language", default="English", help="label for the language being probed")
    p.add_argument("--question-key", default="question_en", help="item field holding the prompt")
    args = p.parse_args(argv)

    items = SEED_ITEMS
    if args.items:
        with open(args.items, encoding="utf-8") as f:
            items = json.load(f)

    # No answer_fn wired here - dry mode. Wire one in code for a measured run.
    score = run(items, args.language, args.model, question_key=args.question_key, answer_fn=None)
    print(json.dumps(score.__dict__, ensure_ascii=False, indent=2))
    if not score.measured:
        print("\n[dry mode] No answer_fn wired - reported tokens only, no accuracy. "
              "Wire an OpenAI/Anthropic/HF endpoint to measure the reasoning gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
