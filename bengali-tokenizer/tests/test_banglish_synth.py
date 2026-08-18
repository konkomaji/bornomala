"""Tests for bntok.banglish_synth's reverse phonetic table.

Regression coverage for the ba-glide fix: real Dakshina spellings render a
chained (non-initial) conjunct ব as the labial glide "w", not the standalone
"b" sound - matching the already-established pattern for chained য
(ya-phala). Found by scripts/validate_banglish_synth.py against real
dev.tsv data (হ্বামীজীর/বিশ্বাস cases), see docs/known-issues.md.
"""

from __future__ import annotations

import random

from bntok.banglish_synth import render_word_latin


def _renders(word: str, seeds: int = 20) -> set[str]:
    return {render_word_latin(word, random.Random(s)) for s in range(seeds)}


class TestBaGlide:
    def test_chained_ba_can_render_as_w(self):
        # স্ব (স + virama + ব): ব is the second, chained onset consonant -
        # "w" (real: "swamijir") is now a reachable variant, matching
        # _BA_GLIDE_LATIN's weighting ("w" first/dominant, "b" kept as a
        # real but rarer tail variant, same precedent as _YA_GLIDE_LATIN).
        renders = _renders("স্বামীজীর")
        assert any(r.startswith("sw") for r in renders), renders

    def test_standalone_ba_still_renders_as_b(self):
        # বই ("book"): ব is word-initial, standalone - keeps its own "b"
        # sound, unaffected by the glide special-case (i > 0 only).
        renders = _renders("বই")
        assert all(r.startswith("b") for r in renders), renders
        assert not any(r.startswith("w") for r in renders), renders
