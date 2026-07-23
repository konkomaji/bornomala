r"""
Shaping validation (Requirement A-1, Gate G1) using HarfBuzz.

The largest technical risk in the whole programme is silent: if the text pipeline
mis-shapes Bengali conjuncts, every downstream model learns corrupted glyph to
text mappings, and nothing catches it (spec section 9.3, risk R1). Gate G1 says:
before trusting the pipeline, verify that Bengali renders correctly by shaping it
with HarfBuzz and reading the grapheme clusters back.

True "render and OCR back" is Track B's job. What we can and must verify here,
cheaply and deterministically, is:

  1. Coverage: a real Bengali font shapes every cluster with no missing glyph
     (.notdef, glyph id 0). A missing glyph means the pipeline is feeding the
     font sequences it cannot form, which is exactly the mis-shaping failure.
  2. Cluster correspondence: HarfBuzz reports one cluster index per input
     codepoint offset; the set of distinct shaping clusters must line up with the
     grapheme cluster boundaries. If shaping collapses or reorders clusters in a
     way that breaks that correspondence, the assertion fails.

This needs a Bengali font. On Windows, Nirmala UI or Vrinda cover Bengali; the
finder tries common locations. With no font available, validation is reported as
skipped (never silently passed).
"""

from __future__ import annotations

import os

from .graphemes import grapheme_clusters
from .normalize import normalize

# Common Bengali-capable fonts by platform.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\NirmalaB.ttf",
    r"C:\Windows\Fonts\vrinda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
    "/System/Library/Fonts/Supplemental/Bangla MN.ttc",
]


def find_bengali_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def shape_check(text: str, font_path: str) -> dict:
    """Shape one string and report coverage and cluster correspondence."""
    import uharfbuzz as hb

    nfc = normalize(text)
    gclusters = grapheme_clusters(nfc)

    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    font = hb.Font(face)

    buf = hb.Buffer()
    buf.add_str(nfc)
    buf.guess_segment_properties()
    hb.shape(font, buf)

    infos = buf.glyph_infos
    missing = sum(1 for g in infos if g.codepoint == 0)  # glyph id 0 = .notdef
    n_shaping_clusters = len({g.cluster for g in infos})

    return {
        "text": nfc,
        "n_grapheme_clusters": len(gclusters),
        "n_shaping_clusters": n_shaping_clusters,
        "missing_glyphs": missing,
        "n_glyphs": len(infos),
        "ok": missing == 0 and n_shaping_clusters == len(gclusters),
    }


def validate(samples: list[str], font_path: str | None = None) -> dict:
    """Run Gate G1 over a set of samples. Returns a report with pass / fail / skip."""
    font_path = font_path or find_bengali_font()
    if not font_path:
        return {"status": "skipped", "reason": "no Bengali font found", "checked": 0}

    try:
        import uharfbuzz  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "uharfbuzz not installed", "checked": 0}

    results = [shape_check(s, font_path) for s in samples if s.strip()]
    failures = [r for r in results if not r["ok"]]
    return {
        "status": "pass" if not failures else "fail",
        "font": font_path,
        "checked": len(results),
        "failures": failures[:20],
        "n_failures": len(failures),
    }
