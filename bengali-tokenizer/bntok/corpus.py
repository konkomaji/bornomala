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
import itertools
import os
import re

from .errors import ConfigError, EmptyCorpusError


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


_CC100_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "bntok", "cc100")


def _download_cc100_shard(repo_path: str) -> str:
    """Download one CC-100 parquet shard by plain HTTP streaming, cached locally.

    `huggingface_hub.hf_hub_download` hangs indefinitely on this specific
    repository (verified directly: a plain HTTP GET to the exact same
    resolved URL succeeds immediately and streams at normal speed, while
    `hf_hub_download` on the same file, same revision, does not return even
    after many minutes, with or without the Xet transfer backend). This is
    not diagnosed further than that; it is bypassed here, for CC-100 only -
    every other stream_* function in this module still uses
    `hf_hub_download` successfully and is unaffected. See
    docs/known-issues.md for the full account.
    """
    try:
        import requests
    except ImportError as e:
        raise ConfigError(
            "streaming CC-100 needs the requests library: pip install requests"
        ) from e
    os.makedirs(_CC100_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(_CC100_CACHE_DIR, repo_path.replace("/", "_"))
    if os.path.exists(local_path):
        return local_path
    url = f"https://huggingface.co/datasets/cc100/resolve/refs%2Fconvert%2Fparquet/{repo_path}"
    tmp_path = local_path + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.writelines(r.iter_content(chunk_size=1 << 20))
        os.replace(tmp_path, local_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise ConfigError(f"could not download CC-100 shard {repo_path}: {e}") from e
    return local_path


def stream_cc100(
    lang: str = "bn", limit_lines: int = 300_000, offset_lines: int = 0,
) -> list[str]:
    """Stream CC-100 (CommonCrawl-derived web text) for one language.

    Optional: requires `pandas`, `pyarrow`, `huggingface_hub` (for listing
    shards), and `requests` (for downloading them; see
    `_download_cc100_shard`). CC-100 (Wenzek et al., 2020; the corpus behind
    XLM-R's training data) is a large, general-web Bengali source, orders of
    magnitude bigger than any single source currently in
    `build_configured_corpus`, but it is 2018-vintage CommonCrawl text:
    noisier and not literary-weighted, so it is meant as a bulk general-web
    supplement, not a replacement for the literary-weighted mix. Each row is
    already one line/paragraph (CC-100's own format separates documents by a
    blank line and paragraphs by a single newline, and the HF parquet mirror
    stores one row per paragraph), so `offset_lines` skips that many rows
    before collecting, letting a disjoint held-out slice be drawn the same
    way as the other sources here. Reads shards in order, only as many as
    needed to satisfy `offset_lines + limit_lines`.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ConfigError(
            "streaming CC-100 needs pandas: pip install pandas pyarrow"
        ) from e
    if limit_lines < 1:
        raise ConfigError(f"limit_lines must be >= 1, got {limit_lines}")
    if offset_lines < 0:
        raise ConfigError(f"offset_lines must be >= 0, got {offset_lines}")

    files = _list_hf_parquet_files("cc100", f"{lang}/train/", revision="refs/convert/parquet")
    out: list[str] = []
    seen = 0
    for repo_path in files:
        local = _download_cc100_shard(repo_path)
        df = pd.read_parquet(local, columns=["text"])
        for text in df["text"]:
            if not isinstance(text, str) or not text.strip():
                continue
            seen += 1
            if seen <= offset_lines:
                continue
            out.append(text.strip())
            if seen - offset_lines >= limit_lines:
                break
        if seen - offset_lines >= limit_lines:
            break
    if not out:
        raise EmptyCorpusError(f"CC-100 stream for '{lang}' yielded no text")
    return out


def _split_lines(text: str) -> list[str]:
    """Split one document into lines, falling back to sentence splits.

    Most documents carry paragraph breaks (`\\n`); a few (news summaries) are a
    single block, so those are split on the Bengali sentence terminator "।"
    instead. A document with neither is kept whole rather than dropped.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines
    parts = [p.strip() for p in re.split(r"(?<=৷)\s+|(?<=।)\s+|(?<=॥)\s+", text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


_BENGALI_OR_ASCII = re.compile(r"[ঀ-৿ -~]")


def _is_clean_bengali_line(line: str, min_ratio: float = 0.75, min_len: int = 4) -> bool:
    """Heuristic OCR-garbage filter for scanned/PDF-sourced text.

    Digitised pre-modern Bengali books carry real OCR noise: misrecognised
    glyphs from other scripts, stray control characters, page-number/header
    debris. Rather than trying to detect specific failure modes, this rejects
    lines where too small a share of characters are Bengali or plain ASCII
    (which covers digits, Latin loanwords, and punctuation). It is a coarse
    filter, not a correctness guarantee: it reduces noise, it does not remove
    all of it, and that limitation is documented, not hidden.
    """
    if len(line) < min_len:
        return False
    matches = len(_BENGALI_OR_ASCII.findall(line))
    return (matches / len(line)) >= min_ratio


def _list_hf_parquet_files(repo_id: str, prefix: str, revision: str | None = None) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ConfigError(
            "streaming this source needs huggingface_hub: pip install huggingface_hub"
        ) from e
    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        raise EmptyCorpusError(f"no parquet files under '{prefix}' in {repo_id}")
    return files


def stream_sangraha(
    lang: str = "ben", doc_type: str | None = None, limit_docs: int = 20000, max_files: int = 4,
    offset_docs: int = 0, clean: bool = False, max_files_scan: int | None = None,
) -> list[str]:
    """Stream AI4Bharat Sangraha 'verified' documents for one language.

    Optional: requires `pandas`, `pyarrow`, and `huggingface_hub`. `doc_type`
    filters the source-provenance column ("web", "pdf", "speech"); pdf-typed
    documents are the closest available proxy for formal/book register, since
    Sangraha does not label pre-1950 public-domain literature separately, and
    tend to carry real OCR noise (`clean=True` applies a coarse Bengali/ASCII
    character-ratio filter, see `_is_clean_bengali_line`). Downloads only as
    many parquet shards (`max_files`, scanning up to `max_files_scan` if a
    `doc_type` filter makes shards sparse) as needed to reach `limit_docs`.
    `offset_docs` skips that many matching documents first, so a disjoint
    held-out slice can be drawn from documents training never saw.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ConfigError(
            "streaming Sangraha needs pandas and huggingface_hub: "
            "pip install pandas huggingface_hub"
        ) from e
    if limit_docs < 1:
        raise ConfigError(f"limit_docs must be >= 1, got {limit_docs}")
    if offset_docs < 0:
        raise ConfigError(f"offset_docs must be >= 0, got {offset_docs}")

    files = _list_hf_parquet_files("ai4bharat/sangraha", f"verified/{lang}/")
    scan_limit = max_files_scan if max_files_scan is not None else max_files
    out: list[str] = []
    docs_matched = 0
    for path in files[:scan_limit]:
        local = hf_hub_download(repo_id="ai4bharat/sangraha", filename=path, repo_type="dataset")
        df = pd.read_parquet(local, columns=["text", "type"])
        if doc_type is not None:
            df = df[df["type"] == doc_type]
        for text in df["text"]:
            if not isinstance(text, str) or not text.strip():
                continue
            docs_matched += 1
            if docs_matched <= offset_docs:
                continue
            lines = _split_lines(text)
            if clean:
                lines = [ln for ln in lines if _is_clean_bengali_line(ln)]
            out.extend(lines)
            if docs_matched - offset_docs >= limit_docs:
                break
        if docs_matched - offset_docs >= limit_docs:
            break
    if not out:
        raise EmptyCorpusError(f"Sangraha stream for '{lang}'/{doc_type} yielded no text")
    return out


def stream_wikisource(lang: str = "bn", limit_docs: int | None = None) -> list[str]:
    """Stream Wikimedia Wikisource documents for one language (public-domain texts).

    Optional: requires `pandas`, `pyarrow`, and `huggingface_hub`. Bengali
    Wikisource is small (a few dozen proofread pages as of this dataset
    snapshot), so this is a genuine but thin source of pre-modern public-domain
    Bengali text; it is meant to be combined with other formal-register sources.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ConfigError(
            "streaming Wikisource needs pandas and huggingface_hub: "
            "pip install pandas huggingface_hub"
        ) from e
    files = _list_hf_parquet_files("wikimedia/wikisource", f"20231201.{lang}/")
    out: list[str] = []
    docs_seen = 0
    for path in files:
        local = hf_hub_download(repo_id="wikimedia/wikisource", filename=path, repo_type="dataset")
        df = pd.read_parquet(local, columns=["text"])
        for text in df["text"]:
            if not isinstance(text, str) or not text.strip():
                continue
            out.extend(_split_lines(text))
            docs_seen += 1
            if limit_docs is not None and docs_seen >= limit_docs:
                break
        if limit_docs is not None and docs_seen >= limit_docs:
            break
    if not out:
        raise EmptyCorpusError(f"Wikisource stream for '{lang}' yielded no text")
    return out


def stream_xlsum(lang: str = "bengali", limit_docs: int = 20000, offset_docs: int = 0) -> list[str]:
    """Stream XL-Sum news articles for one language (contemporary news register).

    Optional: requires `pandas`, `pyarrow`, and `huggingface_hub`. Uses the
    Hub's auto-converted parquet mirror of csebuetnlp/xlsum so no dataset
    loading script is needed. Combines the train/validation/test splits since
    XL-Sum's Bengali split is small (about 8k articles: pass `limit_docs` below
    the true total and use `offset_docs` to reserve a disjoint held-out slice,
    since the whole split would otherwise be consumed by training.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ConfigError(
            "streaming XL-Sum needs pandas and huggingface_hub: "
            "pip install pandas huggingface_hub"
        ) from e
    if limit_docs < 1:
        raise ConfigError(f"limit_docs must be >= 1, got {limit_docs}")
    if offset_docs < 0:
        raise ConfigError(f"offset_docs must be >= 0, got {offset_docs}")

    out: list[str] = []
    docs_matched = 0
    for split in ("train", "validation", "test"):
        try:
            local = hf_hub_download(
                repo_id="csebuetnlp/xlsum", filename=f"{lang}/{split}/0000.parquet",
                repo_type="dataset", revision="refs/convert/parquet",
            )
        except Exception as e:
            raise ConfigError(f"could not fetch XL-Sum split '{split}' for '{lang}': {e}") from e
        df = pd.read_parquet(local, columns=["text"])
        for text in df["text"]:
            if not isinstance(text, str) or not text.strip():
                continue
            docs_matched += 1
            if docs_matched <= offset_docs:
                continue
            out.extend(_split_lines(text))
            if docs_matched - offset_docs >= limit_docs:
                break
        if docs_matched - offset_docs >= limit_docs:
            break
    if not out:
        raise EmptyCorpusError(f"XL-Sum stream for '{lang}' yielded no text")
    return out


def weighted_corpus(
    loaders: dict[str, list[str]], weights: dict[str, float], total_lines: int,
) -> list[str]:
    """Combine named, already-loaded sources into one corpus of `total_lines`.

    Weights are relative and need not sum to 1 (they are normalised here); a
    source short of its target share is cycled from the start to fill it, so
    the contribution stays honestly proportional even when a source is small
    (e.g. Wikisource). This is a sampling scheme, not deduplication: a thin
    source will legitimately repeat.
    """
    if not weights:
        raise ConfigError("no weights given")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ConfigError("weights must sum to a positive number")
    if total_lines < 1:
        raise ConfigError(f"total_lines must be >= 1, got {total_lines}")

    combined: list[str] = []
    for name, weight in weights.items():
        if name not in loaders or not loaders[name]:
            raise ConfigError(f"source '{name}' has a weight but no loaded text")
        target = max(1, round(total_lines * weight / total_weight))
        combined.extend(itertools.islice(itertools.cycle(loaders[name]), target))
    if not combined:
        raise EmptyCorpusError("weighted corpus is empty")
    return combined



# Doc/article budgets shared between build_configured_corpus (training) and
# build_register_held_out (evaluation), so the two never draw from the same
# document: held-out reads start exactly where training's budget ends.
WIKIPEDIA_TRAIN_ARTICLES = 15_000
SANGRAHA_PDF_TRAIN_DOCS = 50_000
SANGRAHA_WEB_TRAIN_DOCS = 200_000
XLSUM_TRAIN_DOCS = 8_000  # XL-Sum bn is ~10.1k docs total; this reserves ~2k for held-out
CC100_TRAIN_LINES = 300_000  # bulk general-web supplement; not in the default corpus_sources weights yet


def build_configured_corpus(config: dict, log=lambda msg: None) -> list[str]:
    """Build the weighted induction corpus described by a training config dict.

    Reads `corpus_sources` (name -> weight) and `total_lines`, and maps each
    known source name to a real, publicly available loader:

      * public_domain_literature: Wikisource bn (genuine but thin public-domain
        text) combined with Sangraha verified/ben pdf-typed documents (a proxy
        for formal/book register, since no clean pre-1950-labelled corpus is
        available).
      * sangraha_verified_bn: Sangraha verified/ben web-typed documents.
      * bengali_wikipedia: Bengali Wikipedia (via `stream_wikipedia`).
      * contemporary_news: XL-Sum Bengali news articles.
      * cc100_general_web: CC-100 Bengali (via `stream_cc100`), a large
        2018-vintage CommonCrawl web-text supplement. Available as a source
        name but not part of the shipped `configs/bpe-64k.json` weights: it
        has not yet been retrained-and-benchmarked in, so including it here
        does not change the current artifact until a config opts in.

    `government_administrative` and `code_mixed_bn_en` from the original spec
    have no clean public dataset and are silently skipped if present in the
    config; weights are normalised over whatever sources remain, so their
    omission does not need manual rebalancing.
    """
    sources = config.get("corpus_sources")
    if not sources:
        raise ConfigError("config has no 'corpus_sources'")
    total_lines = config.get("total_lines", 1_500_000)

    known = {
        "public_domain_literature", "sangraha_verified_bn", "bengali_wikipedia",
        "contemporary_news", "cc100_general_web",
    }
    unavailable = {"government_administrative", "code_mixed_bn_en"}
    weights = {k: v for k, v in sources.items() if k in known}
    skipped = {k: v for k, v in sources.items() if k in unavailable}
    unknown = set(sources) - known - unavailable
    if unknown:
        raise ConfigError(f"unknown corpus source(s) in config: {sorted(unknown)}")
    if skipped:
        log(f"skipping source(s) with no public dataset available: {sorted(skipped)}")
    if not weights:
        raise ConfigError("no usable corpus sources left after dropping unavailable ones")

    loaders: dict[str, list[str]] = {}
    if "public_domain_literature" in weights:
        log("loading public_domain_literature: Wikisource bn + Sangraha pdf-typed (OCR-filtered) ...")
        lit = stream_wikisource("bn")
        lit += stream_sangraha("ben", doc_type="pdf", limit_docs=SANGRAHA_PDF_TRAIN_DOCS, max_files=4, clean=True)
        loaders["public_domain_literature"] = lit
        log(f"  {len(lit)} lines")
    if "sangraha_verified_bn" in weights:
        log("loading sangraha_verified_bn: Sangraha web-typed ...")
        loaders["sangraha_verified_bn"] = stream_sangraha(
            "ben", doc_type="web", limit_docs=SANGRAHA_WEB_TRAIN_DOCS, max_files=4)
        log(f"  {len(loaders['sangraha_verified_bn'])} lines")
    if "bengali_wikipedia" in weights:
        log("loading bengali_wikipedia ...")
        loaders["bengali_wikipedia"] = stream_wikipedia("bn", limit=WIKIPEDIA_TRAIN_ARTICLES)
        log(f"  {len(loaders['bengali_wikipedia'])} lines")
    if "contemporary_news" in weights:
        log("loading contemporary_news: XL-Sum bn (holding out the tail for eval) ...")
        loaders["contemporary_news"] = stream_xlsum("bengali", limit_docs=XLSUM_TRAIN_DOCS)
        log(f"  {len(loaders['contemporary_news'])} lines")
    if "cc100_general_web" in weights:
        log("loading cc100_general_web: CC-100 bn (holding out the tail for eval) ...")
        loaders["cc100_general_web"] = stream_cc100("bn", limit_lines=CC100_TRAIN_LINES)
        log(f"  {len(loaders['cc100_general_web'])} lines")

    log(f"combining into {total_lines} weighted lines ...")
    return weighted_corpus(loaders, weights, total_lines)


def build_register_held_out(limit_docs: int = 1000, log=lambda msg: None) -> dict[str, list[str]]:
    """Build small held-out slices per non-Wikipedia register, disjoint from training.

    Wikipedia held-out is handled separately (scripts/compare.py's own
    `--skip`, kept consistent with the v0.1 benchmark's methodology). This
    covers the sources that `build_configured_corpus` would otherwise consume
    entirely: Sangraha pdf (literary/formal proxy, OCR-filtered), Sangraha web
    (general), and XL-Sum (news). Wikisource is not included: its ~65 lines
    are entirely used in training and too small to split meaningfully.
    """
    out = {}
    log("held-out: sangraha pdf-typed (literary/formal proxy) ...")
    out["literary_formal"] = stream_sangraha(
        "ben", doc_type="pdf", limit_docs=limit_docs, max_files=4,
        offset_docs=SANGRAHA_PDF_TRAIN_DOCS, clean=True,
    )
    log("held-out: sangraha web-typed (general) ...")
    out["general_web"] = stream_sangraha(
        "ben", doc_type="web", limit_docs=limit_docs, max_files=4,
        offset_docs=SANGRAHA_WEB_TRAIN_DOCS,
    )
    log("held-out: XL-Sum (news) ...")
    out["news"] = stream_xlsum("bengali", limit_docs=limit_docs, offset_docs=XLSUM_TRAIN_DOCS)
    for name, lines in out.items():
        log(f"  {name}: {len(lines)} lines")
    return out
