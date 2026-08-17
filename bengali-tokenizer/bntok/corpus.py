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


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Project-local by default (not the user's home cache) - large downloaded
# shards belong next to the repo that uses them, on whichever drive the
# repo itself lives on, not silently filling the OS drive's user profile.
# BNTOK_CACHE_DIR overrides this for anyone who wants a shared cache
# location instead. Already gitignored (see .gitignore's ".cache/" entry).
_CC100_CACHE_DIR = os.path.join(
    os.environ.get("BNTOK_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache")), "bntok", "cc100",
)


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


def stream_indiccorp_v2(
    lang: str = "bn", limit_lines: int = 300_000, offset_lines: int = 0,
) -> list[str]:
    """Stream AI4Bharat IndicCorp v2 (30.0B Bengali tokens: 10.6B verified,
    13.8B synthetic, 5.6B unverified - the largest published Indic-origin
    Bengali corpus as of 2026-08, 2.6x bigger than IndicCorp v2's own
    predecessor per AI4Bharat's own comparison) for one language.

    Optional: requires `requests`. Hosted as one large plain-text file per
    language (`data/{lang}.txt` in `ai4bharat/IndicCorpV2`, already one
    sentence/line), not parquet shards, so this streams the raw HTTP body
    line-by-line and stops as soon as `offset_lines + limit_lines` lines are
    read - it does not download the full file (each language file can run
    into the multi-GB range across the full 275GB dataset). Same
    line-based offset/limit contract as `stream_cc100`, so a disjoint
    held-out slice can be drawn the same way. Unlike CC-100 (2018 web
    crawl, BD/IN mixed origin), IndicCorp v2 is built by AI4Bharat (IIT
    Madras) specifically for Indian-language NLP, making it the best
    available proxy for India-sourced (as opposed to Bangladesh-sourced)
    Bengali web text - though the README does not itself label per-document
    provenance by country, so this is a pipeline-origin proxy, not a
    verified geographic filter.
    """
    try:
        import requests
    except ImportError as e:
        raise ConfigError(
            "streaming IndicCorp v2 needs the requests library: pip install requests"
        ) from e
    if limit_lines < 1:
        raise ConfigError(f"limit_lines must be >= 1, got {limit_lines}")
    if offset_lines < 0:
        raise ConfigError(f"offset_lines must be >= 0, got {offset_lines}")

    url = f"https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/{lang}.txt"
    out: list[str] = []
    seen = 0
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                seen += 1
                if seen <= offset_lines:
                    continue
                out.append(line)
                if seen - offset_lines >= limit_lines:
                    break
    except requests.RequestException as e:
        raise ConfigError(f"could not stream IndicCorp v2 for '{lang}': {e}") from e
    if not out:
        raise EmptyCorpusError(f"IndicCorp v2 stream for '{lang}' yielded no text")
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


def is_clean_bengali_line(line: str, min_ratio: float = 0.75, min_len: int = 4) -> bool:
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


_BANGLISH_MARKERS = frozenset({
    "ami", "amar", "amake", "amra", "amader", "tumi", "tomar", "tomake",
    "tomra", "tomader", "apni", "apnar", "apnake", "she", "tini", "ora",
    "era", "eta", "ota", "eita", "oita", "ei", "oi", "ache",
    "achi", "achen", "achis", "chilo", "chilam", "hobe", "hoyeche",
    "hoise", "hocche", "hoyese", "korbo", "korchi", "korche", "korchen",
    "korlam", "korlo", "korbe", "korte", "korar", "bhalo",
    "valo", "bhalobasha", "kharap", "sundor", "kemon", "keno", "kivabe",
    "kothay", "kobe", "koto", "kotota", "ki", "kina", "naki", "na", "hae",
    "haan", "jani", "jano", "jane", "bujhi", "bujho", "bujhte", "dekhi",
    "dekho", "shuni", "shono", "bolo", "bolchi", "bolte", "jabo", "jacchi",
    "jachhi", "asche", "eshechi", "gesi", "gesilo", "khub", "onek",
    "kichu", "shob", "sob", "kotha", "din", "raat", "sokal", "bikel",
    "ajke", "kalke", "porshu", "bhai", "apu", "dada", "didi", "bondhu",
})


def is_clean_banglish_line(line: str, min_len: int = 20, min_markers: int = 2,
                            min_marker_ratio: float = 0.06) -> bool:
    """Heuristic filter for genuine romanized-Bengali (Banglish) chat text
    inside a noisy web-crawl source.

    CC-100's `bn_rom` config is entirely Latin-script, so the script-ratio
    trick `is_clean_bengali_line` uses does not apply: everything here already
    passes an ASCII check. What actually separates real Banglish ("tumi kemon
    acho") from the English/business/news boilerplate that dominates a 2018
    CommonCrawl dump in this language slot is vocabulary, so this checks for a
    minimum count and share of common romanized Bengali function words and
    pronouns against a fixed marker list. Coarse, lexicon-based, not a
    language-ID model: it favours precision (real Banglish) over recall (it
    will miss valid Banglish that happens to avoid every marker word), which
    is the right trade-off for a held-out evaluation set, where a few clean
    lines are worth more than a large noisy one. Same honesty standard as
    `is_clean_bengali_line`: this reduces noise, it does not remove all of it.
    """
    if len(line) < min_len:
        return False
    words = re.findall(r"[a-zA-Z']+", line.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _BANGLISH_MARKERS)
    return hits >= min_markers and (hits / len(words)) >= min_marker_ratio


def build_banglish_held_out(limit_lines: int = 2000, scan_lines: int = 60_000,
                             log=lambda msg: None) -> list[str]:
    """Held-out romanized-Bengali (Banglish) chat-style text, filtered from
    CC-100's `bn_rom` config (Wenzek et al., 2020).

    Never used anywhere in `build_configured_corpus` or any training config in
    this repository (checked directly, not assumed), so there is no training
    corpus to stay disjoint from and no offset bookkeeping is needed, unlike
    `build_register_held_out`'s three registers. `bn_rom` is 2018-vintage
    CommonCrawl and mostly non-Bengali (English news/business text that
    happened to be crawled under this language code): `scan_lines` bounds how
    many raw rows are pulled and filtered through `is_clean_banglish_line`
    before giving up, so a request for more clean lines than the source
    actually contains fails loudly (`EmptyCorpusError` via `stream_cc100`)
    rather than hanging.
    """
    log(f"held-out: CC-100 bn_rom (banglish), scanning up to {scan_lines} raw lines ...")
    raw = stream_cc100(lang="bn_rom", limit_lines=scan_lines, offset_lines=0)
    out = [ln for ln in raw if is_clean_banglish_line(ln)][:limit_lines]
    log(f"  banglish: {len(out)}/{len(raw)} raw lines passed the filter")
    if not out:
        raise EmptyCorpusError("no clean Banglish lines found in the scanned CC-100 bn_rom sample")
    return out


def build_flores_held_out(splits: tuple[str, ...] = ("dev", "devtest"),
                           log=lambda msg: None) -> list[str]:
    """Held-out text from FLORES+ (openlanguagedata/flores_plus), the
    maintained successor to FLORES-200 - professionally translated,
    domain-diverse (wikinews/wikijunior/wikivoyage) parallel sentences,
    `ben_Beng` config.

    Motivation: an external paper ("The Tokenizer Tax", Srivastava 2026)
    measures Bengali fertility on FLORES-200's 997-sentence dev set
    specifically, and this project's own numbers were cross-walked against
    theirs (see docs/known-issues.md and _personal/SESSION_LOG.md session 6) -
    matching almost exactly on three tokenizers, independent corpora. This
    register lets a future comparison be measured on the SAME corpus their
    paper uses, instead of relying on that cross-walk.

    Two real access dead ends, checked directly rather than assumed, before
    this one worked:
      - facebook/flores and the original FLORES-200 loaders on Hugging Face
        (Muennighoff/flores200, gsarti/flores_101, facebook-llama/flores) are
        either manually-gated (needs human approval) or pure Python loading
        scripts with no underlying data files, which `datasets` no longer
        executes at all (the same class of break `stream_cc100` hit).
      - openlanguagedata/flores_plus is "auto-gated": the license auto-grants
        on the requesting account once accepted on the dataset's HF page (a
        one-time step; this repository's own huggingface_hub login already
        had it, verified by successfully loading 997 dev + 1012 devtest
        Bengali rows). No workaround needed once that page is visited once.

    Never used in `build_configured_corpus` or any training config, so - like
    `build_banglish_held_out` - there is no training range to stay disjoint
    from and no offset bookkeeping is needed.
    """
    from datasets import load_dataset
    log(f"held-out: FLORES+ (openlanguagedata/flores_plus), ben_Beng, splits {splits} ...")
    try:
        ds = load_dataset("openlanguagedata/flores_plus", "ben_Beng")
    except Exception as e:
        raise EmptyCorpusError(
            f"could not load openlanguagedata/flores_plus (ben_Beng): {type(e).__name__}: {e}. "
            "This dataset is auto-gated - visit "
            "https://huggingface.co/datasets/openlanguagedata/flores_plus once, accept the "
            "license, and make sure `huggingface_hub` is logged in with that account "
            "(`huggingface-cli login` or HF_TOKEN)."
        ) from e
    out = []
    for split in splits:
        if split not in ds:
            continue
        out.extend(row["text"] for row in ds[split] if row.get("text", "").strip())
    log(f"  flores: {len(out)} sentences across splits {[s for s in splits if s in ds]}")
    if not out:
        raise EmptyCorpusError("openlanguagedata/flores_plus (ben_Beng) loaded but yielded no text rows")
    return out


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
    character-ratio filter, see `is_clean_bengali_line`). Downloads only as
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
                lines = [ln for ln in lines if is_clean_bengali_line(ln)]
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
INDICCORP_V2_TRAIN_LINES = 300_000  # India-origin bulk web supplement (see stream_indiccorp_v2)


def build_configured_corpus(config: dict, log=lambda msg: None, dedup: bool = False) -> list[str]:
    """Build the weighted induction corpus described by a training config dict.

    `dedup=True` runs each loaded source through `bntok.dedup`'s full
    pipeline (exact dedup, MinHash near dedup, rule-based quality filter -
    see `docs/track-a2-corpus-survival.md`) before weighting, per-source,
    NOT on the final weighted output: `weighted_corpus` deliberately
    cycles a thin source (e.g. Wikisource, ~65 lines) to hit its target
    share, and deduping the final output would strip out that intentional
    repetition, defeating the weighting scheme entirely. Default False -
    this changes what a retrained artifact would contain versus the
    shipped `bn-bpe-64k`/`bmbt-64k`, so it is opt-in, not a silent
    behaviour change to the existing default. Near dedup is the slow
    stage (pure-Python MinHash, roughly 1,800-2,000 lines/sec on the
    machine this was measured on - see the survival-ratio doc for real
    wall-clock numbers on sources this size), so turning this on adds
    real time to a training run; every source's removal counts are
    logged via `log()`, not hidden.

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
      * indiccorp_v2_bn: AI4Bharat IndicCorp v2 Bengali (via
        `stream_indiccorp_v2`), the largest published India-origin Bengali
        corpus (30.0B tokens). Unlike cc100_general_web this IS part of the
        shipped `configs/bpe-64k.json` weights as of 2026-08 - added
        specifically to skew the corpus toward Indian-pipeline (vs
        Bangladesh-sourced) Bengali text; see docs/known-issues.md.

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
        "contemporary_news", "cc100_general_web", "indiccorp_v2_bn",
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
    if "indiccorp_v2_bn" in weights:
        log("loading indiccorp_v2_bn: AI4Bharat IndicCorp v2 bn (India-origin, holding out the tail for eval) ...")
        loaders["indiccorp_v2_bn"] = stream_indiccorp_v2("bn", limit_lines=INDICCORP_V2_TRAIN_LINES)
        log(f"  {len(loaders['indiccorp_v2_bn'])} lines")

    if dedup:
        from .dedup import (  # local: avoids a corpus<->dedup import cycle
            exact_dedup,
            near_dedup,
            quality_filter,
        )

        for name, lines in loaders.items():
            before = len(lines)
            lines, removed_exact = exact_dedup(lines)
            lines, removed_near = near_dedup(lines)
            lines, removed_quality = quality_filter(lines)
            loaders[name] = lines
            log(
                f"  dedup {name}: {before} -> {len(lines)} lines "
                f"(-{removed_exact} exact, -{removed_near} near-dup, -{removed_quality} low-quality)"
            )

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
