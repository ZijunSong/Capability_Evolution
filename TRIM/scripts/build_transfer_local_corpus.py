#!/usr/bin/env python3
"""Build offline query pools + BM25 indexes for TRIM transfer eval.

LongSeal: vtllms/sealqa ``longseal`` (gold + 30_docs).
FRAMES: google/frames-benchmark wiki links; page text from Wikipedia API or the
Wikimedia enwiki multistream dump (Range-fetch) when wikipedia.org is blocked.
HotpotQA: ``hotpot_qa`` distractor validation, 493-query subset (seed 42).
Web / Patents: try the private kellyhongg HF dumps; skip if they have no
document text (the paper corpora were never released publicly).

Example:
  PYTHONPATH=TRIM python TRIM/scripts/build_transfer_local_corpus.py --benchmark longsealqa
  PYTHONPATH=TRIM python TRIM/scripts/build_transfer_local_corpus.py --benchmark all
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.browsecomp_retrieval import _configure_java_runtime
from trim.eval.transfer_benchmarks import (
    OPTIONAL_PRIVATE_BENCHMARKS,
    TRANSFER_BENCHMARKS,
    canonical_transfer_benchmark,
    chunk_doc_id,
    default_transfer_root,
    normalize_url_id,
    wiki_title_key,
)
from trim.eval.wiki_dump import (
    decompress_multistream,
    parse_pages_from_stream,
    parse_redirect,
    scan_index_for_titles,
    wikitext_to_text,
)

WIKI_API = "https://en.wikipedia.org/w/api.php"
DUMP_INDEX_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2"
DUMP_XML_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
USER_AGENT = "TRIM-transfer-corpus/1.0 (local retrieval eval; https://github.com/pat-jj/harness-1)"
HOTPOT_SUBSET = 493
HOTPOT_SEED = 42
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200


def _is_lucene_index(path: Path) -> bool:
    return path.is_dir() and any(p.name.startswith("segments_") and p.is_file() for p in path.iterdir())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _chunk_text(text: str) -> list[str]:
    body = " ".join(str(text or "").split())
    if not body:
        return []
    if len(body) <= CHUNK_CHARS:
        return [body]
    out: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + CHUNK_CHARS)
        out.append(body[start:end])
        if end >= len(body):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return out


def _corpus_chunks(docid: str, text: str, *, title: str = "") -> list[dict[str, str]]:
    prefix = f"{title}\n\n" if title and title not in (text or "")[:200] else ""
    chunks = _chunk_text(prefix + (text or ""))
    if not chunks:
        return []
    if len(chunks) == 1:
        return [{"id": docid, "contents": chunks[0]}]
    return [{"id": chunk_doc_id(docid, i), "contents": chunk} for i, chunk in enumerate(chunks)]


def _index_jsonl(jsonl_path: Path, index_dir: Path, *, threads: int, python: str) -> None:
    _configure_java_runtime()
    os.environ.setdefault("OPENAI_API_KEY", "sk-pyserini-local")
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    collection = jsonl_path.parent
    cmd = [
        python,
        "-m",
        "pyserini.index.lucene",
        "-collection",
        "JsonCollection",
        "-input",
        str(collection),
        "-index",
        str(index_dir),
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(max(1, int(threads))),
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw",
        "-uniqueDocid",
    ]
    print("indexing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                return [text] if text else []
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _doc_record(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return {"id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], "title": "", "text": text, "url": ""}
    if not isinstance(raw, dict):
        return None
    url = str(raw.get("url") or raw.get("uri") or raw.get("link") or "").strip()
    title = str(raw.get("title") or raw.get("name") or "").strip()
    text = str(raw.get("text") or raw.get("contents") or raw.get("content") or raw.get("document_text") or "")
    docid = normalize_url_id(url) if url else (wiki_title_key(title) or title)
    if not docid:
        blob = (title + "\n" + text)[:200]
        if not blob.strip():
            return None
        docid = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    if not text.strip():
        return None
    return {"id": docid, "title": title, "text": text, "url": url}


def build_longseal(out: Path, *, limit: int | None) -> dict[str, Any]:
    from datasets import load_dataset

    local = out.parent / "_hf" / "longseal.parquet"
    if local.is_file():
        ds = load_dataset("parquet", data_files=str(local), split="train")
        source = f"local:{local}"
    else:
        ds = load_dataset("vtllms/sealqa", name="longseal", split="test")
        source = "vtllms/sealqa:longseal"
    queries: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, str]] = {}
    n = len(ds) if limit is None else min(int(limit), len(ds))
    for i in range(n):
        row = ds[i]
        qid = f"longseal-{i:03d}"
        golds = [_doc_record(x) for x in _as_list(row.get("golds"))]
        golds = [x for x in golds if x]
        pool_docs = []
        for key in ("30_docs", "20_docs", "12_docs", "golds"):
            pool_docs.extend(_doc_record(x) for x in _as_list(row.get(key)))
        for rec in golds + [x for x in pool_docs if x]:
            prev = corpus.get(rec["id"])
            if prev is None or len(rec["text"]) > len(prev["text"]):
                corpus[rec["id"]] = rec
        gold_ids = [x["id"] for x in golds] or [x["id"] for x in pool_docs if x][:1]
        queries.append(
            {
                "query_id": qid,
                "query": str(row.get("question") or ""),
                "answer": str(row.get("answer") or ""),
                "gold_docids": gold_ids,
                "evidence_docids": gold_ids,
                "official_split": "test",
                "source": source,
            }
        )
    chunks: list[dict[str, str]] = []
    for rec in corpus.values():
        chunks.extend(_corpus_chunks(rec["id"], rec["text"], title=rec.get("title") or ""))
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "corpus.jsonl", chunks)
    return {
        "benchmark": "longsealqa",
        "n_queries": len(queries),
        "n_source_docs": len(corpus),
        "n_chunks": len(chunks),
        "source": source,
    }


def _parse_wiki_links(row: dict[str, Any]) -> list[str]:
    links: list[str] = []
    raw = row.get("wiki_links") or row.get("wikipedia_links")
    for item in _as_list(raw):
        if isinstance(item, str) and item.strip():
            links.append(item.strip())
    for key, value in row.items():
        if str(key).startswith("wikipedia_link") and value:
            text = str(value).strip()
            if text and text.lower() not in {"null", "none", "nan"}:
                links.append(text)
    seen: set[str] = set()
    out: list[str] = []
    for link in links:
        title = wiki_title_key(link)
        if not title or title in seen:
            continue
        seen.add(title)
        out.append(title)
    return out


def _wiki_get(params: dict[str, str], *, retries: int = 6, timeout: int = 90) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{WIKI_API}?{query}", headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"wikipedia API failed: {last}")


def _mediawiki_available() -> bool:
    try:
        payload = _wiki_get({"action": "query", "format": "json", "titles": "Main Page"}, retries=1, timeout=8)
        return bool(payload.get("query"))
    except Exception:
        return False


def _wiki_cache_path() -> Path:
    return default_transfer_root() / "_hf" / "wiki_extracts.json"


def _load_wiki_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, rec in payload.items():
        if isinstance(rec, dict) and rec.get("text"):
            out[str(key)] = rec
    return out


def _save_wiki_cache(path: Path, pages: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _ingest_wiki_query(query: dict[str, Any], pages: dict[str, dict[str, str]]) -> None:
    normalized = {str(item.get("from")): str(item.get("to")) for item in query.get("normalized") or []}
    redirects = {str(item.get("from")): str(item.get("to")) for item in query.get("redirects") or []}
    alias: dict[str, str] = {}
    for src, dst in {**normalized, **redirects}.items():
        alias[wiki_title_key(src)] = wiki_title_key(dst)
    for page in (query.get("pages") or {}).values():
        if page.get("missing") or page.get("invalid"):
            continue
        title = wiki_title_key(str(page.get("title") or ""))
        extract = str(page.get("extract") or "").strip()
        if not title or not extract:
            continue
        rec = {
            "id": title,
            "title": title,
            "text": extract,
            "url": str(page.get("fullurl") or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"),
        }
        pages[title] = rec
    for src, dst in alias.items():
        if dst in pages and src not in pages:
            copied = dict(pages[dst])
            copied["id"] = src
            pages[src] = copied


def _dump_index_path() -> Path:
    return default_transfer_root() / "_hf" / "enwiki-multistream-index.txt.bz2"


def _curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "20",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    print("download:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure_dump_index() -> Path:
    path = _dump_index_path()
    if path.is_file() and path.stat().st_size > 100_000_000:
        return path
    _curl_download(DUMP_INDEX_URL, path)
    return path


def _dump_xml_size() -> int:
    req = urllib.request.Request(DUMP_XML_URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length") or "0")


def _http_range(url: str, start: int, end_exclusive: int, *, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Range": f"bytes={int(start)}-{int(end_exclusive) - 1}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"range fetch failed {start}-{end_exclusive}: {last}")


def _wiki_record(title: str, text: str) -> dict[str, str]:
    return {
        "id": title,
        "title": title,
        "text": text,
        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
    }


def _jobs_from_streams(streams: dict[int, dict[str, object]], xml_size: int) -> list[tuple[int, int, list]]:
    max_stream = 8 * 1024 * 1024
    jobs = []
    skipped = 0
    for offset, meta in streams.items():
        end = int(meta.get("end") or xml_size)
        if end <= int(offset) or (end - int(offset)) > max_stream:
            skipped += 1
            continue
        jobs.append((int(offset), end, list(meta.get("hits") or [])))
    if skipped:
        print(f"  skipped {skipped} dump streams with empty/huge ranges", flush=True)
    return jobs


def fetch_pages_from_enwiki_dump(titles: list[str], *, workers: int = 3) -> dict[str, dict[str, str]]:
    wanted = [wiki_title_key(t) or t for t in titles if t]
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        return {}
    index_path = _ensure_dump_index()
    xml_size = _dump_xml_size()
    print(f"  dump index scan for {len(wanted)} titles ({index_path.name})", flush=True)
    streams = scan_index_for_titles(index_path, wanted, xml_size=xml_size)
    print(f"  dump streams to fetch: {len(streams)}", flush=True)
    raw_pages: dict[str, str] = {}

    def _one(offset: int, end: int) -> dict[str, str]:
        blob = _http_range(DUMP_XML_URL, int(offset), int(end))
        xml = decompress_multistream(blob)
        return parse_pages_from_stream(xml)

    jobs = _jobs_from_streams(streams, xml_size)
    done = 0
    failed = 0
    batch_size = 24
    n_workers = max(1, int(workers))
    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {pool.submit(_one, offset, end): (offset, hits) for offset, end, hits in batch}
            for fut in as_completed(future_map):
                offset, hits = future_map[fut]
                try:
                    parsed = fut.result()
                except Exception as exc:
                    failed += 1
                    print(f"  dump stream {offset} failed: {exc}", flush=True)
                    continue
                by_key = {wiki_title_key(title): text for title, text in parsed.items()}
                for requested, dump_title in hits:
                    text = by_key.get(wiki_title_key(dump_title)) or by_key.get(wiki_title_key(requested))
                    if text is None:
                        continue
                    raw_pages[requested] = text
                    raw_pages[wiki_title_key(dump_title)] = text
                done += 1
        print(f"  dump pages {done}/{len(jobs)} streams (have {len(raw_pages)} titles, failed={failed})", flush=True)

    extra: list[str] = []
    for title, text in list(raw_pages.items()):
        target = parse_redirect(text)
        if target and target not in raw_pages:
            extra.append(target)
    extra = [t for t in dict.fromkeys(extra) if t]
    if extra:
        print(f"  dump following {len(extra)} redirects", flush=True)
        more = fetch_pages_from_enwiki_dump(extra, workers=workers)
        for rec in more.values():
            raw_pages[rec["id"]] = rec["text"]
            raw_pages[wiki_title_key(rec["title"])] = rec["text"]

    out: dict[str, dict[str, str]] = {}
    for title, text in raw_pages.items():
        target = parse_redirect(text)
        body = raw_pages.get(target, text) if target else text
        plain = wikitext_to_text(body)
        if not plain:
            continue
        key = wiki_title_key(title) or title
        out[key] = _wiki_record(key, plain)
    return out


def fetch_wikipedia_pages(titles: list[str]) -> dict[str, dict[str, str]]:
    cache_path = _wiki_cache_path()
    pages = _load_wiki_cache(cache_path)
    pending = [t for t in titles if t and t not in pages]
    if pending and _mediawiki_available():
        batch_size = 10
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            payload = _wiki_get(
                {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts|info",
                    "explaintext": "1",
                    "exlimit": str(batch_size),
                    "exsectionformat": "plain",
                    "inprop": "url",
                    "redirects": "1",
                    "titles": "|".join(batch),
                }
            )
            _ingest_wiki_query(payload.get("query") or {}, pages)
            print(f"  wikipedia {min(i + len(batch), len(pending))}/{len(pending)} new pages (cache={len(pages)})", flush=True)
            if i == 0 or (i + batch_size) % 50 == 0 or i + batch_size >= len(pending):
                _save_wiki_cache(cache_path, pages)
            time.sleep(0.15)
    still = [t for t in pending if t not in pages]
    if still:
        print(f"  fetching {len(still)} pages from Wikimedia dump", flush=True)
        pages.update(fetch_pages_from_enwiki_dump(still))
    if pending:
        _save_wiki_cache(cache_path, pages)
    return {t: pages[t] for t in titles if t in pages}


def _load_frames_rows(*, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    local = default_transfer_root() / "_hf" / "frames_test.tsv"
    if local.is_file() and local.stat().st_size > 1000:
        with local.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        source = f"local:{local}"
    else:
        from datasets import load_dataset

        ds = load_dataset("google/frames-benchmark", split="test")
        rows = [ds[i] for i in range(len(ds))]
        source = "google/frames-benchmark"
    if limit is not None:
        rows = rows[: int(limit)]
    return rows, source


def build_frames(out: Path, *, limit: int | None) -> dict[str, Any]:
    ds_rows, source = _load_frames_rows(limit=limit)
    queries_raw: list[dict[str, Any]] = []
    titles: list[str] = []
    for i, row in enumerate(ds_rows):
        gold = _parse_wiki_links(row)
        raw_idx = row.get("Unnamed: 0") or row.get("") or row.get(None) or i
        try:
            qid = f"frames-{int(raw_idx):03d}"
        except (TypeError, ValueError):
            qid = f"frames-{i:03d}"
        queries_raw.append(
            {
                "query_id": qid,
                "query": str(row.get("Prompt") or row.get("prompt") or row.get("question") or ""),
                "answer": str(row.get("Answer") or row.get("answer") or ""),
                "gold_docids": gold,
                "evidence_docids": gold,
                "official_split": "test",
                "source": source,
            }
        )
        titles.extend(gold)
    unique_titles = list(dict.fromkeys(titles))
    print(f"FRAMES: fetching {len(unique_titles)} wikipedia pages", flush=True)
    pages = fetch_wikipedia_pages(unique_titles)
    queries: list[dict[str, Any]] = []
    for rec in queries_raw:
        gold = [t for t in rec["gold_docids"] if t in pages]
        if not gold:
            continue
        rec = dict(rec)
        rec["gold_docids"] = gold
        rec["evidence_docids"] = gold
        queries.append(rec)
    chunks: list[dict[str, str]] = []
    for rec in pages.values():
        chunks.extend(_corpus_chunks(rec["id"], rec["text"], title=rec.get("title") or ""))
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "corpus.jsonl", chunks)
    return {
        "benchmark": "frames",
        "n_queries": len(queries),
        "n_source_docs": len(pages),
        "n_chunks": len(chunks),
        "n_dropped_no_pages": len(queries_raw) - len(queries),
        "source": f"{source} + Wikipedia dump/API extracts",
    }


def build_hotpotqa(out: Path, *, limit: int | None) -> dict[str, Any]:
    from datasets import load_dataset

    ds = None
    source = "hotpot_qa:distractor"
    for args in (("hotpot_qa", "distractor"), ("hotpotqa/hotpot_qa", "distractor")):
        try:
            ds = load_dataset(args[0], args[1], split="validation")
            source = f"{args[0]}:{args[1]}:validation"
            break
        except Exception as exc:
            last = exc
            ds = None
    if ds is None:
        raise RuntimeError(f"could not load HotpotQA: {last}")
    rows = list(range(len(ds)))
    rng = random.Random(HOTPOT_SEED)
    want = HOTPOT_SUBSET if limit is None else min(int(limit), HOTPOT_SUBSET, len(rows))
    picked = sorted(rng.sample(rows, want)) if want < len(rows) else rows[:want]
    queries: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, str]] = {}
    for idx in picked:
        row = ds[int(idx)]
        qid = str(row.get("id") or f"hotpot-{idx:04d}")
        supporting = row.get("supporting_facts") or {}
        if isinstance(supporting, dict):
            gold_titles = [wiki_title_key(str(t)) for t in _as_list(supporting.get("title"))]
        else:
            gold_titles = [wiki_title_key(str(item[0] if isinstance(item, (list, tuple)) else item)) for item in _as_list(supporting)]
        gold_titles = [t for t in dict.fromkeys(gold_titles) if t]
        context = row.get("context") or {}
        if isinstance(context, dict):
            titles = _as_list(context.get("title"))
            sentences = _as_list(context.get("sentences"))
        else:
            pairs = _as_list(context)
            titles = [item[0] if isinstance(item, (list, tuple)) else item for item in pairs]
            sentences = [item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else [] for item in pairs]
        for title_raw, sents in zip(titles, sentences + [[]] * len(titles)):
            title = wiki_title_key(str(title_raw))
            if isinstance(sents, (list, tuple)):
                text = " ".join(str(x) for x in sents)
            else:
                text = str(sents or "")
            if not title or not text.strip():
                continue
            rec = {"id": title, "title": title, "text": text, "url": ""}
            prev = corpus.get(title)
            if prev is None or len(text) > len(prev["text"]):
                corpus[title] = rec
        if not gold_titles:
            gold_titles = list(corpus)[-2:]
        queries.append(
            {
                "query_id": qid,
                "query": str(row.get("question") or ""),
                "answer": str(row.get("answer") or ""),
                "gold_docids": gold_titles,
                "evidence_docids": gold_titles,
                "official_split": "test",
                "source": source,
            }
        )
    chunks: list[dict[str, str]] = []
    for rec in corpus.values():
        chunks.extend(_corpus_chunks(rec["id"], rec["text"], title=rec.get("title") or ""))
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "corpus.jsonl", chunks)
    return {
        "benchmark": "hotpotqa",
        "n_queries": len(queries),
        "n_source_docs": len(corpus),
        "n_chunks": len(chunks),
        "subset": want,
        "seed": HOTPOT_SEED,
        "source": source,
        "note": (
            "493-query subset sampled from HotpotQA distractor validation with seed 42. "
            "This matches the Harness-1 count, not necessarily kellyhongg/hotpotqa_subset ids."
        ),
    }


def _hf_has_document_text(ds) -> bool:
    if len(ds) == 0:
        return False
    row = ds[0]
    for key in ("document_text", "contents", "text", "chunk_text", "passages", "documents", "corpus"):
        if key in row and row[key]:
            return True
    return False


def build_private_hf(name: str, out: Path, *, limit: int | None) -> dict[str, Any]:
    from datasets import load_dataset

    specs = {
        "web": [("kellyhongg/1_17_web_test", "test"), ("kellyhongg/web_1_17_test", "test")],
        "patents": [("kellyhongg/1_18_patents_test", "test")],
    }
    last: Exception | None = None
    for hf_path, split in specs[name]:
        try:
            ds = load_dataset(hf_path, split=split)
        except Exception as exc:
            last = exc
            continue
        if not _hf_has_document_text(ds):
            raise RuntimeError(
                f"{hf_path} has queries/qrels but no document text. "
                "The Context-1 Web/Patents Chroma corpora were not released; cannot rebuild locally."
            )
        raise RuntimeError(f"{hf_path} has unexpected document fields; extend the builder")
    raise RuntimeError(
        f"could not load a public {name} corpus ({last}). "
        "Skip Web/Patents: the paper indexes are private Chroma collections."
    )


BUILDERS = {
    "longsealqa": build_longseal,
    "frames": build_frames,
    "hotpotqa": build_hotpotqa,
    "web": lambda out, limit: build_private_hf("web", out, limit=limit),
    "patents": lambda out, limit: build_private_hf("patents", out, limit=limit),
}


def build_one(name: str, *, root: Path, threads: int, python: str, limit: int | None, force: bool, skip_index: bool) -> dict[str, Any]:
    canon = canonical_transfer_benchmark(name)
    if canon is None or canon not in BUILDERS:
        raise SystemExit(f"unknown benchmark {name}")
    out = root / canon
    index_dir = out / "indexes" / "bm25"
    if (out / "queries.jsonl").is_file() and (out / "corpus.jsonl").is_file() and not force:
        if skip_index or _is_lucene_index(index_dir):
            print(f"{canon}: already built at {out}")
            return {"benchmark": canon, "skipped": True, "path": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    print(f"== building {canon} -> {out}", flush=True)
    stats = BUILDERS[canon](out, limit=limit)
    jsonl_dir = out / "collections" / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    src = out / "corpus.jsonl"
    shard = jsonl_dir / "docs.json"
    shutil.copyfile(src, shard)
    if not skip_index:
        try:
            _index_jsonl(shard, index_dir, threads=threads, python=python)
            stats["index"] = str(index_dir)
        except Exception as exc:
            stats["index_error"] = repr(exc)
            print(f"{canon}: Lucene index failed ({exc}); eval can fall back to corpus.jsonl", flush=True)
    stats["path"] = str(out)
    (out / "BUILD.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="all",
        help="longsealqa | frames | hotpotqa | web | patents | all",
    )
    parser.add_argument("--root", type=Path, default=default_transfer_root())
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-benchmark query cap (smoke).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args(argv)

    requested = str(args.benchmark).strip().lower()
    if requested in {"all", "*"}:
        names = list(TRANSFER_BENCHMARKS) + list(OPTIONAL_PRIVATE_BENCHMARKS)
    else:
        names = [requested]
    results = []
    failures = []
    for name in names:
        canon = canonical_transfer_benchmark(name) or name
        try:
            results.append(
                build_one(
                    canon,
                    root=args.root,
                    threads=int(args.threads),
                    python=str(args.python),
                    limit=args.limit,
                    force=bool(args.force),
                    skip_index=bool(args.skip_index),
                )
            )
        except Exception as exc:
            if canon in OPTIONAL_PRIVATE_BENCHMARKS:
                print(f"SKIP {canon}: {exc}", flush=True)
                results.append({"benchmark": canon, "skipped": True, "reason": str(exc)})
            else:
                failures.append({"benchmark": canon, "error": repr(exc)})
                print(f"FAIL {canon}: {exc}", flush=True)
    summary = {"root": str(args.root), "results": results, "failures": failures}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"failed: {[f['benchmark'] for f in failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
