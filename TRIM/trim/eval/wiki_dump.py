"""Helpers for reading specific English Wikipedia pages from a multistream dump."""

from __future__ import annotations

import bz2
import html
import re
from pathlib import Path
from typing import Iterable

from trim.eval.transfer_benchmarks import wiki_title_key

PAGE_RE = re.compile(r"<page>(.*?)</page>", re.S)
TITLE_RE = re.compile(r"<title>(.*?)</title>")
NS_RE = re.compile(r"<ns>(.*?)</ns>")
TEXT_RE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
REDIRECT_RE = re.compile(r"#\s*redirect\s*\[\[([^\]|#]+)", re.I)
FILE_RE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.I)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
SPACE_RE = re.compile(r"\s+")


def parse_index_line(line: str) -> tuple[int, int, str] | None:
    text = line.strip()
    if not text or text.count(":") < 2:
        return None
    offset_s, pageid_s, title = text.split(":", 2)
    try:
        return int(offset_s), int(pageid_s), html.unescape(title)
    except ValueError:
        return None


def title_lookup_keys(title: str) -> set[str]:
    key = wiki_title_key(title)
    out = {str(title or "").strip(), key}
    if key:
        cap = key[:1].upper() + key[1:]
        out.update({cap, key.replace(" ", "_"), cap.replace(" ", "_")})
    return {item for item in out if item}


def parse_redirect(wikitext: str) -> str | None:
    match = REDIRECT_RE.search(wikitext or "")
    if not match:
        return None
    return wiki_title_key(match.group(1).strip()) or None


def wikitext_to_text(raw: str) -> str:
    text = html.unescape(raw or "")
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = FILE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    for _ in range(8):
        nxt = TEMPLATE_RE.sub(" ", text)
        if nxt == text:
            break
        text = nxt
    def _link(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "|" in inner:
            return inner.split("|")[-1]
        return inner
    text = LINK_RE.sub(_link, text)
    text = text.replace("'''", "").replace("''", "")
    return SPACE_RE.sub(" ", text).strip()


def parse_pages_from_stream(data: bytes) -> dict[str, str]:
    xml = data.decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for block in PAGE_RE.findall(xml):
        title_m = TITLE_RE.search(block)
        if not title_m:
            continue
        ns_m = NS_RE.search(block)
        if ns_m and ns_m.group(1).strip() != "0":
            continue
        title = html.unescape(title_m.group(1))
        text_m = TEXT_RE.search(block)
        out[title] = html.unescape(text_m.group(1)) if text_m else ""
    return out


def scan_index_for_titles(
    index_path: Path,
    titles: Iterable[str],
    *,
    xml_size: int,
) -> dict[int, dict[str, object]]:
    """Map dump stream offsets to requested titles contained in that stream."""
    wanted: dict[str, str] = {}
    for title in titles:
        requested = wiki_title_key(title) or str(title).strip()
        if not requested:
            continue
        for key in title_lookup_keys(requested):
            wanted[key] = requested

    streams: dict[int, dict[str, object]] = {}
    current: int | None = None
    with bz2.open(index_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_index_line(line)
            if parsed is None:
                continue
            offset, _pageid, dump_title = parsed
            if offset != current:
                if current is not None and current in streams and streams[current]["end"] is None:
                    streams[current]["end"] = offset
                current = offset
            requested = wanted.get(wiki_title_key(dump_title)) or wanted.get(dump_title)
            if requested is None:
                continue
            rec = streams.setdefault(offset, {"end": None, "hits": []})
            hits = rec["hits"]
            assert isinstance(hits, list)
            hits.append((requested, dump_title))
    if current is not None and current in streams and streams[current]["end"] is None:
        streams[current]["end"] = int(xml_size)
    return streams


def decompress_multistream(blob: bytes) -> bytes:
    dec = bz2.BZ2Decompressor()
    out = dec.decompress(blob)
    while not dec.eof:
        extra = dec.unused_data
        if not extra:
            break
        dec = bz2.BZ2Decompressor()
        out += dec.decompress(extra)
    return out
