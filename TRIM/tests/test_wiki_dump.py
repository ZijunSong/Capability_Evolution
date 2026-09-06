from __future__ import annotations

import bz2

from trim.eval.wiki_dump import (
    decompress_multistream,
    parse_index_line,
    parse_pages_from_stream,
    parse_redirect,
    scan_index_for_titles,
    wikitext_to_text,
)


def test_parse_index_line_and_redirect(tmp_path):
    assert parse_index_line("100:12:Albert Einstein") == (100, 12, "Albert Einstein")
    assert parse_redirect("#REDIRECT [[United States#History]]") == "United States"
    assert parse_redirect("Albert Einstein was a physicist") is None
    text = wikitext_to_text("Hello {{cite}} [[Barack Obama|Obama]] and [[Earth]].")
    assert "Obama" in text
    assert "Earth" in text
    assert "{{" not in text


def test_parse_and_scan_multistream(tmp_path):
    xml = (
        "<page><title>Harriet Lane</title><ns>0</ns>"
        "<text xml:space='preserve'>Harriet Lane was First Lady. [[James Buchanan|Buchanan]] hosted her.</text>"
        "</page>"
        "<page><title>Talk:Skip</title><ns>1</ns><text>ignore</text></page>"
    ).encode("utf-8")
    blob = bz2.compress(xml)
    parsed = parse_pages_from_stream(decompress_multistream(blob))
    assert "Harriet Lane" in parsed
    assert "Talk:Skip" not in parsed
    index = tmp_path / "index.bz2"
    with bz2.open(index, "wt", encoding="utf-8") as handle:
        handle.write("10:1:Harriet Lane\n20:2:Other\n")
    streams = scan_index_for_titles(index, ["Harriet Lane"], xml_size=99)
    assert 10 in streams
    assert streams[10]["end"] == 20
    assert streams[10]["hits"] == [("Harriet Lane", "Harriet Lane")]
