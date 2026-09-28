"""Parser-output adapters: native JSON first, Markdown only as a fallback.

Design rationale (verified against this project's raw outputs):

- All three parsers emit native JSON, so Markdown heuristics are not the
  primary source of block structure:
  - PaddleOCR ``PPStructureV3`` writes ``<document>_<page>_res.json`` with
    ``parsing_res_list`` entries carrying ``block_label`` (``doc_title``,
    ``paragraph_title``, ``text``, ``table``, ``image``, ``figure_title``),
    ``block_content``, ``block_bbox``, ``block_id`` and ``block_order``.
  - MinerU 4.x ``--format zip`` bundles ``middle_json.json`` whose page blocks
    carry ``type`` (``doc_title``, ``paragraph_title``, ``text``, ``table``,
    ``image``), ``level``, normalized ``bbox`` and clean content spans. The
    bundled ``markdown.md`` escapes list markers (``\\-``), so the JSON spans
    are more reliable than the Markdown.
  - Docling writes ``document.json`` with ``texts`` carrying ``label``
    (``section_header``, ``text``, ``list_item``), ``level``, page/bbox
    provenance, plus ``pictures`` (with child text refs), ``tables`` (with
    ``table_cells``) and ``groups``. ``body.children`` gives the exact
    interleaved reading order.
- Native JSON still does not label footnotes or references explicitly, so a
  small, documented, section-aware heuristic layer resolves those on top of
  native labels and heading context.
- Bounding boxes are preserved verbatim when present, but coordinate systems
  differ per tool (PaddleOCR pixels, MinerU normalized ratios, Docling PDF
  points), so cross-parser bbox comparison is out of scope for this POC.
"""

from __future__ import annotations

import html.parser
import json
import re
from pathlib import Path

from .models import Block
from .parsers import ParserRunError


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
NUMBERED_MARKER = re.compile(r"^(\d+)[.)]\s?(.*)$")
FIGURE_LABEL = re.compile(r"^(Figure\s+\d+\.\s*)(.*)$", re.DOTALL)
FIGURE_PREFIX = re.compile(r"^(figure\s+\d+\.|fig\.\s+\d+\.)\s*(.*)$", re.DOTALL | re.IGNORECASE)
REFERENCES_HEADING = re.compile(r"^(references|bibliography|works?\s+cited|acknowledg(e)?ments?)$")
YEAR = re.compile(r"\b(19|20)\d{2}\b")
CITATION_EXTRA = re.compile(r"(\d+\s*:\s*\d+|\bvol\.|\bpp\.\s*\d|journal|proceedings|transactions)", re.IGNORECASE)
BULLET_PREFIXES = ("- ", "* ", "\u2022 ", "\ufffd ", "\ufffd")
SENTENCE_END = re.compile(r"(?<=[.!?])\s*(?=[A-Z0-9(\[])")

# Block kinds that may terminate a chunk in ground-truth boundary evaluation.
BOUNDARY_KINDS = {"heading", "paragraph", "table", "list", "caption", "reference"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _unescape_markdown(text: str) -> str:
    """Remove backslash escapes that parsers emit for list markers/headings."""
    return re.sub(r"\\([-*#])", r"\1", text)


class _Item:
    """A provisional canonical block awaiting cross-block resolution."""

    __slots__ = ("kind", "text", "level", "page", "bbox", "tentative_footnote", "marker_number", "resolved_path")

    def __init__(
        self,
        kind: str,
        text: str,
        level: int | None = None,
        page: int = 1,
        bbox: list[float] | None = None,
        tentative_footnote: bool = False,
        marker_number: int | None = None,
    ) -> None:
        self.kind = kind
        self.text = text
        self.level = level
        self.page = page
        self.bbox = bbox
        self.tentative_footnote = tentative_footnote
        self.marker_number = marker_number
        self.resolved_path: list[str] = []


class _HTMLTableParser(html.parser.HTMLParser):
    """Collect table rows/cells from MinerU/Paddle HTML table fragments."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(_clean("".join(self._current_cell)))
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def serialize_table_cells(rows: list[list[str]]) -> str:
    """Serialize table rows in the ground-truth ``a | b; c | d`` text format."""
    return "; ".join(" | ".join(cell for cell in row) for row in rows if row)


def table_html_to_text(html_text: str) -> str:
    parser = _HTMLTableParser()
    parser.feed(html_text)
    return serialize_table_cells(parser.rows)


def split_caption(text: str) -> tuple[str, str]:
    """Split a ``Figure N.`` caption from a paragraph merged into the same export.

    Several exporters concatenate the caption sentence and the following body
    sentence (sometimes without an intervening space). The caption keeps the
    ``Figure N.`` label plus its first sentence; anything after becomes a
    paragraph so no content is discarded.
    """
    match = FIGURE_LABEL.match(text.strip())
    if not match:
        return text, ""
    label, body = match.group(1), match.group(2).strip()
    sentences = SENTENCE_END.split(body)
    if len(sentences) < 2:
        return text.strip(), ""
    return f"{label.strip()} {sentences[0].strip()}".strip(), " ".join(part.strip() for part in sentences[1:] if part.strip())


def _strip_list_marker(text: str) -> str:
    """Remove a leading bullet/ordered marker, mirroring ground-truth list text."""
    cleaned = _unescape_markdown(text.strip())
    for prefix in ("- ", "* ", "\u2022 ", "\ufffd ", "\ufffd"):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    numbered = NUMBERED_MARKER.match(cleaned)
    if numbered:
        return numbered.group(2).strip()
    return cleaned


def _classify_text_line(text: str, last_numbered: int | None) -> tuple[str, bool, int | None]:
    """Classify a plain-text export line.

    Returns ``(kind, tentative_footnote, marker_number)``. A numbered line is
    only a *tentative* footnote here: an incrementing ``1. 2. 3.`` run is an
    ordered list, while a lone number is resolved later against section
    context (see :func:`_resolve_footnotes`).
    """
    cleaned = _unescape_markdown(text.strip())
    numbered = NUMBERED_MARKER.match(cleaned)
    if numbered:
        number = int(numbered.group(1))
        if last_numbered is not None and number == last_numbered + 1:
            return "list", False, number
        return "paragraph", True, number
    if cleaned.startswith(BULLET_PREFIXES):
        return "list", False, None
    if FIGURE_PREFIX.match(cleaned):
        return "caption", False, None
    return "paragraph", False, None


def _is_references_heading(text: str) -> bool:
    return REFERENCES_HEADING.match(_clean(text).lower()) is not None


def _looks_like_citation(text: str) -> bool:
    return YEAR.search(text) is not None and CITATION_EXTRA.search(text) is not None


def _next_heading(items: list[_Item], start: int) -> _Item | None:
    for item in items[start:]:
        if item.kind == "heading":
            return item
    return None


def _resolve_footnotes(items: list[_Item]) -> None:
    """Resolve tentative footnotes using section context.

    A lone numbered marker (one that does not continue an incrementing ordered
    list) becomes a footnote when the next heading starts a new top-level
    section, heads a references-like section, or when the marker is the final
    content block. Otherwise it is an ordered-list item.
    """
    for index, item in enumerate(items):
        if not item.tentative_footnote:
            continue
        following = _next_heading(items, index + 1)
        if following is None or (following.level or 2) <= 1 or _is_references_heading(following.text):
            item.kind = "footnote"
            item.tentative_footnote = False
        else:
            item.kind = "list"
            item.tentative_footnote = False
            item.text = _strip_list_marker(item.text)


def _classify_references(items: list[_Item]) -> None:
    """Classify citation-like paragraphs inside references-like sections."""
    top_heading: str | None = None
    for item in items:
        if item.kind == "heading" and (item.level or 2) <= 1:
            top_heading = item.text
        if item.kind == "paragraph" and top_heading is not None and _is_references_heading(top_heading) and _looks_like_citation(item.text):
            item.kind = "reference"


def _join_adjacent_lists(items: list[_Item]) -> list[_Item]:
    """Join consecutive same-section list items, mirroring ground-truth lists."""
    merged: list[_Item] = []
    for item in items:
        if (
            item.kind == "list"
            and merged
            and merged[-1].kind == "list"
            and merged[-1].page == item.page
        ):
            merged[-1].text = f"{merged[-1].text}; {_strip_list_marker(item.text)}"
        else:
            merged.append(item)
    return merged


def _normalize_heading_levels(items: list[_Item]) -> None:
    """Map observed heading depths so the shallowest becomes level 1.

    Exported depths are relative (PaddleOCR emits ``##``/``###`` for H1/H2),
    so shifting preserves nesting without trusting absolute ``#`` counts.
    Standalone references-like headings always reset to level 1.
    """
    levels = [item.level for item in items if item.kind == "heading" and item.level]
    if levels:
        shift = min(levels) - 1
        for item in items:
            if item.kind == "heading" and item.level:
                item.level = max(1, item.level - shift)
    for item in items:
        if item.kind == "heading" and _is_references_heading(item.text):
            item.level = 1


def _rebuild_heading_paths(items: list[_Item]) -> None:
    """Recompute heading ancestry from normalized levels (stored alongside)."""
    path: list[str] = []
    item_paths: list[list[str]] = []
    for item in items:
        if item.kind == "heading":
            level = item.level or 1
            path = path[: level - 1] + [item.text]
            item_paths.append(list(path))
        else:
            item_paths.append(list(path))
    for item, resolved in zip(items, item_paths, strict=True):
        item.resolved_path = resolved


def _finalize(items: list[_Item], document_name: str) -> list[Block]:
    _normalize_heading_levels(items)
    _rebuild_heading_paths(items)
    _resolve_footnotes(items)
    _classify_references(items)
    blocks: list[Block] = []
    for item in _join_adjacent_lists(items):
        text = _strip_list_marker(item.text) if item.kind == "list" else _clean(item.text)
        if not text:
            # Empty exports (for example MinerU image bodies without OCR text)
            # carry no comparable content; dropping them avoids phantom blocks.
            continue
        path = list(getattr(item, "resolved_path", []))
        if item.kind == "heading":
            level = item.level or 1
            path = path[:level]
        blocks.append(Block(f"{document_name}-{len(blocks):04d}", text, item.kind, item.page, item.level, list(path), item.bbox))
    return blocks


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParserRunError(f"Could not read native parser JSON at {path}: {exc}") from exc


def _paddle_title_levels(raw_dir: Path, document_stem: str, titles: list[str]) -> dict[str, int]:
    """Recover heading depths from PaddleOCR Markdown (native JSON has none).

    Titles are matched by normalized text so page-file ordering differences do
    not misalign levels; unmatched titles fall back to depth 2 (body heading).
    """
    levels: dict[str, int] = {}
    ordered: list[int] = []
    for markdown_file in sorted(raw_dir.glob(f"{document_stem}_*.md")):
        if markdown_file.name == "benchmark.md":
            continue
        for line in markdown_file.read_text(encoding="utf-8").splitlines():
            match = HEADING.match(line.strip())
            if match:
                ordered.append(len(match.group(1)))
                levels.setdefault(_clean(match.group(2)).lower(), len(match.group(1)))
    for position, title in enumerate(titles):
        key = _clean(title).lower()
        if key not in levels:
            levels[key] = ordered[position] if position < len(ordered) else 2
    return levels


def _paddle_items(raw_dir: Path, document_stem: str) -> list[_Item] | None:
    result_files = sorted(raw_dir.glob(f"{document_stem}_*_res.json"))
    if not result_files:
        return None
    titles: list[str] = []
    for path in result_files:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("parsing_res_list", []):
            if entry.get("block_label") in ("doc_title", "paragraph_title") and _clean(str(entry.get("block_content", ""))):
                titles.append(_clean(str(entry.get("block_content", ""))))
    title_levels = _paddle_title_levels(raw_dir, document_stem, titles)
    items: list[_Item] = []
    last_numbered: int | None = None
    for path in result_files:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        page = int(payload.get("page_index", 0)) + 1
        entries = payload.get("parsing_res_list", [])

        def sort_key(entry: dict) -> tuple[int, float, int]:
            order = entry.get("block_order")
            ident = entry.get("block_id")
            ident_value = ident if isinstance(ident, int) else 0
            if isinstance(order, int):
                return (0, float(order), ident_value)
            # Entries without an explicit order (tables, images, figure titles)
            # keep layout position just after their numeric block id.
            return (0, float(ident_value) + 0.5, ident_value)

        for entry in sorted(entries, key=sort_key):
            label = entry.get("block_label")
            content = _clean(str(entry.get("block_content", "")))
            bbox = entry.get("block_bbox")
            box = [float(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
            if label in ("doc_title", "paragraph_title"):
                if not content:
                    continue
                items.append(_Item("heading", content, title_levels.get(content.lower(), 2), page, box))
                last_numbered = None
            elif label == "table":
                text = table_html_to_text(content)
                if text:
                    items.append(_Item("table", text, None, page, box))
                last_numbered = None
            elif label == "image":
                # Diagram OCR text (node labels) is the comparable content; an
                # image without OCR text carries nothing comparable, so the
                # shared finalizer drops it instead of emitting a phantom block.
                if content:
                    items.append(_Item("figure", content, None, page, box))
                last_numbered = None
            elif label == "figure_title":
                if content:
                    caption, remainder = split_caption(content)
                    items.append(_Item("caption", caption, None, page, box))
                    if remainder:
                        items.append(_Item("paragraph", remainder, None, page, box))
                last_numbered = None
            else:
                kind, tentative, number = _classify_text_line(content, last_numbered)
                items.append(_Item(kind, content, None, page, box, tentative, number))
                last_numbered = number if number is not None else last_numbered
    return _finalize(items, document_stem)


def _miner_span_text(node: object) -> str:
    """Flatten MinerU content spans to plain text (tables/images handled apart)."""
    parts: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            node_type = value.get("type")
            if node_type in ("text", "image_caption", "table_caption"):
                visit(value.get("content", ""))
            elif node_type in ("image_body", "table_body"):
                return
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return _clean(" ".join(part for part in parts if part))


def _miner_items(raw_dir: Path, document_stem: str) -> list[_Item] | None:
    path = raw_dir / "middle_json.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ParserRunError(f"Unexpected MinerU middle JSON shape in {path}.")
    items: list[_Item] = []
    last_numbered: int | None = None
    for page in sorted(payload.get("pages", []), key=lambda entry: entry.get("page_idx", 0)):
        page_no = int(page.get("page_idx", 0)) + 1
        for block in page.get("blocks", []):
            block_type = block.get("type")
            level = block.get("level")
            level_value = int(level) if isinstance(level, int) else None
            bbox = block.get("bbox")
            box = [float(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
            content = block.get("content", [])
            if block_type in ("doc_title", "paragraph_title"):
                text = _miner_span_text(content)
                if text:
                    items.append(_Item("heading", text, level_value or (1 if block_type == "doc_title" else 2), page_no, box))
                last_numbered = None
            elif block_type == "table":
                html_text = ""
                captions: list[str] = []
                spans = content if isinstance(content, list) else [content]
                for span in spans:
                    if isinstance(span, dict) and span.get("type") == "table_body":
                        html_text = str(span.get("content", ""))
                    elif isinstance(span, dict) and span.get("type") in ("table_caption", "image_caption"):
                        captions.append(_miner_span_text(span))
                text = table_html_to_text(html_text)
                if text:
                    items.append(_Item("table", text, None, page_no, box))
                for caption_text in captions:
                    caption, remainder = split_caption(caption_text)
                    items.append(_Item("caption", caption, None, page_no, box))
                    if remainder:
                        items.append(_Item("paragraph", remainder, None, page_no, box))
                last_numbered = None
            elif block_type == "image":
                bodies: list[str] = []
                captions: list[str] = []
                spans = content if isinstance(content, list) else [content]
                for span in spans:
                    if isinstance(span, dict) and span.get("type") == "image_body":
                        bodies.append(_clean(str(span.get("content", ""))))
                    elif isinstance(span, dict) and span.get("type") in ("image_caption", "table_caption"):
                        captions.append(_miner_span_text(span))
                body_text = _clean(" ".join(part for part in bodies if part))
                if body_text:
                    items.append(_Item("figure", body_text, None, page_no, box))
                for caption_text in captions:
                    caption, remainder = split_caption(caption_text)
                    items.append(_Item("caption", caption, None, page_no, box))
                    if remainder:
                        items.append(_Item("paragraph", remainder, None, page_no, box))
                last_numbered = None
            else:
                text = _miner_span_text(content)
                if not text:
                    continue
                kind, tentative, number = _classify_text_line(text, last_numbered)
                items.append(_Item(kind, text, None, page_no, box, tentative, number))
                last_numbered = number if number is not None else last_numbered
    return _finalize(items, document_stem)


def _docling_ref(ref: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"#/([a-z_]+)/(\d+)", ref)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _docling_box(prov: list[dict]) -> tuple[int, list[float] | None]:
    page = 1
    box: list[float] | None = None
    if prov:
        first = prov[0]
        page = int(first.get("page_no", 1))
        raw = first.get("bbox", {})
        if isinstance(raw, dict) and all(key in raw for key in ("l", "t", "r", "b")):
            box = [float(raw["l"]), float(raw["t"]), float(raw["r"]), float(raw["b"])]
    return page, box


def _docling_items(raw_dir: Path, document_stem: str) -> list[_Item] | None:
    path = raw_dir / "document.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ParserRunError(f"Unexpected Docling document shape in {path}.")
    texts = payload.get("texts", [])
    pictures = {index: entry for index, entry in enumerate(payload.get("pictures", []))}
    tables = {index: entry for index, entry in enumerate(payload.get("tables", []))}
    groups = {index: entry for index, entry in enumerate(payload.get("groups", []))}
    body = payload.get("body", {})
    children = body.get("children", []) if isinstance(body, dict) else []
    items: list[_Item] = []
    last_numbered: int | None = None

    def emit_text(entry: dict) -> None:
        nonlocal last_numbered
        label = entry.get("label")
        text = _clean(str(entry.get("text", "")))
        if not text:
            return
        page, box = _docling_box(entry.get("prov", []))
        level = entry.get("level")
        level_value = int(level) if isinstance(level, int) else None
        if label == "section_header":
            items.append(_Item("heading", text, level_value or 1, page, box))
            last_numbered = None
        elif label == "list_item":
            # Docling strips the bullet marker natively; the marker is
            # re-added so list joining/stripping stays consistent.
            items.append(_Item("list", f"- {text}", None, page, box))
        else:
            kind, tentative, number = _classify_text_line(text, last_numbered)
            items.append(_Item(kind, text, None, page, box, tentative, number))
            last_numbered = number if number is not None else last_numbered

    def emit_picture(entry: dict) -> None:
        labels: list[str] = []
        for child in entry.get("children", []):
            resolved = _docling_ref(child.get("$ref", "")) if isinstance(child, dict) else None
            if resolved and resolved[0] == "texts" and resolved[1] < len(texts):
                child_text = _clean(str(texts[resolved[1]].get("text", "")))
                if child_text:
                    labels.append(child_text)
        prov = entry.get("prov", [])
        page, box = _docling_box(prov if isinstance(prov, list) else [])
        if labels:
            items.append(_Item("figure", " ".join(labels), None, page, box))

    def emit_table(entry: dict) -> None:
        data = entry.get("data", {})
        cells = data.get("table_cells", []) if isinstance(data, dict) else []
        rows: dict[int, dict[int, str]] = {}
        for cell in cells:
            row = int(cell.get("start_row_offset_idx", 0))
            col = int(cell.get("start_col_offset_idx", 0))
            rows.setdefault(row, {})[col] = _clean(str(cell.get("text", "")))
        ordered = [[rows[row][col] for col in sorted(rows[row])] for row in sorted(rows)]
        text = serialize_table_cells(ordered)
        prov = entry.get("prov", [])
        page, box = _docling_box(prov if isinstance(prov, list) else [])
        if text:
            items.append(_Item("table", text, None, page, box))

    def emit_group(entry: dict, is_last: bool) -> None:
        members: list[dict] = []
        for child in entry.get("children", []):
            resolved = _docling_ref(child.get("$ref", "")) if isinstance(child, dict) else None
            if resolved and resolved[0] == "texts" and resolved[1] < len(texts):
                members.append(texts[resolved[1]])
        if entry.get("label") == "list" or entry.get("name") == "list":
            parts = [_clean(str(member.get("text", ""))) for member in members]
            parts = [part for part in parts if part]
            if not parts:
                return
            first = members[0]
            page, box = _docling_box(first.get("prov", []))
            if len(parts) == 1 and is_last:
                # A lone trailing "list" of one marker-free line is shaped like
                # an endnote/footnote (Docling may drop the numeric marker), so
                # resolve it with the same section-context rules as other
                # ambiguous numbered lines.
                items.append(_Item("paragraph", parts[0], None, page, box, True, 1))
            else:
                items.append(_Item("list", "- " + "; ".join(parts), None, page, box))
        else:
            for member in members:
                emit_text(member)

    if children:
        for position, child in enumerate(children):
            resolved = _docling_ref(child.get("$ref", "")) if isinstance(child, dict) else None
            if not resolved:
                continue
            kind, index = resolved
            is_last = position == len(children) - 1
            if kind == "texts" and index < len(texts):
                emit_text(texts[index])
            elif kind == "pictures" and index in pictures:
                emit_picture(pictures[index])
            elif kind == "tables" and index in tables:
                emit_table(tables[index])
            elif kind == "groups" and index in groups:
                emit_group(groups[index], is_last)
    else:
        for entry in texts:
            emit_text(entry)
    return _finalize(items, document_stem)


def _markdown_items(markdown: str) -> list[_Item]:
    """Best-effort Markdown fallback when native JSON is unavailable."""
    items: list[_Item] = []
    paragraph: list[str] = []
    table: list[str] = []
    last_numbered: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph, last_numbered
        if paragraph:
            text = _clean(" ".join(paragraph))
            kind, tentative, number = _classify_text_line(text, last_numbered)
            items.append(_Item(kind, text, None, 1, None, tentative, number))
            last_numbered = number if number is not None else last_numbered
            paragraph = []

    def flush_table() -> None:
        nonlocal table, last_numbered
        if table:
            text = _clean(" ".join(table))
            if text:
                items.append(_Item("table", text, None, 1, None))
            table = []
            last_numbered = None

    for raw in markdown.splitlines():
        line = _unescape_markdown(raw.strip())
        if not line:
            flush_paragraph()
            flush_table()
            continue
        match = HEADING.match(line)
        if match:
            flush_paragraph()
            flush_table()
            items.append(_Item("heading", _clean(match.group(2)), len(match.group(1)), 1, None))
            last_numbered = None
            continue
        if line.startswith("![") or line.startswith("<img") or line.startswith("<div"):
            flush_paragraph()
            if line.startswith("!["):
                items.append(_Item("figure", "", None, 1, None))
            elif "figure" in line.lower() or "fig." in line.lower():
                caption, remainder = split_caption(re.sub(r"<[^>]+>", " ", line))
                if _clean(caption):
                    items.append(_Item("caption", caption, None, 1, None))
                if remainder:
                    items.append(_Item("paragraph", remainder, None, 1, None))
            else:
                items.append(_Item("paragraph", _clean(re.sub(r"<[^>]+>", " ", line)), None, 1, None))
            last_numbered = None
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            if not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line):
                table.append(line)
            continue
        flush_table()
        kind, tentative, number = _classify_text_line(line, last_numbered)
        if kind == "paragraph":
            paragraph.append(line)
            last_numbered = number if number is not None else last_numbered
        else:
            flush_paragraph()
            if kind == "caption":
                caption, remainder = split_caption(line)
                items.append(_Item("caption", caption, None, 1, None))
                if remainder:
                    paragraph.append(remainder)
            else:
                items.append(_Item(kind, line, None, 1, None))
            last_numbered = number if number is not None else last_numbered
    flush_paragraph()
    flush_table()
    return items


def adapt_parser_output(raw_dir: Path, parser_name: str, document_name: str) -> list[Block]:
    """Build canonical blocks, preferring each parser's native JSON export."""
    builders = {"paddle": _paddle_items, "mineru": _miner_items, "docling": _docling_items}
    builder = builders.get(parser_name)
    if builder is not None:
        try:
            blocks = builder(raw_dir, document_name)
        except ParserRunError:
            raise
        except Exception as exc:
            raise ParserRunError(f"Could not adapt {parser_name} output in {raw_dir}: {exc}") from exc
        if blocks:
            return blocks
    markdown = raw_dir.joinpath("benchmark.md")
    if not markdown.exists():
        raise FileNotFoundError(f"Expected parser Markdown at {markdown}")
    return _finalize(_markdown_items(markdown.read_text(encoding="utf-8")), document_name)


def blocks_from_markdown(markdown: str, document_name: str) -> list[Block]:
    """Convert a usable Markdown export into the small shared benchmark format."""
    return _finalize(_markdown_items(markdown), document_name)