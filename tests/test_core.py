from pathlib import Path

import pymupdf
import pytest

from grantbench import pipeline
from grantbench.adapters import adapt_parser_output, blocks_from_markdown
from grantbench.chunkers import chunk
from grantbench.config import Settings, load_settings
from grantbench.evaluate import boundary_f1, character_error_rate, context_retention, lcs_ratio, macro_type_f1, word_error_rate
from grantbench.fixtures import generate_fixtures
from grantbench.io import read_json, write_json, write_text
from grantbench.models import Block, Chunk
from grantbench.parsers import ParserRunError
from grantbench.report import generate_report


def test_markdown_adapter_preserves_heading_path() -> None:
    blocks = blocks_from_markdown("# Title\n\n## Methods\n\nStudy text.", "doc")
    assert [block.kind for block in blocks] == ["heading", "heading", "paragraph"]
    assert blocks[-1].heading_path == ["Title", "Methods"]


def test_markdown_adapter_detects_canonical_block_kinds() -> None:
    markdown = (
        "# Title\n\n"
        "An introductory paragraph.\n\n"
        "| Year | Total |\n\n"
        "- First bullet\n\n"
        "Figure 1. A caption describing the chart.\n\n"
        "## References\n\n"
        "Smith J, Patel R. Community navigation outcomes. Journal of Primary Care. 2024;18:101-110.\n"
    )
    blocks = blocks_from_markdown(markdown, "doc")
    assert [block.kind for block in blocks] == ["heading", "paragraph", "table", "list", "caption", "heading", "reference"]
    assert blocks[-1].heading_path == ["References"]


def _hierarchy_blocks() -> list[Block]:
    return [
        Block("h1", "Alpha", "heading", 1, 1, ["Alpha"]),
        Block("p1", "First supporting paragraph.", "paragraph", 1, None, ["Alpha"]),
        Block("h2", "Beta", "heading", 1, 2, ["Alpha", "Beta"]),
        Block("p2", "Second supporting paragraph.", "paragraph", 1, None, ["Alpha", "Beta"]),
        Block("h3", "Gamma", "heading", 1, 1, ["Gamma"]),
        Block("p3", "Third supporting paragraph.", "paragraph", 1, None, ["Gamma"]),
    ]


def test_chunkers_respect_heading_boundaries() -> None:
    blocks = _hierarchy_blocks()
    hierarchical = chunk("hierarchical", blocks, 100)
    sectional = chunk("sectional", blocks, 100)
    heading = chunk("heading", blocks, 100)
    paragraph = chunk("paragraph", blocks, 100)

    # Sibling headings start a new hierarchical or top-level sectional group.
    assert [[block for block in item.block_ids] for item in hierarchical] == [["h1", "p1", "h2", "p2"], ["h3", "p3"]]
    assert [[block for block in item.block_ids] for item in sectional] == [["h1", "p1", "h2", "p2"], ["h3", "p3"]]
    # A new group starts at every heading for the heading strategy, and at every
    # block for the paragraph strategy.
    assert [[block for block in item.block_ids] for item in heading] == [["h1", "p1"], ["h2", "p2"], ["h3", "p3"]]
    assert [[block for block in item.block_ids] for item in paragraph] == [["h1"], ["p1"], ["h2"], ["p2"], ["h3"], ["p3"]]


def test_chunkers_propagate_heading_context() -> None:
    hierarchical = chunk("hierarchical", _hierarchy_blocks(), 100)
    assert hierarchical[0].heading_path == ["Alpha", "Beta"]
    assert hierarchical[0].text.splitlines()[0] == "Alpha > Beta"
    assert hierarchical[1].heading_path == ["Gamma"]


def test_paragraph_chunker_splits_oversized_paragraph() -> None:
    block = Block("p1", "one two three four five", "paragraph", heading_path=["Methods"])
    chunks = chunk("paragraph", [block], 2)
    assert len(chunks) == 3
    assert all(item.heading_path == ["Methods"] for item in chunks)


def test_paragraph_chunker_does_not_split_other_oversized_blocks() -> None:
    block = Block("table1", "one two three four five", "table", heading_path=["Budget"])
    chunks = chunk("paragraph", [block], 2)
    assert [item.block_ids for item in chunks] == [["table1"]]


def test_unknown_chunker_reports_available_strategies() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk("unknown", [], 100)


def test_word_error_rate_and_lcs() -> None:
    assert word_error_rate("a b c", "a b c") == 0
    assert word_error_rate("a b", "a x") == 0.5
    assert lcs_ratio(["a", "b", "c"], ["a", "c"]) == 2 / 3


def test_character_error_rate_counts_character_edits() -> None:
    assert character_error_rate("clinic", "clinic") == 0
    assert character_error_rate("clinic", "clinc") == 1 / 6
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "clinic") == 1.0


def _matching_blocks() -> tuple[list[Block], list[Block], dict[str, str]]:
    expected = [
        Block("truth-heading", "Methods", "heading", 1, 1, ["Methods"]),
        Block("truth-body", "Study text.", "paragraph", 1, None, ["Methods"]),
    ]
    actual = [
        Block("output-heading", "Methods", "heading", 1, 1, ["Methods"]),
        Block("output-body", "Study text.", "paragraph", 1, None, ["Methods"]),
    ]
    mapping = {"output-heading": "truth-heading", "output-body": "truth-body"}
    return expected, actual, mapping


def test_macro_type_f1_scores_matched_and_mismatched_kinds() -> None:
    expected, actual, mapping = _matching_blocks()
    assert macro_type_f1(expected, actual, mapping) == 1.0

    mismatched = [Block("output-body", "Study text.", "paragraph", 1, None, ["Methods"])]
    assert macro_type_f1([expected[0]], mismatched, {"output-body": "truth-heading"}) == 0.0


def _boundary_chunks() -> tuple[list[Chunk], dict[str, str]]:
    chunks = [
        Chunk("chunk-1", "paragraph", "first", ["output-1"], []),
        Chunk("chunk-2", "paragraph", "second", ["output-2"], []),
        Chunk("chunk-3", "paragraph", "third", ["output-3"], []),
    ]
    mapping = {"output-1": "truth-1", "output-2": "truth-2", "output-3": "truth-3"}
    return chunks, mapping


def test_boundary_f1_scores_predicted_chunk_ends() -> None:
    chunks, mapping = _boundary_chunks()
    assert boundary_f1({"truth-1", "truth-2"}, chunks, mapping) == 1.0
    assert boundary_f1({"missing"}, chunks, mapping) == 0.0


def test_heading_context_retention_requires_heading_path_text() -> None:
    expected = [Block("truth-final", "Navigator outreach text.", "paragraph", 1, None, ["Methods", "Workflow"])]
    mapping = {"output-final": "truth-final"}
    retained = [Chunk("chunk-1", "hierarchical", "Methods > Workflow\nNavigator outreach text.", ["output-final"], ["Methods", "Workflow"])]
    assert context_retention(retained, expected, mapping) == 1.0


def test_heading_context_retention_rejects_wrong_or_missing_paths() -> None:
    expected = [Block("truth-final", "Navigator outreach text.", "paragraph", 1, None, ["Methods", "Workflow"])]
    mapping = {"output-final": "truth-final"}
    mismatched = [Chunk("chunk-1", "hierarchical", "Methods > Workflow\nNavigator outreach text.", ["output-final"], ["Methods"])]
    assert context_retention(mismatched, expected, mapping) == 0.0
    # Without structured metadata the metric falls back to heading text.
    text_only = [Chunk("chunk-1", "hierarchical", "Methods > Workflow\nNavigator outreach text.", ["output-final"], [])]
    assert context_retention(text_only, expected, mapping) == 1.0
    dropped = [Chunk("chunk-1", "hierarchical", "Navigator outreach text.", ["output-final"], [])]
    assert context_retention(dropped, expected, mapping) == 0.0


def test_generate_fixtures_preserve_ground_truth_and_image_only_pdfs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scanned = generate_fixtures(data_dir)
    assert {path.stem for path in scanned} == {"single_column", "two_column", "tables", "figure_caption", "lists_notes", "mixed_appendix"}

    truth = read_json(data_dir / "ground_truth" / "single_column.json")
    block_ids = {block["id"] for block in truth["blocks"]}
    assert truth["blocks"][0]["kind"] == "heading"
    assert truth["valid_boundary_after"]
    assert set(truth["valid_boundary_after"]) <= block_ids

    for pdf in scanned:
        document = pymupdf.open(pdf)
        try:
            assert len(document) >= 1
            assert all(not page.get_text().strip() for page in document)
        finally:
            document.close()


def test_report_rejects_empty_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    write_json(metrics, [])
    with pytest.raises(ValueError, match="No metric rows"):
        generate_report(metrics, tmp_path / "reports")


class _StubParser:
    name = "stub"
    fail_on = "two_column"

    def parse(self, document: Path, raw_dir: Path, device: str) -> dict[str, str]:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if document.stem == self.fail_on:
            raise ParserRunError("simulated failure")
        write_text(raw_dir / "benchmark.md", f"# Stub {document.stem}\n\nBody.")
        return {"parser": self.name, "device": device}


def test_parse_isolates_parser_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    generate_fixtures(data_dir)
    settings = Settings(
        root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "outputs",
        device="cpu",
        chunk_token_budget=180,
        parsers=["stub"],
        chunkers=[],
        documents=[],
    )
    monkeypatch.setattr(pipeline, "get_parser", lambda name: _StubParser() if name == "stub" else (_ for _ in ()).throw(ValueError(name)))

    outcomes = pipeline.parse(settings, "stub")

    assert outcomes["single_column"] == "ok"
    assert outcomes["two_column"] == "simulated failure"
    assert (settings.output_dir / "normalized" / "stub" / "single_column.json").exists()
    assert not (settings.output_dir / "normalized" / "stub" / "two_column.json").exists()
    failure = read_json(settings.output_dir / "raw" / "stub" / "two_column" / "failure.json")
    assert failure == {"parser": "stub", "document": "two_column.pdf", "error": "simulated failure"}


def test_paddle_adapter_uses_native_labels_and_bboxes(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "paddle" / "doc"
    raw.mkdir(parents=True)
    write_json(raw / "doc_0_res.json", {"page_index": 0, "parsing_res_list": [
        {"block_label": "paragraph_title", "block_content": "Title", "block_bbox": [0, 0, 10, 10], "block_id": 0, "block_order": 1},
        {"block_label": "text", "block_content": "- item one", "block_bbox": [0, 0, 10, 10], "block_id": 1, "block_order": 2},
        {"block_label": "text", "block_content": "1. A trailing note.", "block_bbox": [0, 0, 10, 10], "block_id": 2, "block_order": 3},
        {"block_label": "image", "block_content": "", "block_bbox": [0, 0, 10, 10], "block_id": 3, "block_order": None},
    ]})
    write_text(raw / "doc_0.md", "# Title\n")
    blocks = adapt_parser_output(raw, "paddle", "doc")
    assert [block.kind for block in blocks] == ["heading", "list", "footnote"]
    assert blocks[0].heading_level == 1
    assert blocks[0].bbox == [0.0, 0.0, 10.0, 10.0]
    assert blocks[0].page == 1


def test_mineru_adapter_prefers_middle_json_spans(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "mineru" / "doc"
    raw.mkdir(parents=True)
    write_json(raw / "middle_json.json", {"pages": [{"page_idx": 0, "blocks": [
        {"type": "doc_title", "level": 1, "bbox": [0, 0, 1, 1], "content": [{"type": "text", "content": "Title"}]},
        {"type": "text", "level": None, "bbox": [0, 0, 1, 1], "content": [{"type": "text", "content": "\\- escaped bullet"}]},
        {"type": "table", "level": None, "bbox": [0, 0, 1, 1], "content": [{"type": "table_body", "content": "<table><tr><td>A</td><td>B</td></tr></table>"}]},
        {"type": "image", "level": None, "bbox": [0, 0, 1, 1], "content": [
            {"type": "image_body", "content": ""},
            {"type": "image_caption", "content": [{"type": "text", "content": "Figure 1. A caption."}]},
        ]},
    ]}]})
    blocks = adapt_parser_output(raw, "mineru", "doc")
    assert [block.kind for block in blocks] == ["heading", "list", "table", "caption"]
    assert blocks[1].text == "escaped bullet"
    assert blocks[2].text == "A | B"


def _docling_text(label: str, text: str, level: int | None = None) -> dict:
    prov = [{"page_no": 1, "bbox": {"l": 0, "t": 0, "r": 10, "b": 10}}]
    return {"label": label, "text": text, "level": level, "prov": prov}


def test_docling_adapter_follows_body_order_and_groups(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "docling" / "doc"
    raw.mkdir(parents=True)
    texts = [
        _docling_text("section_header", "Title", 1),
        _docling_text("text", "Body text."),
        _docling_text("list_item", "One"),
        _docling_text("list_item", "Two"),
        _docling_text("text", "1. An end note."),
        _docling_text("section_header", "References", 1),
        _docling_text("text", "Smith J, Patel R. Outcomes. Journal. 2024;18:101."),
    ]
    write_json(raw / "document.json", {
        "texts": texts,
        "pictures": [],
        "tables": [],
        "groups": [{"label": "list", "children": [{"$ref": "#/texts/2"}, {"$ref": "#/texts/3"}]}],
        "body": {"children": [
            {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/groups/0"},
            {"$ref": "#/texts/4"}, {"$ref": "#/texts/5"}, {"$ref": "#/texts/6"},
        ]},
    })
    blocks = adapt_parser_output(raw, "docling", "doc")
    assert [block.kind for block in blocks] == ["heading", "paragraph", "list", "footnote", "heading", "reference"]
    assert blocks[2].text == "One; Two"
    assert blocks[-1].heading_path == ["References"]


def test_ordered_list_sequence_stays_list() -> None:
    blocks = blocks_from_markdown("# Title\n\n1. First step\n\n2. Second step\n\n## Next\n", "doc")
    assert [block.kind for block in blocks] == ["heading", "list", "heading"]
    assert blocks[1].text == "First step; Second step"


def test_lone_number_before_references_is_footnote() -> None:
    blocks = blocks_from_markdown("# Title\n\n1. A note about sources.\n\n## References\n", "doc")
    assert [block.kind for block in blocks] == ["heading", "footnote", "heading"]


def test_caption_split_preserves_trailing_paragraph() -> None:
    blocks = blocks_from_markdown("Figure 1. A caption. Trailing body text.", "doc")
    assert [(block.kind, block.text) for block in blocks] == [("caption", "Figure 1. A caption."), ("paragraph", "Trailing body text.")]


def test_heading_levels_normalize_and_references_resets() -> None:
    blocks = blocks_from_markdown("## Title\n\n### Sub\n\n## References\n", "doc")
    assert [(block.kind, block.heading_level) for block in blocks] == [("heading", 1), ("heading", 2), ("heading", 1)]
    assert blocks[1].heading_path == ["Title", "Sub"]
    assert blocks[2].heading_path == ["References"]


def test_ground_truth_figure_uses_rendered_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_fixtures(data_dir)
    truth = read_json(data_dir / "ground_truth" / "figure_caption.json")
    figure = next(block for block in truth["blocks"] if block["kind"] == "figure")
    assert "Referral" in figure["text"]
    assert "Workflow diagram showing" not in figure["text"]


def test_adapt_falls_back_to_markdown_and_reports_missing(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "paddle" / "doc"
    raw.mkdir(parents=True)
    write_text(raw / "benchmark.md", "# Hi\n")
    assert [block.kind for block in adapt_parser_output(raw, "paddle", "doc")] == ["heading"]
    with pytest.raises(FileNotFoundError, match="Expected parser Markdown"):
        adapt_parser_output(tmp_path / "raw" / "paddle" / "absent", "paddle", "absent")


def _stub_settings(tmp_path: Path, **overrides: object) -> Settings:
    fields: dict[str, object] = {
        "root": tmp_path,
        "data_dir": tmp_path / "data",
        "output_dir": tmp_path / "outputs",
        "device": "cpu",
        "chunk_token_budget": 180,
        "parsers": ["stub"],
        "chunkers": [],
        "documents": [],
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def test_run_chunker_requires_normalized_output(tmp_path: Path) -> None:
    settings = _stub_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="No normalized output"):
        pipeline.run_chunker(settings, "paddle", "paragraph")


def test_config_rejects_unknown_parser_and_bad_budget(tmp_path: Path) -> None:
    bad_parser = tmp_path / "bad.toml"
    write_text(bad_parser, '[paths]\ndata_dir = "data"\noutput_dir = "outputs"\n[benchmark]\ndevice = "cpu"\nparsers = ["nope"]\nchunkers = ["paragraph"]\n')
    with pytest.raises(ValueError, match="Unknown parser"):
        load_settings(bad_parser)
    bad_budget = tmp_path / "budget.toml"
    write_text(bad_budget, '[paths]\ndata_dir = "data"\noutput_dir = "outputs"\n[benchmark]\ndevice = "cpu"\nchunk_token_budget = 0\nparsers = ["paddle"]\nchunkers = ["paragraph"]\n')
    with pytest.raises(ValueError, match="must be positive"):
        load_settings(bad_budget)
    with pytest.raises(FileNotFoundError, match="Could not read config"):
        load_settings(tmp_path / "missing.toml")


def test_config_documents_filter_selects_fixtures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    generate_fixtures(data_dir)
    settings = _stub_settings(tmp_path, documents=["tables"])
    assert [path.stem for path in pipeline.documents(settings)] == ["tables"]

