# Healthcare Grant OCR & Chunking POC

A small, CPU-first benchmark for comparing PaddleOCR PP-StructureV3, MinerU 4.x, and Docling on fictional scanned healthcare-grant documents. It measures document text/structure preservation and chunk boundary/context preservation; it deliberately does **not** evaluate retrieval.

## Setup

PaddlePaddle currently has no Windows wheel for Python 3.14, so use Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The three parser dependencies are heavy and download model weights on first use. If a tool cannot be installed or initialized, runs for the other tools still complete and the failure is recorded in `outputs/`.

## Commands

```powershell
python -m pytest -q
python -m grantbench generate-data
python -m grantbench parse --parser docling
python -m grantbench chunk --parser docling --strategy hierarchical
python -m grantbench evaluate
python -m grantbench report
python -m grantbench run-all
```

Use `--config config.toml` to point at another configuration. `run-all` processes every configured parser and chunker sequentially on CPU.

## Output layout

- `data/source/`: authored PDFs used only to create fixtures
- `data/scanned/`: image-only PDFs sent to parsers
- `data/ground_truth/`: ordered semantic blocks and valid chunk boundaries
- `outputs/raw/<parser>/<document>/`: untouched native JSON/Markdown/assets
- `outputs/normalized/<parser>/<document>.json`: benchmark blocks
- `outputs/chunks/<parser>/<strategy>/<document>.json`
- `outputs/metrics/metrics.json`: detailed per-document metrics
- `outputs/reports/`: CSV detail, Markdown summary/runtime tables, and PNG charts

## Comparison design

Native parser JSON formats remain untouched. Each parser has a thin native-JSON adapter that creates the common `Block` list (text, kind, page, optional box, heading path):

- PaddleOCR `*_res.json` block labels, order, page index, and bounding boxes (heading depths come from the sibling Markdown export, which the native JSON lacks);
- MinerU `middle_json.json` block types, levels, normalized boxes, and clean content spans (the bundled Markdown escapes list markers, so JSON is preferred);
- Docling `document.json` labels, levels, page/box provenance, picture children, table cells, and list groups, following `body.children` reading order.

A hardened Markdown adapter remains only as a fallback when native JSON is missing. Footnotes and references are resolved with section context because no parser labels them explicitly: incrementing numbered runs stay ordered lists, lone numbers before a new top-level/references section (or at end of document) become footnotes, and citation-like paragraphs inside references-like sections become references. Structure metrics therefore describe usable parser output, not private JSON schemas, but block typing now comes from native labels rather than Markdown regexes.

Metrics are reported separately, not collapsed into a score:

- word error rate (WER), character error rate (CER), block-type macro F1, and reading-order LCS for parsing;
- semantic-boundary F1 and heading-context retention for chunking;
- wall-clock duration when a parser run succeeds.

## Notes

The data is fictional and contains no PHI. The default baseline is CPU; tool/version/configuration metadata is saved with every parser run. Review MinerU's license before using it outside this evaluation.
