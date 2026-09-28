# Healthcare Grant OCR and Chunking Benchmark — Explanation

## Purpose

This benchmark compares three OCR/document parsers and four chunking strategies using fictional scanned healthcare-grant PDFs. It separately measures:

- OCR transcription accuracy
- Document-structure preservation
- Reading-order preservation
- Chunk-boundary quality
- Heading-context preservation
- Runtime

The benchmark intentionally does not calculate a single composite score.

## How the code works

### 1. Fixture generation

Six synthetic documents are authored programmatically:

- `single_column`
- `two_column`
- `tables`
- `figure_caption`
- `lists_notes`
- `mixed_appendix`

The generator creates:

```text
data/source/<document>.pdf
data/scanned/<document>.pdf
data/ground_truth/<document>.json
```

Source PDFs contain the authored layout. Scanned PDFs are rasterized image-only PDFs, so parsers must perform genuine OCR/layout analysis. Ground-truth JSON contains the authoritative block sequence, block types, heading hierarchy, and valid chunk boundaries.

### 2. OCR parsing

Each parser saves its untouched native results under:

```text
outputs/raw/<parser>/<document>/
```

A thin per-parser adapter then converts the parser's **native JSON** export into canonical benchmark blocks with:

- `id`
- `text`
- `kind`
- `page`
- `heading_path`
- Optional `bbox`

Markdown is used only as a fallback when native JSON is unavailable, plus PaddleOCR heading depths (which its JSON omits). Footnotes and references use section-aware rules because no parser labels them: incrementing numbered runs stay ordered lists, lone numbers before a new top-level/references section or at end of document become footnotes, and citation-like paragraphs inside references-like sections become references.

This allows identical chunking and evaluation despite the parsers having different native formats.

### 3. Chunking

All chunkers use the same canonical blocks and the configured 180-word token budget.

- **Hierarchical:** creates a new group at a sibling or higher-level heading.
- **Sectional:** groups content beneath each top-level section.
- **Paragraph:** creates one chunk per block and splits only oversized paragraphs.
- **Heading:** creates a new chunk at every heading.

Each chunk retains:

- Its source block IDs
- Its heading path
- Heading ancestry prepended to the chunk text, for example:

```text
Methods > Workflow
Navigator outreach text...
```

## Metric formulas

### Text normalization

Before WER, CER, alignment, and heading-context calculations, text is normalized as:

```python
re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
```

Thus, the metrics ignore case and punctuation differences and focus on alphanumeric content.

### Word error rate

```text
WER = D_words / N_reference_words
```

Where:

- `D_words` is the minimum number of word substitutions, deletions, and insertions needed to convert OCR output into the reference.
- `N_reference_words` is the number of normalized reference words.

Lower is better.

Example:

```text
character_error_rate("clinic", "clinc") == 1 / 6
```

### Character error rate

```text
CER = D_characters / N_reference_characters
```

Where:

- `D_characters` is the minimum number of character substitutions, deletions, and insertions.
- `N_reference_characters` is the number of normalized reference characters.

Lower is better. CER detects finer OCR mistakes than WER, such as one wrong letter inside an otherwise recognized word.

### Reading-order LCS

```text
Reading-order score = length(LCS) / number_of_reference_blocks
```

Where `LCS` is the longest common subsequence between:

1. Ground-truth block IDs in document order.
2. Successfully aligned parser-block IDs in parser output order.

Higher is better. A score of `1.0` means all aligned blocks remained in the correct relative order.

### Block alignment

Parser blocks are matched to ground-truth blocks through:

1. Normalized text comparison.
2. `SequenceMatcher` similarity.
3. A minimum similarity threshold of `0.45`.
4. Greedy one-to-one assignment, highest similarity first.

Each parser block and ground-truth block can be used at most once.

### Block-type macro F1

For every block type present in ground truth:

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2TP / (2TP + FP + FN)
```

Where:

- `TP`: aligned blocks whose parser and ground-truth kinds match.
- `FP`: parser blocks assigned to that kind without a matching ground-truth block.
- `FN`: ground-truth blocks of that kind without a matching parser block.

The reported value is the unweighted mean across all ground-truth block types:

```text
Macro F1 = mean(F1_heading, F1_paragraph, F1_list, ...)
```

Higher is better.

### Semantic-boundary F1

Predicted boundaries are the mapped ground-truth IDs corresponding to the final block in every chunk except the final chunk:

```text
Precision = valid_predicted_boundaries / all_predicted_boundaries
Recall = valid_predicted_boundaries / all_valid_boundaries
Boundary F1 = 2 × Precision × Recall / (Precision + Recall)
```

Higher is better.

### Heading-context retention

For each chunk that can be aligned to ground truth and has a nonempty heading path:

```text
Heading-context retention =
  chunks_containing_all_required_heading_text
  / chunks_requiring_heading_context
```

A chunk retains context when every normalized heading in its applicable `heading_path` appears in its normalized chunk text.

Higher is better.

### Runtime

Parser runtime is wall-clock seconds recorded separately for each document.

Reports show:

```text
Mean seconds per document = total_seconds / number_of_documents
```

and the total seconds across all six fixtures.

## Where the ground-truth scores came from

The ground truth was not downloaded from an external dataset and was not manually annotated after OCR.

It was generated from the authored fixture definitions in:

```text
grantbench/fixtures.py
```

The `FIXTURES` list explicitly defines the source text, block order, block type, and heading level for every fixture. `generate_fixtures()` then:

1. Draws the authored source PDF.
2. Rasterizes it into an image-only scanned PDF.
3. Converts the same authored fixture entries into ground-truth blocks.
4. Builds heading ancestry from heading levels.
5. Derives `valid_boundary_after` using this rule:

```text
Every non-final heading, paragraph, table, list, caption, or reference
is a valid boundary.
```

The resulting ground truth is saved as:

```text
data/ground_truth/<document>.json
```

Therefore, “ground-truth scores” are really comparisons between:

- Text and structure authored before OCR, and
- Text and structure recovered from scanned images.

Parser metric values come from `evaluate_document()` in:

```text
grantbench/evaluate.py
```

and are stored in:

```text
outputs/metrics/metrics.json
outputs/reports/all_metrics.csv
```

## Current results

### OCR results

| Parser | WER ↓ | CER ↓ | Type F1 ↑ | Order LCS ↑ | Seconds/document |
|---|---:|---:|---:|---:|---:|
| MinerU | 0.054 | 0.044 | 0.762 | 0.944 | 11.10 |
| Docling | 0.177 | 0.130 | 0.741 | 0.924 | 19.29 |
| PaddleOCR | 0.318 | 0.183 | 0.632 | 0.944 | 100.49 |

### Chunking results

| Chunker | Boundary F1 ↑ | Heading context ↑ |
|---|---:|---:|
| Hierarchical | 0.402 | 0.870 |
| Sectional | 0.000 | 1.000 |
| Paragraph | 0.957 | 0.873 |
| Heading | 0.506 | 0.889 |

MinerU had the best transcription and structure scores and the lowest runtime. Paragraph chunking best preserved fine-grained boundaries. Sectional chunking preserved heading context completely, but its zero boundary score reflects coarse chunk placement rather than necessarily poor retrieval behavior.

## Charts and visualizations

Manager-ready visuals are saved in:

```text
outputs/reports/
```

### `ocr_comparison.png`

Grouped bars for each parser showing:

- WER
- CER
- Block-type F1
- Reading-order LCS

Best slide for parser accuracy and structure quality.

### `chunking_comparison.png`

Grouped bars for each chunking strategy showing:

- Semantic-boundary F1
- Heading-context retention

Best slide for chunking behavior.

### `parser_chunker_heatmap.png`

Two heatmaps covering every parser–chunker combination:

1. Boundary F1.
2. Heading-context retention.

Best slide for showing whether chunking performance depends on OCR output quality.

### `summary.md`

Contains:

- Parser comparison table
- Chunker comparison table
- Runtime table

### `all_metrics.csv`

Contains all 90 detailed per-document and per-chunker metric rows, including the new CER column.
