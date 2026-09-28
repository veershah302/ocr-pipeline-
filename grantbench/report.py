from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .io import read_json, write_text


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _total(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows if key in row)


def _ocr_plot(parser_rows: dict[str, list[dict[str, object]]], reports_dir: Path) -> None:
    names = list(parser_rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(names)))
    series = {
        "WER ↓": [_mean(parser_rows[name], "wer") for name in names],
        "CER ↓": [_mean(parser_rows[name], "cer") for name in names],
        "Type F1 ↑": [_mean(parser_rows[name], "type_macro_f1") for name in names],
        "Order LCS ↑": [_mean(parser_rows[name], "reading_order_lcs") for name in names],
    }
    width = 0.8 / len(series)
    for position, (label, values) in enumerate(series.items()):
        offset = (position - (len(series) - 1) / 2) * width
        ax.bar([i + offset for i in x], values, width, label=label)
    ax.set_xticks(x, names)
    # WER/CER can exceed 1.0 when OCR inserts excess text; scale dynamically.
    ax.set_ylim(0, max(1.05, max(max(values) for values in series.values()) * 1.05))
    ax.legend()
    ax.set_title("OCR / parser comparison")
    fig.tight_layout()
    fig.savefig(reports_dir / "ocr_comparison.png", dpi=160)
    plt.close(fig)


def _chunking_plot(chunk_rows: dict[str, list[dict[str, object]]], reports_dir: Path) -> None:
    names = list(chunk_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(names)))
    ax.bar([i - 0.18 for i in x], [_mean(chunk_rows[name], "boundary_f1") for name in names], 0.36, label="Boundary F1")
    ax.bar([i + 0.18 for i in x], [_mean(chunk_rows[name], "heading_context_retention") for name in names], 0.36, label="Heading context")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Chunking comparison")
    fig.tight_layout()
    fig.savefig(reports_dir / "chunking_comparison.png", dpi=160)
    plt.close(fig)


def _heatmap(matrix: dict[tuple[str, str], list[dict[str, object]]], parsers: list[str], strategies: list[str], reports_dir: Path) -> None:
    metrics = ("boundary_f1", "heading_context_retention")
    titles = ("Boundary F1 ↑", "Heading context retention ↑")
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5), squeeze=False)
    for ax, metric, title in zip(axes[0], metrics, titles, strict=True):
        # Missing parser/chunker combinations stay blank (NaN) instead of
        # rendering as a misleading 0.0.
        values = [
            [
                _mean(matrix[(parser, strategy)], metric) if matrix[(parser, strategy)] else float("nan")
                for strategy in strategies
            ]
            for parser in parsers
        ]
        image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(range(len(strategies)), strategies, rotation=25, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=8)
        ax.set_yticks(range(len(parsers)), parsers)
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                if value == value:  # skip NaN (missing combination) labels
                    ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label=metric)
        ax.set_title(title)
    fig.suptitle("Parser × chunker")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.88)
    fig.savefig(reports_dir / "parser_chunker_heatmap.png", dpi=160)
    plt.close(fig)


def _summary(
    parser_rows: dict[str, list[dict[str, object]]],
    chunk_rows: dict[str, list[dict[str, object]]],
) -> str:
    parsers, strategies = list(parser_rows), list(chunk_rows)
    lines = ["# Benchmark summary", ""]
    if parsers:
        lines += ["| Parser | WER ↓ | CER ↓ | Type F1 ↑ | Order LCS ↑ | Seconds/document |", "|---|---:|---:|---:|---:|---:|"]
        for name in parsers:
            subset = parser_rows[name]
            lines.append(f"| {name} | {_mean(subset, 'wer'):.3f} | {_mean(subset, 'cer'):.3f} | {_mean(subset, 'type_macro_f1'):.3f} | {_mean(subset, 'reading_order_lcs'):.3f} | {_mean(subset, 'duration_seconds'):.2f} |")
    if strategies:
        lines += ["", "| Chunker | Boundary F1 ↑ | Heading context ↑ |", "|---|---:|---:|"]
        for name in strategies:
            subset = chunk_rows[name]
            lines.append(f"| {name} | {_mean(subset, 'boundary_f1'):.3f} | {_mean(subset, 'heading_context_retention'):.3f} |")
    lines += ["", "## Runtime", "", "| Parser | Documents | Seconds/document | Total seconds |", "|---|---:|---:|---:|"]
    for name in parsers:
        subset = parser_rows[name]
        lines.append(f"| {name} | {len(subset)} | {_mean(subset, 'duration_seconds'):.2f} | {_total(subset, 'duration_seconds'):.2f} |")
    return "\n".join(lines) + "\n"


def generate_report(metrics_file: Path, reports_dir: Path) -> None:
    rows: list[dict[str, object]] = read_json(metrics_file)
    if not rows:
        raise ValueError(f"No metric rows in {metrics_file}. Run `parse` and `chunk` before `report`.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    parser_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    chunk_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    matrix: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("strategy") == "parser":
            parser_rows[str(row["parser"])].append(row)
        else:
            chunk_rows[str(row["strategy"])].append(row)
            matrix[(str(row["parser"]), str(row["strategy"]))].append(row)

    if parser_rows:
        _ocr_plot(parser_rows, reports_dir)
    if chunk_rows:
        _chunking_plot(chunk_rows, reports_dir)
    if parser_rows and chunk_rows:
        _heatmap(matrix, sorted(parser_rows), sorted(chunk_rows), reports_dir)
    write_text(reports_dir / "summary.md", _summary(parser_rows, chunk_rows))
