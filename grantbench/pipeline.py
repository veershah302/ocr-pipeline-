from __future__ import annotations

from pathlib import Path

from .adapters import adapt_parser_output
from .chunkers import chunk
from .config import Settings
from .evaluate import evaluate_document, write_metrics
from .fixtures import FIXTURES, generate_fixtures
from .io import read_json, write_json
from .models import Block, Chunk
from .parsers import ParserRunError, get_parser
from .report import generate_report


def expected_documents() -> list[str]:
    return [fixture.name for fixture in FIXTURES]


def documents(settings: Settings) -> list[Path]:
    scanned = sorted((settings.data_dir / "scanned").glob("*.pdf"))
    if settings.documents:
        selected = set(settings.documents)
        return [path for path in scanned if path.stem in selected]
    return scanned


def generate_data(settings: Settings) -> list[Path]:
    return generate_fixtures(settings.data_dir)


def parse(settings: Settings, parser_name: str) -> dict[str, str]:
    parser = get_parser(parser_name)
    result: dict[str, str] = {}
    for document in documents(settings):
        raw_dir = settings.output_dir / "raw" / parser_name / document.stem
        normalized_path = settings.output_dir / "normalized" / parser_name / f"{document.stem}.json"
        failure_path = raw_dir / "failure.json"
        try:
            parser.parse(document, raw_dir, settings.device)
            blocks = adapt_parser_output(raw_dir, parser_name, document.stem)
            write_json(normalized_path, [block.to_dict() for block in blocks])
            failure_path.unlink(missing_ok=True)
            result[document.stem] = "ok"
        except (ParserRunError, FileNotFoundError) as exc:
            normalized_path.unlink(missing_ok=True)
            (raw_dir / "benchmark.md").unlink(missing_ok=True)
            write_json(failure_path, {"parser": parser_name, "document": document.name, "error": str(exc)})
            result[document.stem] = str(exc)
    return result


def run_chunker(settings: Settings, parser_name: str, strategy: str) -> int:
    count = 0
    normalized_dir = settings.output_dir / "normalized" / parser_name
    if not normalized_dir.exists():
        raise FileNotFoundError(f"No normalized output for parser '{parser_name}' at {normalized_dir}. Run `parse` first.")
    for file in sorted(normalized_dir.glob("*.json")):
        blocks = [Block.from_dict(item) for item in read_json(file)]
        chunks = chunk(strategy, blocks, settings.chunk_token_budget)
        write_json(settings.output_dir / "chunks" / parser_name / strategy / file.name, [item.to_dict() for item in chunks])
        count += 1
    return count


def evaluate(settings: Settings) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parser_name in settings.parsers:
        normalized_dir = settings.output_dir / "normalized" / parser_name
        for file in sorted(normalized_dir.glob("*.json")):
            blocks = [Block.from_dict(item) for item in read_json(file)]
            truth = settings.data_dir / "ground_truth" / file.name
            parser_metrics = evaluate_document(truth, blocks)
            run_file = settings.output_dir / "raw" / parser_name / file.stem / "run.json"
            if run_file.exists():
                parser_metrics["duration_seconds"] = float(read_json(run_file).get("duration_seconds", 0.0))
            rows.append({"parser": parser_name, "strategy": "parser", "document": file.stem, **parser_metrics})
            for strategy in settings.chunkers:
                chunk_file = settings.output_dir / "chunks" / parser_name / strategy / file.name
                if not chunk_file.exists():
                    continue
                chunks = [Chunk.from_dict(item) for item in read_json(chunk_file)]
                metrics = evaluate_document(truth, blocks, chunks)
                rows.append({"parser": parser_name, "strategy": strategy, "document": file.stem, **metrics})
    write_metrics(settings.output_dir / "metrics" / "metrics.json", rows)
    return rows


def report(settings: Settings) -> None:
    metrics = settings.output_dir / "metrics" / "metrics.json"
    if not metrics.exists():
        raise FileNotFoundError("No metrics found. Run `python -m grantbench evaluate` first.")
    generate_report(metrics, settings.output_dir / "reports")


def run_all(settings: Settings) -> None:
    if not settings.documents:
        available = {path.stem for path in documents(settings)}
        if available != set(expected_documents()):
            generate_data(settings)
    for parser_name in settings.parsers:
        parse(settings, parser_name)
        for strategy in settings.chunkers:
            run_chunker(settings, parser_name, strategy)
    evaluate(settings)
    report(settings)
