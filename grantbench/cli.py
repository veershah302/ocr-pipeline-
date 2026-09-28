from __future__ import annotations

import argparse
from pathlib import Path

from .chunkers import CHUNKERS
from .config import load_settings
from .parsers import PARSERS
from .pipeline import evaluate, generate_data, parse, report, run_all, run_chunker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Healthcare grant OCR and chunking benchmark")
    parser.add_argument("--config", default="config.toml", help="Path to TOML configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate-data", help="Generate fictional scanned PDF fixtures and ground truth")
    parse_command = commands.add_parser("parse", help="Run one parser")
    parse_command.add_argument("--parser", choices=PARSERS, required=True)
    chunk_command = commands.add_parser("chunk", help="Run one chunking strategy on one parser output")
    chunk_command.add_argument("--parser", choices=PARSERS, required=True)
    chunk_command.add_argument("--strategy", choices=CHUNKERS, required=True)
    commands.add_parser("evaluate", help="Calculate available metrics")
    commands.add_parser("report", help="Create CSV, Markdown tables, and plots")
    commands.add_parser("run-all", help="Generate data if needed and run the complete benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config))
    if args.command == "generate-data":
        print(f"Generated {len(generate_data(settings))} scanned fixtures.")
    elif args.command == "parse":
        outcomes = parse(settings, args.parser)
        print(outcomes)
    elif args.command == "chunk":
        print(f"Wrote chunks for {run_chunker(settings, args.parser, args.strategy)} documents.")
    elif args.command == "evaluate":
        print(f"Wrote {len(evaluate(settings))} metric rows.")
    elif args.command == "report":
        report(settings); print("Wrote report files.")
    else:
        run_all(settings); print("Benchmark complete; inspect outputs/reports/summary.md")
    return 0
