from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .chunkers import CHUNKERS
from .parsers import PARSERS


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    output_dir: Path
    device: str
    chunk_token_budget: int
    parsers: list[str]
    chunkers: list[str]
    documents: list[str]


def load_settings(config_path: str | Path = "config.toml") -> Settings:
    config = Path(config_path).resolve()
    try:
        with config.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read config file at {config}: {exc}") from exc
    try:
        benchmark = raw["benchmark"]
        parsers = list(benchmark["parsers"])
        chunkers = list(benchmark["chunkers"])
    except KeyError as exc:
        raise ValueError(f"Config {config} is missing required key: {exc}") from exc
    unknown_parsers = [name for name in parsers if name not in PARSERS]
    if unknown_parsers:
        raise ValueError(f"Unknown parser(s) {unknown_parsers} in {config}. Choose from: {', '.join(PARSERS)}")
    unknown_chunkers = [name for name in chunkers if name not in CHUNKERS]
    if unknown_chunkers:
        raise ValueError(f"Unknown chunking strateg(ies) {unknown_chunkers} in {config}. Choose from: {', '.join(CHUNKERS)}")
    try:
        budget = int(benchmark.get("chunk_token_budget", 180))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid chunk_token_budget in {config}: {exc}") from exc
    if budget <= 0:
        raise ValueError(f"chunk_token_budget in {config} must be positive, got {budget}.")
    root = config.parent
    try:
        paths = raw["paths"]
        data_dir = root / paths["data_dir"]
        output_dir = root / paths["output_dir"]
    except KeyError as exc:
        raise ValueError(f"Config {config} is missing required key: {exc}") from exc
    return Settings(
        root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        device=benchmark.get("device", "cpu"),
        chunk_token_budget=budget,
        parsers=parsers,
        chunkers=chunkers,
        documents=list(benchmark.get("documents", [])),
    )
