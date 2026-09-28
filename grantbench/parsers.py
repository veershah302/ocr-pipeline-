from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .io import write_json, write_text


class ParserRunError(RuntimeError):
    """A parser dependency, model, or invocation failed."""


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class Parser(ABC):
    name: str

    @abstractmethod
    def parse(self, document: Path, raw_dir: Path, device: str) -> dict[str, Any]: ...

    def _finish(self, raw_dir: Path, started: float, metadata: dict[str, Any]) -> dict[str, Any]:
        metadata["duration_seconds"] = round(time.perf_counter() - started, 3)
        write_json(raw_dir / "run.json", metadata)
        return metadata


class PaddleParser(Parser):
    name = "paddle"
    # PaddlePaddle 3.3.x regressed oneDNN/PIR CPU inference (PaddlePaddle#77340), so the CPU
    # baseline runs the plain Paddle kernel path instead of oneDNN.
    disable_mkldnn = True

    def __init__(self) -> None:
        # One CLI invocation parses every fixture with the same instance, so
        # the heavy pipeline is built once and reused across documents.
        self._pipeline: Any = None
        self._pipeline_key: tuple[object, ...] | None = None

    def _get_pipeline(self, device: str) -> Any:
        from paddleocr import PPStructureV3

        key = (device, self.disable_mkldnn)
        if self._pipeline is None or self._pipeline_key != key:
            self._pipeline = PPStructureV3(lang="en", device=device, use_doc_orientation_classify=False, use_doc_unwarping=False)
            self._pipeline_key = key
        return self._pipeline

    def parse(self, document: Path, raw_dir: Path, device: str) -> dict[str, Any]:
        started = time.perf_counter()
        if self.disable_mkldnn and device == "cpu":
            os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        try:
            pipeline = self._get_pipeline(device)
        except ImportError as exc:
            raise ParserRunError("PaddleOCR is unavailable. Install requirements.txt with Python 3.12.") from exc
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            results = list(pipeline.predict(input=str(document)))
            markdown_pages: list[str] = []
            for result in results:
                result.save_to_json(save_path=str(raw_dir))
                result.save_to_markdown(save_path=str(raw_dir))
                payload = result.json if isinstance(getattr(result, "json", None), dict) else None
                if payload:
                    markdown_pages.append(str(payload.get("res", {}).get("markdown", "")))
            markdown_files = [path for path in sorted(raw_dir.rglob("*.md")) if path.name != "benchmark.md"]
            if markdown_files:
                markdown = "\n\n".join(path.read_text(encoding="utf-8") for path in markdown_files)
            else:
                markdown = "\n\n".join(page for page in markdown_pages if page)
            if not markdown.strip():
                raise ParserRunError("PaddleOCR produced no Markdown export.")
            write_text(raw_dir / "benchmark.md", markdown)
            return self._finish(raw_dir, started, {"parser": self.name, "distribution_version": _version("paddleocr"), "paddle_version": _version("paddlepaddle"), "device": device, "input": str(document), "options": {"lang": "en", "use_doc_orientation_classify": False, "use_doc_unwarping": False, "mkldnn_enabled": not (self.disable_mkldnn and device == "cpu")}})
        except Exception as exc:
            self._finish(raw_dir, started, {"parser": self.name, "error": str(exc), "device": device})
            raise ParserRunError(f"PaddleOCR failed for {document.name}: {exc}") from exc


class MinerUParser(Parser):
    name = "mineru"

    def parse(self, document: Path, raw_dir: Path, device: str) -> dict[str, Any]:
        started = time.perf_counter()
        executable = shutil.which("mineru-kit")
        if executable is None:
            raise ParserRunError("mineru-kit is unavailable. Install MinerU 4.x from requirements.txt.")
        raw_dir.mkdir(parents=True, exist_ok=True)
        # mineru-kit exposes no device/tier flags; the local default pipeline
        # is used and the exact invocation is recorded in run.json. Scanned
        # fixtures are image-only, so explicit OCR mode is unnecessary, but all
        # pages are requested explicitly for reproducibility.
        command = [executable, "parse", str(document), "-o", str(raw_dir), "--format", "zip", "--pages", "all"]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        write_text(raw_dir / "command.stdout.txt", stdout)
        write_text(raw_dir / "command.stderr.txt", stderr)
        if completed.returncode != 0:
            self._finish(raw_dir, started, {"parser": self.name, "error": stderr[-1000:], "command": command, "device": device, "output_encoding": "utf-8/errors=replace"})
            raise ParserRunError(f"MinerU failed for {document.name}; see command.stderr.txt")
        markdown_files = [path for path in raw_dir.rglob("*.md") if path.name != "benchmark.md"]
        if not markdown_files:
            # The MinerU 4.x zip bundle contains the native markdown, JSON, and image assets.
            # Extract it untouched so those files are retained alongside the zip archive.
            zip_path = raw_dir / f"{document.stem}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(raw_dir)
                markdown_files = [path for path in raw_dir.rglob("*.md") if path.name != "benchmark.md"]
            else:
                available = sorted(path.name for path in raw_dir.rglob("*.zip"))
                raise ParserRunError(f"MinerU produced no {zip_path.name}; found zip archives: {available}")
        if not markdown_files:
            raise ParserRunError("MinerU completed but no Markdown was materialized; use a MinerU 4.x build supporting --format zip.")
        write_text(raw_dir / "benchmark.md", "\n\n".join(path.read_text(encoding="utf-8") for path in sorted(markdown_files)))
        return self._finish(raw_dir, started, {"parser": self.name, "distribution_version": _version("mineru"), "device": device, "input": str(document), "command": command})


class DoclingParser(Parser):
    name = "docling"

    def parse(self, document: Path, raw_dir: Path, device: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ParserRunError("Docling is unavailable. Install requirements.txt with Python 3.12.") from exc
        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            options = PdfPipelineOptions()
            options.do_ocr = True
            options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU if device == "cpu" else AcceleratorDevice.AUTO)
            converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
            result = converter.convert(document)
            write_json(raw_dir / "document.json", result.document.export_to_dict())
            write_text(raw_dir / "benchmark.md", result.document.export_to_markdown())
            return self._finish(raw_dir, started, {"parser": self.name, "distribution_version": _version("docling"), "device": device, "input": str(document), "do_ocr": True})
        except Exception as exc:
            self._finish(raw_dir, started, {"parser": self.name, "error": str(exc), "device": device})
            raise ParserRunError(f"Docling failed for {document.name}: {exc}") from exc


PARSERS: dict[str, type[Parser]] = {"paddle": PaddleParser, "mineru": MinerUParser, "docling": DoclingParser}


def get_parser(name: str) -> Parser:
    try:
        return PARSERS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown parser '{name}'. Choose one of: {', '.join(PARSERS)}") from exc
