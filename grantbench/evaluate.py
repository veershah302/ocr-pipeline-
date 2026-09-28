from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from .io import read_json, write_json
from .models import Block, Chunk


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, candidate in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1] / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, character in enumerate(ref, 1):
        current = [i]
        for j, candidate in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (character != candidate)))
        previous = current
    return previous[-1] / len(ref)


def lcs_ratio(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 1.0
    dp = [0] * (len(hypothesis) + 1)
    for left in reference:
        diagonal = 0
        for j, right in enumerate(hypothesis, 1):
            saved = dp[j]
            if left == right:
                dp[j] = diagonal + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            diagonal = saved
    return dp[-1] / len(reference)


def align_blocks(expected: list[Block], actual: list[Block]) -> dict[str, str]:
    """Greedy one-to-one text alignment for compact, inspectable POC metrics."""
    candidates: list[tuple[float, str, str]] = []
    for output in actual:
        output_text = normalize(output.text)
        for truth in expected:
            truth_text = normalize(truth.text)
            score = SequenceMatcher(None, output_text, truth_text).ratio()
            if score >= 0.45:
                candidates.append((score, output.id, truth.id))
    used_out: set[str] = set()
    used_truth: set[str] = set()
    mapping: dict[str, str] = {}
    for _, output_id, truth_id in sorted(candidates, reverse=True):
        if output_id not in used_out and truth_id not in used_truth:
            mapping[output_id] = truth_id
            used_out.add(output_id)
            used_truth.add(truth_id)
    return mapping


def macro_type_f1(expected: list[Block], actual: list[Block], mapping: dict[str, str]) -> float:
    truth_by_id, actual_by_id = {block.id: block for block in expected}, {block.id: block for block in actual}
    kinds = sorted({block.kind for block in expected})
    scores: list[float] = []
    for kind in kinds:
        tp = sum(1 for output_id, truth_id in mapping.items() if actual_by_id[output_id].kind == kind and truth_by_id[truth_id].kind == kind)
        fp = sum(1 for block in actual if block.kind == kind) - tp
        fn = sum(1 for block in expected if block.kind == kind) - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 1.0)
    return sum(scores) / len(scores) if scores else 1.0


def boundary_f1(expected_boundaries: set[str], chunks: list[Chunk], mapping: dict[str, str]) -> float:
    predicted = {mapping[chunk.block_ids[-1]] for chunk in chunks[:-1] if chunk.block_ids and chunk.block_ids[-1] in mapping}
    tp = len(expected_boundaries & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected_boundaries) if expected_boundaries else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def context_retention(chunks: list[Chunk], expected: list[Block], mapping: dict[str, str]) -> float:
    """Check that each chunk's propagated heading path matches ground truth.

    Chunkers explicitly propagate ``heading_path`` metadata, so the primary
    check compares that metadata against the aligned ground-truth path. A
    substring fallback covers chunks whose structured path was lost but whose
    text still names every required heading.
    """
    by_id = {block.id: block for block in expected}
    total, retained = 0, 0
    for chunk in chunks:
        truth_ids = [mapping[block_id] for block_id in chunk.block_ids if block_id in mapping]
        if not truth_ids:
            continue
        required = [normalize(title) for title in by_id[truth_ids[-1]].heading_path]
        if not required:
            continue
        total += 1
        propagated = [normalize(title) for title in chunk.heading_path]
        if propagated == required:
            retained += 1
        elif not propagated and all(title in normalize(chunk.text) for title in required):
            retained += 1
    return retained / total if total else 1.0


def evaluate_document(truth_path: Path, blocks: list[Block], chunks: list[Chunk] | None = None) -> dict[str, float]:
    raw = read_json(truth_path)
    expected = [Block.from_dict(item) for item in raw["blocks"]]
    mapping = align_blocks(expected, blocks)
    metrics = {
        "wer": word_error_rate(" ".join(block.text for block in expected), " ".join(block.text for block in blocks)),
        "cer": character_error_rate(" ".join(block.text for block in expected), " ".join(block.text for block in blocks)),
        "type_macro_f1": macro_type_f1(expected, blocks, mapping),
        "reading_order_lcs": lcs_ratio([block.id for block in expected], [mapping[block.id] for block in blocks if block.id in mapping]),
        "aligned_blocks": float(len(mapping)),
    }
    if chunks is not None:
        metrics["boundary_f1"] = boundary_f1(set(raw["valid_boundary_after"]), chunks, mapping)
        metrics["heading_context_retention"] = context_retention(chunks, expected, mapping)
    return metrics


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    write_json(path, rows)
