from __future__ import annotations

from collections.abc import Iterable

from .models import Block, Chunk


def token_count(text: str) -> int:
    return len(text.split())


def _make(strategy: str, index: int, blocks: list[Block]) -> Chunk:
    heading_path = blocks[-1].heading_path if blocks else []
    context = " > ".join(heading_path)
    body = "\n".join(block.text for block in blocks)
    text = f"{context}\n{body}".strip() if context else body
    return Chunk(f"{strategy}-{index:04d}", strategy, text, [block.id for block in blocks], heading_path)


def _split_to_budget(strategy: str, groups: Iterable[list[Block]], budget: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for group in groups:
        current: list[Block] = []
        for block in group:
            if current and token_count(" ".join(item.text for item in current + [block])) > budget:
                chunks.append(_make(strategy, len(chunks), current))
                current = []
            # Split only oversized text blocks; retain a traceable synthetic block id.
            if not current and token_count(block.text) > budget and block.kind == "paragraph":
                words = block.text.split()
                for start in range(0, len(words), budget):
                    piece = Block(block.id, " ".join(words[start : start + budget]), block.kind, block.page, block.heading_level, block.heading_path)
                    chunks.append(_make(strategy, len(chunks), [piece]))
            else:
                current.append(block)
        if current:
            chunks.append(_make(strategy, len(chunks), current))
    return chunks


def hierarchical(blocks: list[Block], budget: int) -> list[Chunk]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_level: int | None = None
    for block in blocks:
        if block.kind == "heading":
            if current and (current_level is None or (block.heading_level or 9) <= current_level):
                groups.append(current)
                current = []
            current_level = block.heading_level
        current.append(block)
    if current:
        groups.append(current)
    return _split_to_budget("hierarchical", groups, budget)


def sectional(blocks: list[Block], budget: int) -> list[Chunk]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    for block in blocks:
        if block.kind == "heading" and block.heading_level == 1 and current:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)
    return _split_to_budget("sectional", groups, budget)


def paragraph(blocks: list[Block], budget: int) -> list[Chunk]:
    return _split_to_budget("paragraph", ([block] for block in blocks), budget)


def heading(blocks: list[Block], budget: int) -> list[Chunk]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    for block in blocks:
        if block.kind == "heading" and current:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)
    return _split_to_budget("heading", groups, budget)


CHUNKERS = {"hierarchical": hierarchical, "sectional": sectional, "paragraph": paragraph, "heading": heading}


def chunk(strategy: str, blocks: list[Block], budget: int) -> list[Chunk]:
    try:
        return CHUNKERS[strategy](blocks, budget)
    except KeyError as exc:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose one of: {', '.join(CHUNKERS)}") from exc
