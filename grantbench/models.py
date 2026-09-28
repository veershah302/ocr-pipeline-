from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    id: str
    text: str
    kind: str
    page: int = 1
    heading_level: int | None = None
    heading_path: list[str] = field(default_factory=list)
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Block":
        return cls(**value)


@dataclass
class Chunk:
    id: str
    strategy: str
    text: str
    block_ids: list[str]
    heading_path: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)
