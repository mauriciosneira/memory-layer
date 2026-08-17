from __future__ import annotations
from typing import Literal, TypedDict, Optional


MemoryType = Literal["semantic", "episodic", "procedural"]
MemoryScope = Literal["user", "instance"]


class MemoryRecord(TypedDict, total=False):
    content: str
    type: MemoryType
    created_at: str
    updated_at: str
    session_id: str
    score: Optional[float]