from __future__ import annotations

from langgraph.store.base import BaseStore, SearchItem


class SimpleRetrieval:
    """Fetches N most recent memories by recency. No embeddings required."""

    def __init__(self, store: BaseStore, limit: int = 10) -> None:
        self._store = store
        self._limit = limit

    def fetch(
        self,
        namespace: tuple[str, ...],
        *,
        memory_type: str | None = None,
    ) -> list[SearchItem]:
        filter_ = {"type": memory_type} if memory_type else None
        return self._store.search(namespace, filter=filter_, limit=self._limit)

    async def afetch(
        self,
        namespace: tuple[str, ...],
        *,
        memory_type: str | None = None,
    ) -> list[SearchItem]:
        filter_ = {"type": memory_type} if memory_type else None
        return await self._store.asearch(namespace, filter=filter_, limit=self._limit)

    def to_prompt_block(self, memories: list[SearchItem]) -> str:
        if not memories:
            return ""
        lines = "\n".join(f"- {m.value.get('content', '')}" for m in memories)
        return f"## User memory\n{lines}"
