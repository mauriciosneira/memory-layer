from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from .types import MemoryType

_EXTRACTION_PROMPT = """You are a memory extraction assistant. Given a conversation, extract facts worth remembering about the user for future sessions.

Rules:
- Only extract facts that would be useful in a *future* conversation — preferences, recurring patterns, key domain context.
- Ignore facts that are specific to this single request and won't generalise.
- Each fact must be a short, standalone sentence.
- Classify each fact as one of: semantic (preferences/facts about the user), episodic (specific past event), procedural (recurring work pattern).
- If nothing is worth remembering, return no facts."""


class _ExtractedFact(BaseModel):
    content: str
    type: MemoryType


class _ExtractedFacts(BaseModel):
    facts: list[_ExtractedFact]


class MemoryWriter:
    def __init__(self, llm: BaseChatModel, store: BaseStore) -> None:
        self._llm = llm
        self._store = store

    async def extract_and_save(
        self,
        namespace: tuple[str, ...],
        messages: list[BaseMessage],
        session_id: str,
    ) -> int:
        conversation = _format_conversation(messages)
        if not conversation:
            return 0

        extraction_messages = [
            SystemMessage(content=_EXTRACTION_PROMPT),
            HumanMessage(content=conversation),
        ]
        # with_structured_output validates the shape via the provider's own tool-calling —
        # no hand-rolled JSON parsing, no risk of a stray sentence around the array breaking it.
        structured_llm = self._llm.with_structured_output(_ExtractedFacts)
        result: _ExtractedFacts = await structured_llm.ainvoke(extraction_messages)

        for fact in result.facts:
            await self._store.aput(
                namespace,
                str(uuid4()),
                {
                    "content": fact.content,
                    "type": fact.type,
                    "session_id": session_id,
                },
            )

        return len(result.facts)


def _format_conversation(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        if role in ("human", "ai") and msg.content:
            lines.append(f"{role.upper()}: {msg.content}")
    return "\n".join(lines)
