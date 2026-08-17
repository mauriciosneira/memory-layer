from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from memory_layer.writer import MemoryWriter, _ExtractedFact, _ExtractedFacts, _format_conversation
from memory_layer.store import DynamoDBStore


NAMESPACE = ("user", "test-user-123")
SESSION_ID = "session-abc"


def _make_llm(facts: list[_ExtractedFact]) -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=_ExtractedFacts(facts=facts))
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


@pytest.fixture
def store(dynamodb_table):
    return DynamoDBStore(table_name="test-memories")


@pytest.mark.asyncio
async def test_extract_and_save_writes_facts(store):
    facts = [
        _ExtractedFact(content="User prefers Spanish", type="semantic"),
        _ExtractedFact(content="User works with the Acme account", type="semantic"),
    ]
    writer = MemoryWriter(llm=_make_llm(facts), store=store)
    messages = [
        HumanMessage(content="Analyze Acme campaigns"),
        AIMessage(content="Here is the analysis..."),
    ]

    count = await writer.extract_and_save(NAMESPACE, messages, SESSION_ID)

    assert count == 2


@pytest.mark.asyncio
async def test_extract_and_save_empty_conversation_skips_llm():
    llm = _make_llm([])
    writer = MemoryWriter(llm=llm, store=AsyncMock())

    count = await writer.extract_and_save(NAMESPACE, [], SESSION_ID)

    assert count == 0
    llm.with_structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_save_empty_facts_writes_nothing():
    store = AsyncMock()
    writer = MemoryWriter(llm=_make_llm([]), store=store)
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]

    count = await writer.extract_and_save(NAMESPACE, messages, SESSION_ID)

    assert count == 0
    store.aput.assert_not_called()


def test_format_conversation_includes_human_and_ai():
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    result = _format_conversation(messages)

    assert "HUMAN: Hello" in result
    assert "AI: Hi there" in result


def test_format_conversation_skips_empty_content():
    messages = [HumanMessage(content=""), AIMessage(content="response")]
    result = _format_conversation(messages)

    assert "HUMAN" not in result
    assert "AI: response" in result


def test_extracted_fact_rejects_an_invalid_type():
    with pytest.raises(ValueError):
        _ExtractedFact(content="Bad", type="invalid_type")
