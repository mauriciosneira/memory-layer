from __future__ import annotations

import time

import pytest

from memory_layer.store import DynamoDBStore


@pytest.fixture
def store(dynamodb_table):
    return DynamoDBStore(table_name="test-memories")


NAMESPACE = ("user", "test-user-123")


def test_put_and_get(store):
    store.put(NAMESPACE, "mem-1", {"content": "User prefers Spanish", "type": "semantic"})
    item = store.get(NAMESPACE, "mem-1")

    assert item is not None
    assert item.key == "mem-1"
    assert item.namespace == NAMESPACE
    assert item.value["content"] == "User prefers Spanish"
    assert item.created_at is not None
    assert item.updated_at is not None


def test_put_preserves_created_at_on_update(store):
    store.put(NAMESPACE, "mem-1", {"content": "original", "type": "semantic"})
    item_before = store.get(NAMESPACE, "mem-1")

    store.put(NAMESPACE, "mem-1", {"content": "updated", "type": "semantic"})
    item_after = store.get(NAMESPACE, "mem-1")

    assert item_after.created_at == item_before.created_at
    assert item_after.value["content"] == "updated"


def test_put_with_none_deletes_item(store):
    store.put(NAMESPACE, "mem-1", {"content": "to be deleted", "type": "semantic"})
    store.put(NAMESPACE, "mem-1", None)

    assert store.get(NAMESPACE, "mem-1") is None


def test_get_missing_returns_none(store):
    assert store.get(NAMESPACE, "nonexistent") is None


def test_search_returns_items(store):
    store.put(NAMESPACE, "mem-1", {"content": "first", "type": "semantic"})
    store.put(NAMESPACE, "mem-2", {"content": "second", "type": "episodic"})
    store.put(NAMESPACE, "mem-3", {"content": "third", "type": "semantic"})

    results = store.search(NAMESPACE, limit=10)

    assert len(results) == 3
    keys = {r.key for r in results}
    assert {"mem-1", "mem-2", "mem-3"} == keys


def test_search_with_type_filter(store):
    store.put(NAMESPACE, "mem-1", {"content": "preference", "type": "semantic"})
    store.put(NAMESPACE, "mem-2", {"content": "past event", "type": "episodic"})

    results = store.search(NAMESPACE, filter={"type": "semantic"}, limit=10)

    assert len(results) == 1
    assert results[0].value["type"] == "semantic"


def test_search_limit_and_offset(store):
    for i in range(5):
        store.put(NAMESPACE, f"mem-{i}", {"content": f"memory {i}", "type": "semantic"})

    page1 = store.search(NAMESPACE, limit=2, offset=0)
    page2 = store.search(NAMESPACE, limit=2, offset=2)

    assert len(page1) == 2
    assert len(page2) == 2
    assert {r.key for r in page1}.isdisjoint({r.key for r in page2})


def test_search_isolated_by_namespace(store):
    other_namespace = ("user", "other-user")
    store.put(NAMESPACE, "mem-1", {"content": "my memory", "type": "semantic"})
    store.put(other_namespace, "mem-2", {"content": "their memory", "type": "semantic"})

    results = store.search(NAMESPACE, limit=10)

    assert len(results) == 1
    assert results[0].key == "mem-1"


def test_unsupported_op_raises_type_error(store):
    class UnknownOp:
        pass

    with pytest.raises(TypeError, match="Unsupported op type"):
        store.batch([UnknownOp()])


def test_put_episodic_sets_a_ttl(store):
    store.put(NAMESPACE, "mem-1", {"content": "went to the store", "type": "episodic"})

    raw = store._table.get_item(Key={"owner_id": ":".join(NAMESPACE), "memory_id": "mem-1"})["Item"]
    assert "ttl" in raw
    assert int(raw["ttl"]) > int(time.time())


def test_put_semantic_sets_no_ttl(store):
    store.put(NAMESPACE, "mem-1", {"content": "prefers Spanish", "type": "semantic"})

    raw = store._table.get_item(Key={"owner_id": ":".join(NAMESPACE), "memory_id": "mem-1"})["Item"]
    assert "ttl" not in raw


def test_search_paginates_past_the_first_page_when_filter_thins_it_out(store):
    for i in range(15):
        memory_type = "semantic" if i == 14 else "episodic"
        store.put(NAMESPACE, f"mem-{i}", {"content": f"memory {i}", "type": memory_type})

    results = store.search(NAMESPACE, filter={"type": "semantic"}, limit=1)

    assert len(results) == 1
    assert results[0].value["type"] == "semantic"
