from __future__ import annotations

import time

import pytest

from memory_layer.store import DynamoDBStore


@pytest.fixture
def store(dynamodb_table):
    return DynamoDBStore(table_name="test-memories")


NAMESPACE = ("user", "test-user-123")


def _raw(store, namespace, memory_id):
    from memory_layer.store import _sort_key

    return store._table.get_item(Key={"owner_id": namespace[0], "sort_key": _sort_key(namespace, memory_id)})["Item"]


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


def test_search_isolated_by_top_level_namespace(store):
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

    raw = _raw(store, NAMESPACE, "mem-1")
    assert "ttl" in raw
    assert int(raw["ttl"]) > int(time.time())


def test_put_semantic_sets_no_ttl(store):
    store.put(NAMESPACE, "mem-1", {"content": "prefers Spanish", "type": "semantic"})

    raw = _raw(store, NAMESPACE, "mem-1")
    assert "ttl" not in raw


def test_changing_type_away_from_episodic_removes_the_stale_ttl(store):
    store.put(NAMESPACE, "mem-1", {"content": "went to the store", "type": "episodic"})
    store.put(NAMESPACE, "mem-1", {"content": "went to the store", "type": "semantic"})

    raw = _raw(store, NAMESPACE, "mem-1")
    assert "ttl" not in raw


def test_explicit_ttl_minutes_overrides_the_type_default(store):
    store.put(NAMESPACE, "mem-1", {"content": "short-lived note", "type": "semantic"}, ttl=5)

    raw = _raw(store, NAMESPACE, "mem-1")
    assert "ttl" in raw
    assert int(raw["ttl"]) <= int(time.time()) + 5 * 60 + 1


def test_search_paginates_past_the_first_page_when_filter_thins_it_out(store):
    for i in range(15):
        memory_type = "semantic" if i == 14 else "episodic"
        store.put(NAMESPACE, f"mem-{i}", {"content": f"memory {i}", "type": memory_type})

    results = store.search(NAMESPACE, filter={"type": "semantic"}, limit=1)

    assert len(results) == 1
    assert results[0].value["type"] == "semantic"


def test_list_namespaces_raises_not_implemented(store):
    from langgraph.store.base import ListNamespacesOp

    with pytest.raises(NotImplementedError):
        store.batch([ListNamespacesOp(match_conditions=None, max_depth=None, limit=100, offset=0)])


def test_search_with_query_raises_not_implemented(store):
    store.put(NAMESPACE, "mem-1", {"content": "something", "type": "semantic"})

    with pytest.raises(NotImplementedError):
        store.search(NAMESPACE, query="something relevant", limit=10)


class TestNamespacePrefixSearch:
    """BaseStore's own contract documents namespace_prefix as a hierarchical *prefix* —
    searching by a shorter namespace than what memories were stored under must still
    find them. An exact-match implementation would return an empty list here, silently."""

    def test_search_by_a_shorter_prefix_finds_deeper_namespaces(self, store):
        store.put(("user", "123", "preferences"), "mem-1", {"content": "likes dark mode", "type": "semantic"})

        results = store.search(("user", "123"), limit=10)

        assert len(results) == 1
        assert results[0].namespace == ("user", "123", "preferences")

    def test_search_by_top_level_prefix_finds_everything_under_it(self, store):
        store.put(("user", "123"), "mem-1", {"content": "a", "type": "semantic"})
        store.put(("user", "123", "preferences"), "mem-2", {"content": "b", "type": "semantic"})
        store.put(("user", "456"), "mem-3", {"content": "c", "type": "semantic"})

        results = store.search(("user",), limit=10)

        assert {r.key for r in results} == {"mem-1", "mem-2", "mem-3"}

    def test_sibling_ids_sharing_a_numeric_prefix_do_not_false_match(self, store):
        store.put(("user", "123"), "mem-1", {"content": "a", "type": "semantic"})
        store.put(("user", "1234"), "mem-2", {"content": "b", "type": "semantic"})

        results = store.search(("user", "123"), limit=10)

        assert {r.key for r in results} == {"mem-1"}

    def test_namespace_segments_containing_the_join_separator_survive_a_round_trip(self, store):
        namespace = ("instance", "arn:aws:iam::123456789012:role/acme")
        store.put(namespace, "mem-1", {"content": "tenant-scoped fact", "type": "semantic"})

        item = store.get(namespace, "mem-1")
        results = store.search(("instance",), limit=10)

        assert item.namespace == namespace
        assert results[0].namespace == namespace
