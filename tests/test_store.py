from __future__ import annotations

import time

import pytest

from memory_layer.store import DynamoDBStore


@pytest.fixture
def store(dynamodb_table):
    return DynamoDBStore(table_name="test-memories")


NAMESPACE = ("user", "test-user-123")


def _raw(store, namespace, memory_id):
    from memory_layer.store import _owner_id

    return store._table.get_item(Key={"owner_id": _owner_id(namespace), "sort_key": memory_id})["Item"]


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


def test_search_isolated_by_full_namespace(store):
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


def test_explicit_ttl_accepts_a_float_number_of_minutes(store):
    """Regression test: op.ttl (BaseStore contract) can be a float, and DynamoDB's boto3
    resource layer raises TypeError on a raw Python float — must be coerced to int."""
    store.put(NAMESPACE, "mem-1", {"content": "short-lived note", "type": "semantic"}, ttl=1.5)

    raw = _raw(store, NAMESPACE, "mem-1")
    assert "ttl" in raw
    assert int(raw["ttl"]) <= int(time.time()) + 90 + 1


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


def test_search_recency_is_correct_across_multiple_pages(store):
    """Regression test: recency must be a property of the query (GSI + ScanIndexForward),
    not a client-side re-sort of whatever page(s) happened to be fetched — a re-sort over
    a partial page returns the most recent items *of that page*, not of the whole
    namespace, the moment there's more data than fits in one page."""
    padding = "x" * 40_000
    for i in range(40):
        store.put(NAMESPACE, f"mem-{i:03d}", {"content": padding, "type": "semantic", "seq": i})

    results = store.search(NAMESPACE, limit=5)

    assert [r.value["seq"] for r in results] == [39, 38, 37, 36, 35]


def test_namespace_segments_containing_a_colon_survive_a_round_trip(store):
    namespace = ("instance", "arn:aws:iam::123456789012:role/acme")
    store.put(namespace, "mem-1", {"content": "tenant-scoped fact", "type": "semantic"})

    item = store.get(namespace, "mem-1")
    results = store.search(namespace, limit=10)

    assert item.namespace == namespace
    assert results[0].namespace == namespace


class TestPartitionIsolation:
    """Regression coverage for the hot-partition/cardinality bug: owner_id must be the
    FULL namespace, not just namespace[0] — otherwise every namespace sharing that first
    segment (e.g. every "user" in the whole deployment) collapses into one DynamoDB
    partition, and a principal's own memories can become unrecoverable once enough other,
    unrelated principals under the same first segment have written more recently (Query's
    Limit caps items read from the shared partition before any prefix filter applies)."""

    def test_a_principals_memory_is_found_despite_heavy_traffic_from_unrelated_principals(self, store):
        store.put(("user", "the-one-we-care-about"), "mem-ours", {"content": "important", "type": "semantic"})

        for i in range(150):
            store.put(("user", f"unrelated-{i}"), "mem-noise", {"content": "noise", "type": "semantic"})

        results = store.search(("user", "the-one-we-care-about"), limit=10)

        assert len(results) == 1
        assert results[0].key == "mem-ours"

    def test_search_requires_the_exact_write_time_namespace(self, store):
        """Documents the accepted trade-off: unlike the prior schema, this store does not
        support a namespace_prefix shorter than what was written — it queries a specific,
        different partition and returns nothing, silently. Neither this package's own
        documented usage nor any real caller needs cross-depth prefix search in practice."""
        store.put(("user", "123", "preferences"), "mem-1", {"content": "likes dark mode", "type": "semantic"})

        results = store.search(("user", "123"), limit=10)

        assert results == []
