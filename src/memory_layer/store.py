from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import boto3
from boto3.dynamodb.conditions import Key
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)

_MAX_SEARCH_PAGES = 10
_TTL_SECONDS_BY_TYPE = {"episodic": 90 * 24 * 60 * 60}  # semantic/procedural: no default ttl, never expire
_RECENCY_INDEX_NAME = "owner_id-created_at-index"
# owner_id is the FULL namespace, joined with \x1f (not ":" — real identifiers like ARNs or
# emails can contain ":", which would corrupt the round-trip). This makes owner_id one
# partition per distinct namespace (e.g. per end user), not per namespace[0] — a prior design
# partitioned on namespace[0] alone to support search() with a shorter-than-write namespace
# prefix, which collapsed every namespace sharing that first segment (e.g. every "user" in the
# whole deployment) into a single DynamoDB partition. Confirmed empirically: a principal's own
# memory became unrecoverable once ~150 other principals under the same first segment had
# written more recently (Query's Limit caps items *read* from the shared partition before the
# prefix filter is even applied). Neither this package's own documented usage (`("user", id)`,
# search always run with the exact same namespace as the write) nor Atlas's real call sites
# ever search with a namespace shorter than what was written — so this trades an unused,
# dangerous capability (hierarchical prefix search across depths) for correctness and
# cardinality on the path that's actually exercised. See MEMORY_LAYER.md for the full history.
_SEP = "\x1f"


def _owner_id(namespace: tuple[str, ...]) -> str:
    return _SEP.join(namespace)


def _gsi_sort_key(created_at: str, memory_id: str) -> str:
    # created_at alone isn't unique — two writes in the same millisecond would tie and
    # leave their relative order to chance. Appending memory_id breaks the tie deterministically.
    return f"{created_at}{_SEP}{memory_id}"


def _row_to_item(row: dict[str, Any]) -> Item:
    return Item(
        namespace=tuple(json.loads(row["namespace"])),
        key=row["memory_id"],
        value=json.loads(row["value"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_search_item(row: dict[str, Any]) -> SearchItem:
    return SearchItem(
        namespace=tuple(json.loads(row["namespace"])),
        key=row["memory_id"],
        value=json.loads(row["value"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        score=None,
    )


def _matches_filter(value: dict[str, Any], filter: dict[str, Any]) -> bool:
    for field, condition in filter.items():
        field_val = value.get(field)
        if isinstance(condition, dict):
            for op, operand in condition.items():
                if op == "$eq" and field_val != operand:
                    return False
                elif op == "$ne" and field_val == operand:
                    return False
                elif op == "$gt" and not (field_val is not None and field_val > operand):
                    return False
                elif op == "$gte" and not (field_val is not None and field_val >= operand):
                    return False
                elif op == "$lt" and not (field_val is not None and field_val < operand):
                    return False
                elif op == "$lte" and not (field_val is not None and field_val <= operand):
                    return False
        elif field_val != condition:
            return False
    return True


def _resolve_ttl_seconds(op: PutOp) -> float | None:
    """An explicit numeric ttl (op.ttl, in minutes — the BaseStore contract) always wins
    over the type-based default. NOT_PROVIDED and None both fall back to that default —
    BaseStore.put()'s own _ensure_ttl() collapses "caller didn't pass ttl" into ttl=None
    before a PutOp is ever constructed (confirmed against the installed langgraph
    source), so by the time it reaches here there is no way to distinguish that from an
    explicit ttl=None short of bypassing put() and constructing a PutOp directly."""
    if isinstance(op.ttl, (int, float)):
        return op.ttl * 60
    return _TTL_SECONDS_BY_TYPE.get((op.value or {}).get("type"))


class DynamoDBStore(BaseStore):
    # Without this, BaseStore.put()'s own guard rejects any explicit ttl= argument with
    # NotImplementedError before a PutOp is even constructed.
    supports_ttl = True

    def __init__(
        self,
        table_name: str | None = None,
        *,
        resource: Any = None,
    ) -> None:
        """
        Args:
            table_name: Defaults to the MEMORY_TABLE env var (read here, not at import
                time, so setting it after import still works), falling back to
                "memory-layer-local-memories".
            resource: A pre-configured `boto3.resource("dynamodb", ...)` — pass this to
                target a specific region, endpoint_url (e.g. DynamoDB Local), or session
                instead of relying on boto3's default credential/region resolution.
        """
        resolved_table_name = table_name or os.environ.get("MEMORY_TABLE", "memory-layer-local-memories")
        self._table = (resource or boto3.resource("dynamodb")).Table(resolved_table_name)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                results.append(self._handle_put(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                raise NotImplementedError(
                    "DynamoDBStore does not implement list_namespaces — returning an "
                    "empty/partial result set silently would be worse than failing loudly."
                )
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    def _handle_get(self, op: GetOp) -> Item | None:
        resp = self._table.get_item(Key={"owner_id": _owner_id(op.namespace), "sort_key": op.key})
        row = resp.get("Item")
        return _row_to_item(row) if row else None

    def _handle_put(self, op: PutOp) -> None:
        pk = {"owner_id": _owner_id(op.namespace), "sort_key": op.key}
        if op.value is None:
            self._table.delete_item(Key=pk)
            return

        now = datetime.now(timezone.utc).isoformat()
        update_expression = (
            "SET #v = :v, #ns = :ns, memory_id = :mid, updated_at = :now, "
            "created_at = if_not_exists(created_at, :now), "
            "gsi_sort_key = if_not_exists(gsi_sort_key, :gsk)"
        )
        expression_names: dict[str, str] = {"#v": "value", "#ns": "namespace"}
        expression_values: dict[str, Any] = {
            ":v": json.dumps(op.value),
            ":ns": json.dumps(list(op.namespace)),
            ":mid": op.key,
            ":now": now,
            # Only takes effect on first write (if_not_exists) — an update to an existing
            # memory keeps its original creation-time position in the recency index.
            ":gsk": _gsi_sort_key(now, op.key),
        }

        ttl_seconds = _resolve_ttl_seconds(op)
        expression_names["#ttl"] = "ttl"
        if ttl_seconds is not None:
            update_expression += ", #ttl = :ttl"
            # DynamoDB's boto3 resource layer rejects Python float values outright
            # ("Float types are not supported") — op.ttl (minutes) can be a float per the
            # BaseStore contract, so ttl_seconds can be too (e.g. ttl=1.5 -> 90.0 seconds).
            expression_values[":ttl"] = int(datetime.now(timezone.utc).timestamp() + ttl_seconds)
        else:
            # Without this, a memory that had a ttl set (e.g. created as "episodic") and
            # later updated to a type with no default ttl (e.g. "semantic") would keep
            # the stale ttl attribute forever — DynamoDB would still silently delete it.
            update_expression += " REMOVE #ttl"

        self._table.update_item(
            Key=pk,
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values,
        )

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """namespace_prefix must be the exact namespace items were written under — this store
        does not support searching with a namespace shorter than the write-time namespace (see
        the module-level comment on _SEP for why). A shorter prefix queries a partition where
        nothing was written and returns an empty list, not an error — there is no way to tell
        "caller searched a bare/exact single-segment namespace" from "caller intended a
        hierarchical prefix over a deeper namespace" from the input alone."""
        if op.query is not None:
            raise NotImplementedError(
                "DynamoDBStore does not implement semantic search (SearchOp.query) — "
                "silently falling back to recency order would misrepresent the results "
                "as relevance-ranked. Use SimpleRetrieval, or the semantic-retrieval "
                "extra once available, instead."
            )

        matches: list[dict[str, Any]] = []
        last_evaluated_key: dict[str, Any] | None = None
        pages = 0

        # Recency (ScanIndexForward=False on gsi_sort_key) is a server-side property of the
        # query, not a client-side re-sort — a re-sort only orders whatever page(s) were
        # already fetched, which silently stops being "the N most recent" once a namespace
        # has enough memories to span more than one page.
        query_kwargs_base: dict[str, Any] = {
            "IndexName": _RECENCY_INDEX_NAME,
            "KeyConditionExpression": Key("owner_id").eq(_owner_id(op.namespace_prefix)),
            "ScanIndexForward": False,
        }

        while len(matches) < op.offset + op.limit and pages < _MAX_SEARCH_PAGES:
            query_kwargs = dict(query_kwargs_base)
            if last_evaluated_key is not None:
                query_kwargs["ExclusiveStartKey"] = last_evaluated_key

            resp = self._table.query(**query_kwargs)
            rows = resp.get("Items", [])
            pages += 1

            if op.filter:
                rows = [r for r in rows if _matches_filter(json.loads(r["value"]), op.filter)]
            matches.extend(rows)

            last_evaluated_key = resp.get("LastEvaluatedKey")
            if last_evaluated_key is None:
                break

        if last_evaluated_key is not None and pages >= _MAX_SEARCH_PAGES:
            logger.warning(
                "DynamoDBStore.search hit _MAX_SEARCH_PAGES (%d) before exhausting "
                "results for namespace %r — returned results may be incomplete.",
                _MAX_SEARCH_PAGES,
                op.namespace_prefix,
            )

        return [_row_to_search_item(r) for r in matches[op.offset : op.offset + op.limit]]
