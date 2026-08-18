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
# Namespace segments and memory ids are joined into a single sort key for DynamoDB's
# begins_with prefix queries. \x1f (ASCII unit separator) is used instead of a printable
# character like ":" specifically because it can't appear in tenant ids, emails, or ARNs —
# a ":" join collided with real-world identifiers containing ":" and corrupted round-trips.
_SEP = "\x1f"


def _sort_key(namespace: tuple[str, ...], memory_id: str) -> str:
    return _SEP.join((*namespace[1:], memory_id))


def _sort_key_prefix(namespace_prefix: tuple[str, ...]) -> str:
    rest = namespace_prefix[1:]
    return _SEP.join(rest) + _SEP if rest else ""


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
        resp = self._table.get_item(
            Key={"owner_id": op.namespace[0], "sort_key": _sort_key(op.namespace, op.key)}
        )
        row = resp.get("Item")
        return _row_to_item(row) if row else None

    def _handle_put(self, op: PutOp) -> None:
        pk = {"owner_id": op.namespace[0], "sort_key": _sort_key(op.namespace, op.key)}
        if op.value is None:
            self._table.delete_item(Key=pk)
            return

        now = datetime.now(timezone.utc).isoformat()
        update_expression = (
            "SET #v = :v, #ns = :ns, memory_id = :mid, updated_at = :now, "
            "created_at = if_not_exists(created_at, :now)"
        )
        expression_names: dict[str, str] = {"#v": "value", "#ns": "namespace"}
        expression_values: dict[str, Any] = {
            ":v": json.dumps(op.value),
            ":ns": json.dumps(list(op.namespace)),
            ":mid": op.key,
            ":now": now,
        }

        ttl_seconds = _resolve_ttl_seconds(op)
        expression_names["#ttl"] = "ttl"
        if ttl_seconds is not None:
            update_expression += ", #ttl = :ttl"
            expression_values[":ttl"] = int(datetime.now(timezone.utc).timestamp()) + ttl_seconds
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

        sort_key_prefix = _sort_key_prefix(op.namespace_prefix)
        key_condition = Key("owner_id").eq(op.namespace_prefix[0])
        # DynamoDB rejects begins_with() with an empty-string operand on a key attribute —
        # a top-level-only prefix (e.g. ("user",)) legitimately means "match everything
        # under this partition", which is exactly what omitting the condition does.
        if sort_key_prefix:
            key_condition &= Key("sort_key").begins_with(sort_key_prefix)

        while len(matches) < op.offset + op.limit and pages < _MAX_SEARCH_PAGES:
            query_kwargs: dict[str, Any] = {"KeyConditionExpression": key_condition}
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
                "results for namespace prefix %r — returned results may be incomplete.",
                _MAX_SEARCH_PAGES,
                op.namespace_prefix,
            )

        # The table has no secondary sort order once queries are scoped by begins_with
        # instead of a GSI keyed on created_at — recency ordering is done here instead.
        matches.sort(key=lambda r: r["created_at"], reverse=True)
        return [_row_to_search_item(r) for r in matches[op.offset : op.offset + op.limit]]
