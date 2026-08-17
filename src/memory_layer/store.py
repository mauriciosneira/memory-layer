from __future__ import annotations

import asyncio
import json
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

_TABLE_NAME = os.environ.get("MEMORY_TABLE", "memory-layer-local-memories")
_MAX_SEARCH_PAGES = 10
_TTL_SECONDS_BY_TYPE = {"episodic": 90 * 24 * 60 * 60}  # semantic/procedural: no ttl attribute, never expire


def _owner_id(namespace: tuple[str, ...]) -> str:
    return ":".join(namespace)


def _parse_namespace(owner_id: str) -> tuple[str, ...]:
    return tuple(owner_id.split(":"))


def _row_to_item(row: dict[str, Any]) -> Item:
    return Item(
        namespace=_parse_namespace(row["owner_id"]),
        key=row["memory_id"],
        value=json.loads(row["value"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_search_item(row: dict[str, Any]) -> SearchItem:
    return SearchItem(
        namespace=_parse_namespace(row["owner_id"]),
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


class DynamoDBStore(BaseStore):
    def __init__(self, table_name: str = _TABLE_NAME) -> None:
        self._table = boto3.resource("dynamodb").Table(table_name)

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
                results.append([])
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    def _handle_get(self, op: GetOp) -> Item | None:
        resp = self._table.get_item(
            Key={"owner_id": _owner_id(op.namespace), "memory_id": op.key}
        )
        row = resp.get("Item")
        return _row_to_item(row) if row else None

    def _handle_put(self, op: PutOp) -> None:
        pk = {"owner_id": _owner_id(op.namespace), "memory_id": op.key}
        if op.value is None:
            self._table.delete_item(Key=pk)
            return

        now = datetime.now(timezone.utc).isoformat()
        update_expression = (
            "SET #v = :v, updated_at = :now, "
            "created_at = if_not_exists(created_at, :now)"
        )
        expression_values: dict[str, Any] = {":v": json.dumps(op.value), ":now": now}

        ttl_seconds = _TTL_SECONDS_BY_TYPE.get(op.value.get("type"))
        if ttl_seconds is not None:
            update_expression += ", #ttl = :ttl"
            expression_values[":ttl"] = int(datetime.now(timezone.utc).timestamp()) + ttl_seconds

        self._table.update_item(
            Key=pk,
            UpdateExpression=update_expression,
            ExpressionAttributeNames={"#v": "value", "#ttl": "ttl"} if ttl_seconds is not None else {"#v": "value"},
            ExpressionAttributeValues=expression_values,
        )

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        matches: list[dict[str, Any]] = []
        last_evaluated_key: dict[str, Any] | None = None
        pages = 0

        while len(matches) < op.offset + op.limit and pages < _MAX_SEARCH_PAGES:
            query_kwargs: dict[str, Any] = {
                "IndexName": "owner_id-created_at-index",
                "KeyConditionExpression": Key("owner_id").eq(_owner_id(op.namespace_prefix)),
                "ScanIndexForward": False,
            }
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

        return [_row_to_search_item(r) for r in matches[op.offset : op.offset + op.limit]]
