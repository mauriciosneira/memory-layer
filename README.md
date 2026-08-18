# memory-layer

[![Tests](https://github.com/mauriciosneira/memory-layer/actions/workflows/test.yml/badge.svg)](https://github.com/mauriciosneira/memory-layer/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A DynamoDB-backed `BaseStore` for LangGraph — cross-thread memory for your agents without standing up Postgres/pgvector.**

LangGraph's checkpointer persists state *inside* a thread — when a user opens a new conversation, the graph starts from zero. The `store` protocol is LangGraph's answer to that: memory that survives *across* threads, shared by every session for the same user (or the same tenant). LangGraph ships an official store for Postgres. If your stack is already DynamoDB — which a lot of serverless/Fargate deployments are — there wasn't an official option. `memory-layer` is that option.

```python
from memory_layer import DynamoDBStore

store = DynamoDBStore(table_name="my-app-memories")
graph = builder.compile(checkpointer=checkpointer, store=store)
```

That's the whole integration. No new infra beyond one DynamoDB table you probably already know how to provision.

---

## Why this exists

- **You're already on DynamoDB.** Adding Postgres + pgvector just for agent memory is a real infra cost — a new engine, a new backup story, a new thing to monitor — for a feature that, for most products, doesn't need vector search on day one.
- **LangGraph's `store` protocol is a clean seam.** It's designed so storage is swappable — your agent code shouldn't care whether memories live in Postgres, Dynamo, or Redis. This fills the Dynamo gap in that seam.
- **Memory doesn't have to mean embeddings.** Most products get real value from "the last N things we know about this user," fetched by recency — no vector index required. `memory-layer` starts there, and gives you a clean place to add semantic ranking later if you actually need it.

## Features

- **`DynamoDBStore`** — a complete `langgraph.store.base.BaseStore` implementation: `get`/`put`/`search` (and their async counterparts), real pagination via DynamoDB's `LastEvaluatedKey` (not a "hope the first page has enough" heuristic), per-item filtering, and per-type TTL (e.g. auto-expire `episodic` memories after 90 days while `semantic` ones never expire).
- **`SimpleRetrieval`** — fetch a user's N most recent memories and turn them into a ready-to-inject prompt block. No embeddings, no extra dependencies.
- **`MemoryWriter`** — an LLM-driven extraction step: hand it a conversation, it classifies and persists the facts worth remembering, via `with_structured_output` (a real schema-validated response, not a hand-rolled JSON parser hoping the model didn't wrap the array in a sentence).
- **Scopes, not just users.** Namespaces are plain tuples (`("user", user_id)`, `("instance", tenant_id)`) — model per-user memory, per-tenant shared context, or your own scope, however your product actually shapes ownership.

## Install

```bash
pip install langgraph-dynamodb-store
```

Want LLM-driven extraction? `MemoryWriter` takes any LangChain `BaseChatModel` — bring the one you already use, no extra install needed. `numpy`/`langchain-openai` are only required for the semantic-retrieval extra (see [Roadmap](#roadmap)):

```bash
pip install "langgraph-dynamodb-store[semantic]"
```

## DynamoDB table

One table, one GSI. The base table's `sort_key` (namespace + memory id, `begins_with`-friendly) makes a namespace prefix like `("user", "123")` correctly match everything stored under it (including deeper namespaces like `("user", "123", "preferences")`). The `owner_id-created_at-index` GSI is what `search()` actually queries — `ScanIndexForward=False` on `gsi_sort_key` gives true recency order from DynamoDB itself, not a re-sort of whatever page happened to be fetched (which silently returns the wrong "most recent N" once a namespace has more memories than fit in one page — see [CHANGELOG](MEMORY_LAYER.md)). Create it however you provision infra (CDK/Terraform/console) — here's the raw shape via the AWS CLI, if you just want to try it out:

```bash
aws dynamodb create-table \
  --table-name my-app-memories \
  --attribute-definitions \
      AttributeName=owner_id,AttributeType=S \
      AttributeName=sort_key,AttributeType=S \
      AttributeName=gsi_sort_key,AttributeType=S \
  --key-schema \
      AttributeName=owner_id,KeyType=HASH \
      AttributeName=sort_key,KeyType=RANGE \
  --global-secondary-indexes \
      '[{"IndexName":"owner_id-created_at-index","KeySchema":[{"AttributeName":"owner_id","KeyType":"HASH"},{"AttributeName":"gsi_sort_key","KeyType":"RANGE"}],"Projection":{"ProjectionType":"ALL"}}]' \
  --billing-mode PAY_PER_REQUEST

# Optional but recommended — lets episodic memories actually expire instead of
# accumulating forever. memory-layer sets the `ttl` attribute; DynamoDB does the rest.
aws dynamodb update-time-to-live \
  --table-name my-app-memories \
  --time-to-live-specification "Enabled=true, AttributeName=ttl"
```

Set `MEMORY_TABLE=my-app-memories` or pass `table_name` explicitly — `DynamoDBStore(table_name="my-app-memories")`.

## Quickstart

```python
from memory_layer import DynamoDBStore

store = DynamoDBStore(table_name="my-app-memories")

namespace = ("user", "user-123")

store.put(namespace, "mem-1", {"content": "Prefers responses in Spanish", "type": "semantic"})
store.put(namespace, "mem-2", {"content": "Reviewed Q3 numbers on 2026-08-01", "type": "episodic"})

memories = store.search(namespace, limit=10)
for m in memories:
    print(m.value["content"])
```

### Inside a LangGraph node

LangGraph injects `store` into any node whose signature asks for it:

```python
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from memory_layer import DynamoDBStore
from memory_layer.retrieval import SimpleRetrieval

store = DynamoDBStore(table_name="my-app-memories")
graph = builder.compile(checkpointer=checkpointer, store=store)

def supervisor_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict:
    user_id = state["context"]["user_id"]
    retrieval = SimpleRetrieval(store, limit=5)
    memories = retrieval.fetch(("user", user_id))
    memory_block = retrieval.to_prompt_block(memories)
    # ...inject memory_block into the system prompt
    return {}
```

### Writing memories back

```python
from memory_layer.writer import MemoryWriter

writer = MemoryWriter(llm=your_chat_model, store=store)

async def memory_writer_node(state: AgentState) -> dict:
    await writer.extract_and_save(
        namespace=("user", state["context"]["user_id"]),
        messages=state["messages"],
        session_id=state["context"]["session_id"],
    )
    return {}
```

The extraction LLM only needs `with_structured_output` support — every major provider's LangChain integration has it.

## Memory types

| Type | What it's for | Default TTL |
|---|---|---|
| `semantic` | Stable preferences and facts ("prefers Spanish", "works on the Acme account") | none |
| `episodic` | Specific past events ("reviewed Q3 numbers on 2026-08-01") | 90 days |
| `procedural` | Recurring work patterns ("always starts with a channel breakdown") | none |

TTL policy per type lives in `memory_layer.store._TTL_SECONDS_BY_TYPE` — override it if 90 days isn't the right default for your product.

## Scopes

A namespace is just a tuple — `memory-layer` doesn't prescribe what it means, but the common shapes are:

```python
("user", cognito_sub)        # private to one user
("instance", tenant_id)      # shared across every user of one tenant
```

## Roadmap

- **Semantic retrieval** — embed memories + query, rank by cosine similarity, for products that outgrow "most recent N" (roughly ~20+ memories per user is where this starts to matter). Lives behind the `semantic` extra so the core library stays dependency-light.
- **Deduplication on write** — skip persisting a fact that's a near-duplicate of one already stored.
- **Bring-your-own embeddings backend** — DynamoDB-native cosine similarity to start; pluggable enough to swap in pgvector/a real vector store later if volume ever justifies it.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run against [`moto`](https://github.com/getmoto/moto) — no real AWS account or network access required.

## License

MIT — see [LICENSE](LICENSE).
