# Memory Layer — Design Document

## Visión

Biblioteca Python que provee **memoria cross-thread** para agentes LangGraph. Implementa el protocolo `BaseStore` de LangGraph para que cualquier grafo pueda leer y escribir memorias de usuario sin acoplar lógica de storage al agente.

No es una feature de Atlas — es una dependencia que Atlas (y cualquier otro agente LangGraph) puede importar.

---

## Problema

LangGraph `checkpointer` persiste el estado **dentro** de un thread (conversación). Cuando el usuario abre una nueva sesión, el grafo empieza de cero — no sabe que el usuario prefiere análisis por canal, que trabaja con la cuenta de Nike, o que la última vez quedó una pregunta sin resolver.

El `store` de LangGraph resuelve esto: memoria **cross-thread**, compartida entre todas las sesiones de un mismo usuario.

---

## Tipos de Memoria

| Tipo | Descripción | Ejemplo |
|---|---|---|
| `semantic` | Preferencias y hechos estables del usuario | "Prefiere respuestas en español", "trabaja con cuenta Nike" |
| `episodic` | Eventos pasados relevantes | "El 2025-05-10 analizamos fatiga de creativos Q4 de Nike" |
| `procedural` | Patrones de trabajo recurrentes | "Siempre empieza con breakdown por canal, luego creativos" |

---

## Scopes

```
user:{cognito_sub}          # preferencias personales — privado al usuario
instance:{instance_id}      # contexto compartido del tenant (todos los usuarios lo ven)
```

El scope se mapea directamente al `namespace` de LangGraph Store: `("user", user_id)`.

---

## Integración con LangGraph

LangGraph compila el grafo con `store` al lado del `checkpointer`:

```python
from memory_layer import DynamoDBStore

store = DynamoDBStore(table_name="memory-layer-dev-memories")
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Dentro de un nodo, el store se accede vía `RunnableConfig`:

```python
from langgraph.store.base import BaseStore

def supervisor_node(state: AtlasState, config: RunnableConfig, store: BaseStore) -> dict:
    user_id = state["context"].get("user_id", "")
    memories = store.search(("user", user_id), query=state["messages"][-1].content, limit=5)
    # inject relevant memories into system prompt
```

Al final de la conversación, un nodo `memory_writer` opcional extrae y persiste hechos nuevos:

```python
def memory_writer_node(state: AtlasState, store: BaseStore) -> dict:
    # run LLM to extract key facts from the conversation
    # store.put(("user", user_id), memory_id, {"content": ..., "type": "semantic"})
    return {}
```

---

## Estrategias de Retrieval

### 1. Simple (MVP)
Fetch N memorias más recientes del usuario, inyectar todas en el system prompt.
- Bueno para: pocos usuarios, pocas memorias por usuario
- Límite: el context window crece con las memorias

### 2. Semántica (Phase 2)
Embed la query del usuario + las memorias. Cosine similarity para seleccionar las más relevantes.
- Requiere: embeddings (OpenAI `text-embedding-3-small` o similar)
- Storage: vector en DynamoDB (como string serializado) o migrar a pgvector/Pinecone

### 3. Por tipo (Phase 2)
Filtrar por `type` antes de rankear — siempre incluir `semantic` (preferencias), limitar `episodic`.

**Arrancar con Simple. Migrar a Semántica cuando el número de memorias por usuario supere ~20.**

---

## Estrategias de Escritura

### Automática (recomendada)
Nodo `memory_writer` al final del grafo. LLM extrae hechos nuevos de la conversación y los persiste. Se ejecuta solo si hubo interacción con un agente specialist (no para saludos/clarificaciones).

### Explícita
El agente escribe memorias cuando el usuario dice "recordá que..." o cuando detecta un hecho relevante durante la conversación. Requiere una tool `save_memory` disponible para los agentes.

### Ambas
La combinación más robusta. Explícita para capturar intenciones directas del usuario; automática como red de seguridad.

---

## Data Model — DynamoDB

> **Nota (v0.2.0):** este modelo cambió respecto a la versión original de este documento — el diseño de abajo (`owner_id` = `":".join(namespace)`, con una GSI `owner_id-created_at-index`) no soportaba prefix search de verdad (`search(("user","123"))` no encontraba memorias guardadas en `("user","123","preferences")`) y tenía colisión de separador si algún segmento del namespace contenía `":"`.
>
> **Nota (v0.3.0):** v0.2.0 sacó la GSI para resolver lo de arriba, y con eso rompió la recencia: `search()` ordenaba por `created_at` *después* de paginar, así que una vez que un namespace tenía más memorias de las que entran en una página, el "top N más reciente" pasaba a ser un slice arbitrario (orden lexicográfico de `sort_key`, no de tiempo) — verificado empíricamente con 40 memorias de ~40KB, `limit=5` devolvía los índices 23-19 en vez de 39-35. v0.3.0 reintroduce la GSI, pero con otro rol: `gsi_sort_key` (no `created_at` solo, para evitar empates en el mismo milisegundo).
>
> **Nota (v0.4.0):** v0.2.0/v0.3.0 particionaban por `namespace[0]` solo, para soportar `search()` con un prefijo más corto que el namespace de escritura. Eso colapsaba **todos** los namespaces que comparten el primer segmento (p. ej. todos los `"user"` de todo el deployment) en una sola partición de DynamoDB. Verificado empíricamente: la memoria de un principal se volvía irrecuperable en cuanto ~150 principales no relacionados, bajo el mismo primer segmento, escribían más recientemente — el `Limit` de la query acota los items *leídos* de esa partición compartida antes de aplicar cualquier filtro de prefijo. Ni el uso documentado de este mismo paquete (`("user", user_id)`, donde `search()` siempre corre con el mismo namespace exacto que el `put()`) ni ningún caller real necesitan prefix search genuino entre profundidades — así que v0.4.0 vuelve a particionar por el **namespace completo** (como v0.1.0, máxima cardinalidad) y **elimina el soporte de prefix search jerárquico**: `search()` requiere el namespace exacto con el que se escribió; un namespace más corto consulta una partición distinta y devuelve una lista vacía, no un error (no hay forma de distinguir, solo a partir del input, "namespace exacto de un solo segmento" de "prefijo jerárquico de un namespace más profundo"). El schema real, ver `src/memory_layer/store.py`:

| Campo | Tipo | Descripción |
|---|---|---|
| `owner_id` | PK (String, tabla base y GSI) | El namespace **completo**, unido con `\x1f` (no `:` — evita colisión con IDs reales que contengan `:`, como ARNs) — una partición de DynamoDB por namespace distinto |
| `sort_key` | SK (String, tabla base) | `memory_id`, verbatim |
| `gsi_sort_key` | SK (String, GSI `owner_id-created_at-index`) | `created_at` + `\x1f` + `memory_id` — recencia real vía `ScanIndexForward=False`, con `memory_id` como desempate determinístico |
| `namespace` | String | El tuple de namespace original, serializado como JSON — para reconstruir `Item.namespace` sin volver a parsear el owner_id |
| `memory_id` | String | La key original, verbatim |
| `value` | String | JSON del contenido (`content`, `type`, etc.) |
| `created_at` / `updated_at` | String | ISO 8601 UTC |
| `ttl` | Number | TTL Unix timestamp — presente solo si el tipo tiene TTL por defecto o se pasó `ttl=` explícito en `put()` |

**Con una GSI** (`owner_id-created_at-index`, proyección `ALL`). `search()` consulta la GSI directamente con `ScanIndexForward=False` para recencia real del lado del server — ya no hace falta ningún `FilterExpression` de prefijo, porque `owner_id` ya es el namespace exacto.

---

## Arquitectura del Proyecto

```
memory-layer/
├── src/
│   └── memory_layer/
│       ├── __init__.py
│       ├── store.py        # DynamoDBStore — implementa LangGraph BaseStore
│       ├── types.py        # MemoryRecord, MemoryType, MemoryScope TypedDicts
│       ├── retrieval.py    # Estrategias: SimpleRetrieval, SemanticRetrieval
│       └── writer.py       # MemoryWriter — extracción LLM de hechos
├── tests/
│   ├── conftest.py
│   ├── test_store.py
│   └── test_writer.py
├── pyproject.toml
└── MEMORY_LAYER.md
```

---

## Plan de Implementación

### Phase 1 — Store + Simple Retrieval
- [ ] `DynamoDBStore` implementando `langgraph.store.base.BaseStore`
- [ ] `SimpleRetrieval` — fetch N más recientes, sin embeddings
- [ ] `MemoryRecord` TypedDict + validación
- [ ] Tests con DynamoDB Local

### Phase 2 — Semantic Retrieval
- [ ] Embeddings via `langchain-openai` (`text-embedding-3-small`)
- [ ] `SemanticRetrieval` — cosine similarity sobre embeddings en DynamoDB
- [ ] Migración de registros existentes (backfill embeddings)

### Phase 3 — Memory Writer
- [ ] `MemoryWriter` — nodo LangGraph que extrae hechos al final de cada conversación
- [ ] Deduplicación — no escribir memorias redundantes (LLM judge o embedding similarity)
- [ ] Expiración diferenciada por tipo — `episodic` TTL 90 días, `semantic` sin TTL

### Integración en Atlas
- [ ] Importar `memory_layer` como dependencia en `requirements.txt`
- [ ] Pasar `DynamoDBStore` al compilar el grafo en `core/graph.py`
- [ ] Nodo `memory_writer` opcional al final del grafo (skill `user_memory`)
- [ ] Inyección de memorias en `supervisor_node` vía `store.search()`

---

## Decisiones Pendientes

| Decisión | Opciones | Estado |
|---|---|---|
| ¿Embeddings propios o pgvector? | DynamoDB + cosine en Python vs pgvector en RDS | Pendiente — depende del volumen de usuarios |
| ¿Memory writer siempre activo o como skill? | Siempre vs `required_skill="user_memory"` | Pendiente |
| ¿TTL para memorias semánticas? | Sin TTL vs 1 año | Pendiente |
| ¿Límite de memorias por usuario? | Sin límite vs max 100 + archivado | Pendiente |