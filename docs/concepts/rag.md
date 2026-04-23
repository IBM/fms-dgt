# RAG Data Generation

Retrieval-augmented generation (RAG) training data requires conversations where the assistant synthesizes responses from retrieved documents rather than relying on parametric knowledge alone. DiGiT generates these conversations by treating the retriever as a tool: the same tool infrastructure that drives tool-calling data generation handles document retrieval, and the same conversation pipeline manages the multi-turn loop.

## Retriever as tool

Rather than building a parallel retrieval-specific subsystem, DiGiT models retrieval as a special case of tool use. A `SearchToolEngine` is a `ToolEngine` that returns documents instead of arbitrary API responses. This means:

- The task YAML configures retrieval the same way it configures any other tool engine.
- Document samplers reuse the sampler abstraction already present for tool selection.
- The conversation pipeline, flow controller, and persona machinery are shared with non-RAG databuilders.

The RAG-specific behavior lives entirely in the stages: scenario initialization samples documents and grounds the scenario, and the assistant stage synthesizes from a fixed or live document context rather than calling a general-purpose tool.

## Two modes

### Static

Documents are sampled once at scenario initialization and injected into every subsequent stage as fixed context. The assistant never issues a retrieval call during the conversation; it synthesizes directly from the document set it was given.

Use static mode when:

- You want to train faithfulness and grounded synthesis without retrieval mechanics.
- Your corpus is small enough to pre-load into memory or a local file.
- You want reproducible conversations tied to a known document set.

### Live

The assistant issues a retrieval tool call each turn. The engine executes the query against a live backend (Elasticsearch, or any registered `SearchToolEngine`) and returns results as `ToolCallStep`/`ToolResultStep` pairs in the conversation. The output contains the full retrieval trace.

Use live mode when:

- You want to train the model to formulate retrieval queries.
- You want the training data to reflect the actual retrieval behavior of a production system.
- Your corpus is too large to pre-load.

## Components

```
DocumentSampler ──► SearchToolEngine
                          │
                    ┌─────┴──────┐
                    ▼            ▼
              Static mode    Live mode
           (inject at init)  (call per turn)
```

### SearchToolEngine

A `SearchToolEngine` wraps a document corpus and exposes a search interface. Three backends are available:

| Type | When to use |
|---|---|
| `search/in_memory` | Small corpora loaded at startup; fastest, no external dependencies |
| `search/file` | JSONL corpus on disk; loaded lazily, suitable for medium-sized corpora |
| `search/elasticsearch` | Large corpora or production-mirroring; requires a running ES cluster |

All three implement the same interface. The `projection` field maps corpus field names to the internal `Document` schema (`body`, `doc_id`, `title`, `domain`).

### DocumentSampler

A `DocumentSampler` selects a subset of documents from the corpus to ground a scenario. It runs during initialization, before any turns are generated.

| Type | Behavior |
|---|---|
| `search/random` | Uniform random sample, optionally grouped by a corpus field (e.g., `domain`) |

The `group_by` field stratifies sampling so each scenario is grounded in documents from a single domain or category, which produces more coherent conversations.

## YAML configuration

### Static mode (file corpus)

```yaml
tools:
  engines:
    file_retriever:
      type: search/file
      path: ${DGT_DATA_DIR}/public/rag/static/my_corpus/documents.jsonl
      format: jsonl
      projection:
        body: text
        doc_id: id
      limit: 3

initialization_stages:
  - name: lm/scenario/rag
    generator: generator
    document_samplers:
      - type: search/random
        engine: file_retriever
        group_by: domain
        strategy: uniform
        weight: 1.0
    k: 3

iteration_stages:
  - name: lm/flow_controller/rag
    generator: generator
    patterns: [...]

  - name: lm/user/rag
    generator: generator

  - name: lm/assistant/rag/static
    generator: generator
```

### Static mode (in-memory corpus)

```yaml
tools:
  engines:
    memory_retriever:
      type: search/in_memory
      projection:
        body: text
        doc_id: id
      limit: 5

initialization_stages:
  - name: lm/scenario/rag
    generator: generator
    document_samplers:
      - type: search/random
        engine: memory_retriever
        strategy: uniform
        weight: 1.0
    k: 5
```

Documents are loaded into the engine at task startup via the `corpus` field or programmatically. Use this backend for small, static corpora where startup latency is acceptable.

### Live mode (Elasticsearch)

```yaml
tools:
  engines:
    es_retriever:
      type: search/elasticsearch
      hosts: ["https://localhost:9200"]
      default_index: my_index
      projection:
        body: content
        doc_id: _id
        title: title
      limit: 5

initialization_stages:
  - name: lm/scenario/rag
    generator: generator
    document_samplers:
      - type: search/random
        engine: es_retriever
        strategy: uniform
        weight: 1.0
    k: 5

iteration_stages:
  - name: lm/flow_controller/rag
    generator: generator
    patterns: [...]

  - name: lm/user/rag
    generator: generator

  - name: lm/assistant/rag/live
    generator: generator
```

In live mode the assistant stage issues a search tool call each turn. The output includes `ToolCallStep` and `ToolResultStep` entries alongside the user and assistant turns.

## Corpus format

Documents must be JSONL with at minimum a body field and a unique identifier. Additional fields (`title`, `domain`) are optional but improve sampler behavior when using `group_by`.

```json
{"id": "doc_001", "title": "Vehicle Registration", "domain": "dmv", "text": "Your registration is the sticker placed on your windshield..."}
```

The `projection` block in the engine config maps your field names to the internal schema:

```yaml
projection:
  body: text       # required — the document text
  doc_id: id       # required — unique identifier
  title: title     # optional
  domain: domain   # optional; used by group_by in samplers
```

## Seed example format

Seed examples are complete conversations used as in-context learning examples by the scenario and stage LMs. Each record follows the `ConversationDataPoint` schema: a list of steps with typed roles (`scenario`, `persona`, `flow_controller`, `user`, `assistant`).

See `data/public/rag/static/multi_doc2dial/dmv/seed_examples.jsonl` for a reference.

## Reading path

| I want to... | Go to |
|---|---|
| Run a working RAG example end to end | [RAG: Multi-Doc2Dial](../examples/rag_multi_doc2dial.md) |
| Understand the conversation pipeline | [Conversation Databuilder](databuilders.md) |
| Use the tool subsystem directly | [Tools](tools/index.md) |
