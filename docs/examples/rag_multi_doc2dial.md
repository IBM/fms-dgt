# RAG: Multi-Doc2Dial

This example generates grounded multi-turn RAG conversations using the `public/rag/conversation` databuilder in static mode. Each conversation is anchored to a set of documents pre-selected at initialization; the assistant synthesizes responses from that fixed document context without issuing retrieval calls. The output is a dataset of faithful, document-grounded conversations ready for SFT fine-tuning.

The task configurations are at [`tasks/public/rag/static/multi_doc2dial/`](https://github.com/IBM/fms-dgt/tree/main/tasks/public/rag/static/multi_doc2dial/). Four domains are available: DMV, SSA, StudentAid, and VA.

## Prerequisites

Pull a generation model via Ollama:

```bash
ollama pull granite4:3b
```

> **Model size recommendation:** RAG conversation generation is a demanding task. The scenario stage must ground a coherent information need in multiple documents; the flow controller must reason about conversation state, pattern eligibility, and document coverage simultaneously; the assistant must synthesize faithful responses without hallucinating beyond the document set. Models smaller than roughly 30B parameters tend to produce shallow scenarios, pattern selection that ignores constraints, and assistant responses that drift from the documents. For production-quality output, use a model of 30B parameters or larger.

Set the data directory environment variable (the task configs use `${DGT_DATA_DIR}` to locate the corpus and seed files):

```bash
export DGT_DATA_DIR=./data
```

## Run it

Run a single domain:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/rag/static/multi_doc2dial/dmv/task.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

Run all four domains in one invocation:

```bash
python -m fms_dgt.public \
  --task-paths \
    ./tasks/public/rag/static/multi_doc2dial/dmv/task.yaml \
    ./tasks/public/rag/static/multi_doc2dial/ssa/task.yaml \
    ./tasks/public/rag/static/multi_doc2dial/studentaid/task.yaml \
    ./tasks/public/rag/static/multi_doc2dial/va/task.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

Output is written under `output/public/rag/static/multi_doc2dial/<domain>/`:

- `final_data.jsonl`: full internal representation with all pipeline steps, documents, and scores
- `formatted_output.jsonl`: clean `messages` format with source documents attached, ready for SFT fine-tuning

## The stage pipeline

The `public/rag/conversation` databuilder runs each conversation through two phases.

### Initialization

Initialization stages run once per conversation before any turns are generated.

**`lm/scenario/rag`** samples `k` documents from the corpus using the configured `DocumentSampler`, then generates a scenario that describes a realistic information need grounded in those documents. The sampler uses `group_by: domain` to ensure all sampled documents come from the same domain, producing coherent scenarios rather than mixing unrelated topics. The scenario and documents are carried through every subsequent stage.

**`sample/persona`** assigns a user persona sampled from `data/public/examples/chit_chat/personas.jsonl`. The persona shapes the simulated user's voice, goals, and expertise level throughout the conversation.

### Iteration

Iteration stages run in a loop, one turn at a time.

**`lm/flow_controller/rag`** reads the conversation history and the available documents, then selects one of eight patterns to guide the next user turn:

| Pattern | What it produces |
|---|---|
| `rag/factoid` | Brief, specific factual question answerable directly from the documents |
| `rag/explanation` | Request to explain a concept, policy, or process described in the documents |
| `rag/instructional` | Request for a sequence of steps to accomplish a task from the documents |
| `rag/comparative` | Request to compare two or more entities or options present in the documents |
| `rag/follow_up` | Deeper question on a specific detail from the assistant's last response |
| `rag/clarification` | Request to clarify something the user found unclear in the assistant's last response |
| `rag/ambiguous` | Vague, keyword-style question that prompts the assistant to ask for clarification |
| `rag/termination` | Closing turn when the scenario is resolved or documents are exhausted |

The flow controller enforces constraints defined in the pattern descriptions (for example, `rag/instructional` fires at most twice per conversation; `rag/follow_up` cannot be selected more than twice consecutively). When `rag/termination` is selected, the pipeline ends the conversation if `min_turns` has been reached.

**`lm/user/rag`** generates the user's next message using the scenario, persona, conversation history, and the hint produced by the flow controller.

**`lm/assistant/rag/static`** generates a grounded assistant response by reasoning over the fixed document set. It does not issue a retrieval call; it synthesizes directly from the documents injected at initialization.

## Turn bounds

```yaml
max_turns: 6
min_turns: 3
```

Conversations shorter than `min_turns` are discarded. Conversations that reach `max_turns` without a termination signal are ended and kept if they satisfy `min_turns`.

## Output format

The formatter writes each conversation as a `messages` array with source documents attached:

```json
{
  "conversation_id": "a3f2c1d8",
  "documents": [
    {"doc_id": "Registrations#3_0", "title": "Registrations#3", "text": "Your registration is the sticker placed on your windshield..."},
    {"doc_id": "Registrations#5_0", "title": "Registrations#5", "text": "You can renew your registration online..."}
  ],
  "messages": [
    {"role": "user", "content": "How do I renew my vehicle registration in New York?"},
    {"role": "assistant", "content": "You can renew your registration online and print a 10-day temporary document while the new one arrives by mail..."},
    {"role": "user", "content": "Are there any situations where I cannot renew online?"},
    {"role": "assistant", "content": "Yes. If your vehicle has been altered or stretched to increase the number of passengers, or if other special circumstances apply, you cannot renew online..."}
  ]
}
```

The `conversation_id` matches the corresponding record in `final_data.jsonl`. The `documents` field lists only the documents injected at initialization for this conversation.

## Data files

- **`data/public/rag/static/multi_doc2dial/<domain>/documents.jsonl`**: the document corpus for each domain. Each record has `id`, `title`, `domain`, and `text` fields.
- **`data/public/rag/static/multi_doc2dial/<domain>/seed_examples.jsonl`**: complete conversations used as in-context learning examples by the scenario and stage LMs.

See [`tasks/public/rag/static/multi_doc2dial/README.md`](https://github.com/IBM/fms-dgt/tree/main/tasks/public/rag/static/multi_doc2dial/README.md) for details on the corpus provenance and how to add a new domain.

## Next steps

- Swap the LM engine: see [Changing the Language Model Engine](../tutorials/changing_lm_engine.md).
- Understand the RAG subsystem components: see [RAG Data Generation](../concepts/rag.md).
- For live retrieval (assistant issues search queries during generation): use `tasks/public/rag/live/task.yaml` with a running Elasticsearch cluster.
