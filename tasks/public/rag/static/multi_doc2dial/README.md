# Multi-Doc2Dial: Static RAG Tasks

This directory contains task configurations and data for generating grounded multi-turn RAG conversations in static mode. Data is derived from the [Doc2Dial dataset](https://doc2dial.github.io/), which pairs government service documents with dialogue annotations across four US government service domains.

## Domains

| Domain     | Directory     | Coverage                                                                            |
| ---------- | ------------- | ----------------------------------------------------------------------------------- |
| DMV        | `dmv/`        | Vehicle registration, titles, licenses, and related NY DMV services                 |
| SSA        | `ssa/`        | Social Security benefits, eligibility, applications, and account management         |
| StudentAid | `studentaid/` | Federal student aid, FAFSA, loan repayment, and grant programs                      |
| VA         | `va/`         | Veterans benefits, healthcare enrollment, disability claims, and education benefits |

Each domain is a separate task rather than a single multi-domain task for two reasons. First, seed examples are domain-specific: the ICL pool for DMV conversations should not include SSA conversations, because mixing domains degrades scenario coherence and confuses the stage LMs. Second, output accounting is per-task: `--num-outputs-to-generate` applies independently to each domain, so you can generate different quantities per domain or resume a failed domain without re-running the others.

## File layout

```
<domain>/
├── task.yaml           # Task configuration: pipeline stages, sampler, engine config
├── documents.jsonl     # Document corpus
└── seed_examples.jsonl # ICL seed conversations
```

## Document format

Each line in `documents.jsonl` is a JSON object:

```json
{
  "id": "Registrations#3_0",
  "title": "Registrations#3",
  "domain": "dmv",
  "text": "Your registration is the sticker placed on your windshield..."
}
```

| Field    | Required | Description                                                                                 |
| -------- | -------- | ------------------------------------------------------------------------------------------- |
| `id`     | Yes      | Unique identifier within the corpus                                                         |
| `text`   | Yes      | Document body; the text the assistant synthesizes from                                      |
| `title`  | No       | Human-readable title; included in formatted output                                          |
| `domain` | No       | Domain label; used by `group_by: domain` in the sampler to keep scenario documents coherent |

## Seed example format

Each line in `seed_examples.jsonl` is a complete `ConversationDataPoint` serialized as JSON. A seed includes:

- A `ScenarioStep` describing the information need and the grounding documents
- A `PersonaStep` describing the simulated user
- One or more `FlowControllerStep`, `UserStep`, and `AssistantStep` records interleaved as turns

Seeds are used as in-context learning examples by the scenario generator and stage LMs. More seeds, and seeds that cover diverse patterns and domains, improve generation quality.

## Adding a new domain

1. Create a new directory under `multi_doc2dial/`:

   ```
   multi_doc2dial/
   └── my_domain/
       ├── documents.jsonl
       ├── seed_examples.jsonl
       └── task.yaml
   ```

2. Prepare `documents.jsonl` with at minimum `id` and `text` fields. Add `domain` if you want the sampler to group documents by topic.

3. Bootstrap `seed_examples.jsonl` using a zero-seed run. Writing good seeds from scratch is hard because the internal step schema is verbose and easy to get wrong. Instead, point `seed_datastore.data_path` at an empty file, run a small batch (5 to 10 conversations), and inspect `final_data.jsonl` in the output directory. Pick one or two conversations that are well-structured and cover different patterns, copy them directly into `seed_examples.jsonl`, then re-run. With even a small pool of real model output as seeds, quality improves substantially because the stage LMs now have domain-appropriate ICL examples to follow. Repeat the cycle — run, pick the best outputs, add them to the seed pool — until the pool covers the patterns you care about (aim for at least 3 to 5 conversations, each ending via a different pattern).

4. Copy `task.yaml` from an existing domain and update:

   - `task_name`
   - `task_description`
   - `seed_datastore.data_path`
   - `tools.engines.file_retriever.path`

5. Run a small batch to validate:

   ```bash
   python -m fms_dgt.public \
     --task-paths ./tasks/public/rag/static/multi_doc2dial/my_domain/task.yaml \
     --num-outputs-to-generate 10 \
     --restart
   ```

## Source data

Documents and dialogue annotations are derived from the [Doc2Dial v1.0.1 dataset](https://github.com/doc2dial/sharedtask-dialdoc2021) (Fan et al., 2020). The original dataset is distributed under CC BY-NC 4.0. Documents have been lightly reformatted for compatibility with the DiGiT corpus schema; no content has been altered.
