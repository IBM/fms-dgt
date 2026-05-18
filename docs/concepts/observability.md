# Observability

DiGiT writes structured telemetry alongside generated data so you can understand what happened during a run without adding any instrumentation to your own code. Every run produces two files in the `telemetry/` directory: `events.jsonl` for lifecycle events and `traces.jsonl` for performance spans.

## Output files

| File                     | Contains                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `telemetry/events.jsonl` | Structured lifecycle events: run start/finish/error, task start/finish, epoch boundaries, postprocessing, rejected data points  |
| `telemetry/traces.jsonl` | Performance spans: one record per LLM call or pipeline phase with latency, token usage, model info, and optional payload        |

Both files are append-only JSONL. Each record is one JSON object per line. Files rotate at 100 MB and rotated files older than 14 days are deleted automatically.

## Lifecycle events

Every record in `events.jsonl` carries `build_id`, `run_id`, `event`, `timestamp`, and a human-readable `message`. The `build_id` identifies the experiment; the `run_id` identifies one attempt (preserved across resume restarts so a resumed run shares the same `run_id` as the original).

| Event                    | When it fires                                                                                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_started`            | When `execute_tasks()` begins. Carries `builder_name`, `task_names`, `resumed` flag                                                                                         |
| `run_finished`           | When all tasks complete successfully                                                                                                                                         |
| `run_errored`            | When an unhandled exception stops the run. Carries the exception message                                                                                                     |
| `task_started`           | When a task enters the active set                                                                                                                                            |
| `task_finished`          | When a task completes. Carries `reason`: `complete`, `stalled_generation`, or `stalled_postprocessing`                                                                       |
| `epoch_started`          | At the start of each generation epoch. Carries `epoch`, `active_task_names`, `active_task_count`                                                                            |
| `epoch_finished`         | At the end of each epoch, after postprocessing. Carries `epoch`, `generation_attempts` (how many generator batches ran this epoch), `task_counts`, and `finish_reasons`      |
| `postprocessing_finished`| After postprocessing completes for an epoch. Carries `epoch` and `task_counts` with `before`/`after` counts per task, showing how many records were accepted vs. discarded   |
| `data_point_rejected`    | When a `ValidatorBlock` filters a record. Carries `block_name`, `task_name`, `reason`                                                                                        |

Example record:

```json
{
  "event": "task_finished",
  "message": "Task 'public/examples/misconceptions' finished.",
  "build_id": "abc123",
  "run_id": "def456",
  "task_name": "public/examples/misconceptions",
  "reason": "complete",
  "timestamp": "2026-01-15T10:23:14.001Z"
}
```

## Spans

DiGiT writes spans for both pipeline phases and individual LLM calls. All spans carry `build_id` and `run_id` to link them back to the originating run.

### Pipeline spans

| `span_name`          | What it covers                                                            | Key fields                                        |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------- |
| `dgt.run`            | The entire run from `execute_tasks()` start to finish                     | `builder_name`, `task_names`                      |
| `dgt.epoch`          | One generation epoch, including the generator batch and the block chain   | `epoch`, `task_count`                             |
| `dgt.block`          | One invocation of a single block within an epoch                          | `block_name`, `block_type`                        |
| `dgt.postprocessing` | The postprocessing phase that runs at the end of each epoch               | `epoch`, `task_count`                             |

### LLM call spans

Every record in `traces.jsonl` with `span_name: dgt.llm_call` corresponds to one batched LLM call. Spans are written by the `LMProvider` block after each call completes.

| Field                 | Description                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `span_name`           | Always `dgt.llm_call`                                                                                    |
| `provider`            | Block type: `ollama`, `openai`, `anthropic`, etc.                                                        |
| `model_id`            | The `model_id_or_path` value used                                                                        |
| `method`              | `completion` or `chat_completion`                                                                        |
| `batch_size`          | Number of requests in this call                                                                          |
| `duration_ms`         | Pure API latency (excludes semaphore wait time)                                                          |
| `semaphore_wait_ms`   | Time spent waiting for a concurrency slot                                                                |
| `prompt_tokens`       | Token count from the provider response                                                                   |
| `completion_tokens`   | Token count from the provider response                                                                   |
| `task_names`          | Deduplicated list of task names whose data was in this batch. Present when input dicts include `task_name` |
| `build_id` / `run_id` | Links the span to its run                                                                                |

Example record:

```json
{
  "span_name": "dgt.llm_call",
  "provider": "ollama",
  "model_id": "granite4:3b",
  "method": "chat_completion",
  "batch_size": 10,
  "duration_ms": 4821,
  "semaphore_wait_ms": 12,
  "prompt_tokens": 1840,
  "completion_tokens": 312,
  "task_names": ["public/examples/geography_qa"],
  "build_id": "abc123",
  "run_id": "def456"
}
```

!!! note "Token attribution with multi-task completion batches"
    When using `completion` mode with a multi-task run, a single API call may contain requests from several tasks. In that case `task_names` lists all of them, but `prompt_tokens` and `completion_tokens` are for the entire batch, not per task. For per-task token attribution, prefer single-task runs or `chat_completion` mode (which always sends one request per task item).

## Payload recording

By default, prompts and completions are not written to telemetry files. To include them for debugging, set `DGT_TELEMETRY_RECORD_PAYLOADS=1`. Each span will then carry `prompt` or `messages` and `completion` fields, truncated to `DGT_TELEMETRY_PAYLOAD_MAX_CHARS` characters (default 4096). A `payload_truncated: true` flag is set when truncation occurs.

???+ warning "Payload recording and sensitive data"
    Prompts and completions may contain seed examples, generated outputs, or document content. Do not enable payload recording in environments where that data is sensitive, and ensure the `telemetry/` directory is excluded from version control (it is in `.gitignore` by default).

## Environment variables

| Variable                          | Default      | Description                                                      |
| --------------------------------- | ------------ | ---------------------------------------------------------------- |
| `DGT_TELEMETRY_DIR`               | `telemetry/` | Directory for telemetry output files                             |
| `DGT_TELEMETRY_DISABLE`           | _(unset)_    | Set to any non-empty value to disable all telemetry file writing |
| `DGT_TELEMETRY_RECORD_PAYLOADS`   | _(unset)_    | Set to `1` to include prompts and completions in spans           |
| `DGT_TELEMETRY_PAYLOAD_MAX_CHARS` | `4096`       | Maximum characters per payload field                             |

## Disabling telemetry

```bash
DGT_TELEMETRY_DISABLE=1 python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --restart
```

When disabled, no files are written and no overhead is incurred. All other run behavior is unchanged.
