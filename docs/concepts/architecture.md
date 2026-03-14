# Architecture

Synthetic data generation pipelines tend to collapse into one-off scripts: a prompt template, a loop, an LLM call, some output parsing, and a file write. That works for a single experiment, but it does not scale. Models change, prompts evolve, quality bars shift, and the same underlying generation logic ends up duplicated across a dozen slightly different scripts.

DiGiT is built around a different premise. The **what** and the **how** of data generation are separated into two distinct concepts, Tasks and Databuilders, and everything else in the framework (Blocks, Datastores, Dataloaders) exists to make those two concepts composable, reusable, and efficient at scale.

## Core Concepts

### Tasks

A Task defines **what** data to produce. It is a declarative specification that captures:

- The schema of input and output data (via typed dataclasses)
- Where to load seed or input data from
- How many records to generate, or when to stop
- Where to write intermediate and final output

Tasks are configuration, not code. A developer configures a task in a YAML file and the framework instantiates it at runtime. The same task YAML can be run against different databuilders, or the same databuilder can serve multiple tasks simultaneously.

DiGiT provides two built-in task patterns:

|                | **GenerationTask**                                                                      | **TransformationTask**                              |
| -------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Loop style** | Iterative: runs until target record count is reached                                    | Single-pass: processes each input record once       |
| **Input**      | Seed examples (inline or from file), optionally mixed with previously generated outputs | A fixed dataset to transform                        |
| **Output**     | New synthetic records in the same schema as seeds                                       | Transformed records, potentially a different schema |

See [Tasks](tasks.md) for the full reference.

### Databuilders

A Databuilder defines **how** to produce data. It is a Python class that implements a `__call__` method: given a batch of input records, return a batch of output records. Everything else, looping, stopping, saving, deduplicating, is handled by the framework.

Databuilders are designed to be reusable. A single databuilder can serve multiple tasks in the same run, and its blocks (see below) are initialized once and shared across all of them.

See [Databuilders](databuilders.md) for the full reference.

### Blocks

Blocks are single-operation components used inside a databuilder's `__call__` method. A block takes a list of dictionary-like inputs, performs one operation (an LLM call, a validation check, a deduplication pass), and returns results.

The key design decision here is that all parallelism and concurrency lives inside blocks, not inside `__call__`. This makes databuilder code simple and sequential while the framework handles throughput transparently.

Built-in block types include:

- **LMProvider**: Connects to LLM backends (Ollama, WatsonX, OpenAI, Anthropic, vLLM). Handles async batching, credential pooling, and retry internally.
- **ValidatorBlock**: Filters records by a boolean condition. Rejected records are saved to a separate store for inspection.
- **Utility blocks**: ROUGE-L deduplication, field remapping, list flattening, and others.

See [Blocks](blocks.md) for the full reference.

### Datastores and Dataloaders

Datastores and Dataloaders handle data persistence and iteration respectively.

A **Datastore** is a storage abstraction. It knows how to read and write data in a specific format (JSONL, Parquet, CSV, JSON) to a specific location. DiGiT creates one datastore per output type per task: `data` (intermediate), `final_data` (post-processed), `formatted_data` (after formatting), and block stores for rejected records.

A **Dataloader** is a stateful iterator over a datastore. It supports resumable iteration (useful for long-running generation jobs), optional field remapping, and looping over seed data indefinitely.

See [Datastores](datastores.md) and [Dataloaders](dataloaders.md) for the full reference.

## Execution Model

A DiGiT run proceeds as follows:

1. **Parse**: All task YAML files are loaded and grouped by their `data_builder` field.
2. **Initialize**: One databuilder instance is created per unique builder name, with all associated tasks passed to it. Blocks are initialized once per databuilder.
3. **Generate**: Each databuilder runs its task loop. For generation tasks, the loop continues until all tasks reach their target record count. For transformation tasks, it makes one pass over the input data.
4. **Postprocess**: Postprocessors (which are blocks declared in the builder YAML) run over the accumulated data for each task.
5. **Save**: Final data is written to each task's output datastore.

## Key Architectural Decisions

### Task grouping by databuilder

When multiple tasks share the same databuilder, they run together in a single loop. The databuilder receives a mixed batch of examples from all active tasks, generates data for all of them in one call, and the framework routes results back to the correct task by the `task_name` field on each record.

This matters for throughput: if you have ten tasks using the same LLM, one databuilder instance makes ten tasks worth of LLM calls in a single batched request, instead of ten sequential loops each making independent calls.

### Async LM concurrency with credential pooling

LMProvider blocks issue all requests in a batch concurrently using a persistent async event loop. Concurrent requests are bounded by two limits: a per-block `call_limit` and a global semaphore shared across all blocks that use the same API credential.

The global semaphore is the important part. If you run five databuilders that all call the same OpenAI API key, the total number of in-flight requests across all of them is capped at the semaphore limit. This prevents rate-limit errors even as you scale the number of tasks and builders in a single run.

### Lazy plugin registry

Databuilders, blocks, datastores, dataloaders, and formatters are all registered with decorators (`@register_data_builder`, `@register_block`, and so on). The registry does not import these modules at startup. Instead, it scans source files statically to build a map of decorator calls to file paths, and imports a module only when its registered name is first requested.

This means startup time does not grow with the number of plugins installed, and adding a new databuilder or block requires no changes to any registry configuration file. The decorator on the class is the entire registration.

Plugins are organized into namespaces (`core`, `public`). The `core` namespace is always loaded. Additional namespaces are added by the entry point (`fms_dgt.public`) or via `--include-namespaces` at the command line.
