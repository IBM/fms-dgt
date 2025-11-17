**DGT** is a modular and extensible framework designed to support the generation of synthetic data across a wide range of use cases. It enables users to specify both **what** data they need and **how** it should be generated through two primary components: **Tasks** and **Databuilders**.

### Tasks

Tasks define the **intent** and **requirements** for synthetic data generation. They serve as high-level specifications that guide the overall data generation process.

Each task includes:

- The type of data to be generated
- Required assets (e.g., schemas, seed data)
- Stopping criteria (e.g., number of records, time limits, quality thresholds)

Tasks are declarative and reusable. While there is generally a many-to-one relationship between tasks and databuilders, a task can theoretically be fulfilled by multiple databuilders, provided the generated data adheres to the constraints defined by the databuilder.

### Databuilders

Databuilders are the **operational units** responsible for implementing the logic to generate synthetic data. They transform task specifications into actual data outputs. They are typically opinonated and stateless in their design. Databuilders are engineered with broader applicability and maintainability in mind.

## Execution Model

Parallelism is a core feature of DGT. It supports the execution of multiple tasks—associated with the same or different databuilders—within a single run.

A typical run proceeds as follows:

- **Initialization**: All requested tasks are initialized, followed by their associated databuilders.

<!-- prettier-ignore-start -->
+ **Iteration Loop**:
      - Each databuilder receives a set of incomplete tasks.
      - It generates synthetic data points for those tasks.

<!-- prettier-ignore-end -->

- **Stopping Criteria Check**: After each iteration, the framework checks whether stopping criteria (e.g., record count, stall limit, failure threshold) have been met for each task.
- **Completion**: The run ends when all tasks are either completed or exited.

## Additional Components

### Blocks

Blocks are single-operation components that are initialized once per databuilder and shared across associated tasks. They promote logical isolation, reusability and minimal runtime memory footprint. Few prominent examples of blocks include:

- **LMProvider Block**: Connects to over six Language Model (LM) engines.
- **Utility Blocks**: Provide common operations such as deduplication, filtering, syntax checking, and LLM-as-a-Judge (LLMaJ) capabilities.

### Datastores and Dataloaders

Most data generation pipelines require loading external assets such as In-context learning (ICL) examples, knowledge documents, API specifications, prompt templates. DGT simplifies such asset management through **Datastores** and **Dataloaders**.
