# Knowledge Generation

Data builder used for generating instruction-response pairs driven by examples in the knowledge and foundational skills branches of InstructLab Taxonomy.

It generates data using the specified model as the teacher (generator and validator) and prompt templates.

> [!WARNING]  
> **Issue**
>
> Segmentation faults and similar errors on macOS
>
> **Solution**
>
> Set the following environment parameters
>
> 1. Disable OpenMP threading via `export OMP_NUM_THREADS=1`
> 2. If error still persists, disable PyTorch MPS device via `export PYTORCH_MPS_DISABLE=1`
> 3. If error still persists, disable llama.cpp metal via `export LLAMA_NO_METAL=1`
> 4. Final attempt can be made as a dangerous workaround via `export KMP_DUPLICATE_LIB_OK=TRUE`
>
> Reference: https://github.com/neuml/txtai/issues/813#issuecomment-2485349327

## Task specification

This data builder supports [tasks](./task.py) defining the following parameters:

### Required Parameters

- `task_name`: (str) Name of the task
- `created_by`: (str) Creator of the task
- `task_description`: (str) Description of the task
- `data_builder`: (str) Must be `instructlab/knowledge`

### Data specification through reserved keywords

Tasks executed by this data builder require seed examples and documents that use the following parameters

#### Seed examples

Seed examples can be provided through the `seed_examples` field with the following parameters:

- `question`: (str) question for model to answer
- `answer`: (str) answer that model should produce

#### Knowledge documents

And knowledge documents can be passed using in two ways:

1. Using the `include` directive via the `documents` key.

- `documents`: (Dict) key-value pairs where keys are document names or groups and values are file paths or glob patterns (supported files types are `.md`, `.jsonl`, `.txt`)

For example:

```yaml
include:
  documents:
    photosynthesis: ${DGT_DATA_DIR}/public/instructlab/knowledge/textbook/science/biology/photosynthesis/photosynthesis.md
    structure_of_matter: ${DGT_DATA_DIR}/public/instructlab/knowledge/textbook/science/physics/static_electricity/structure_of_matter.txt

# Task fields
chunk_size: 800 # chunks the documents to at most 800 tokens (whitespace)
loop_over: True
```

> [!NOTE]

> When specifying documents using `include`, the documents will be iterated over only once. An example can be found [here](../../../../../tasks/public/instructlab/knowledge/textbook/history/ibm_history/task.yaml).

2. **[Recommended]** Using the `knowledge` key via a Datastore.

- `datastore_config`: (dict) Datastore configuration
- `fields`: (dict) A mapping of keys to extract and rename. By default retains all keys (`"*": "*"` is wildcard to retain all keys). Only works with `jsonl` and `parquet` data formats.

For example:

```yaml
# from local filesystem
knowledge:
  datastore_config:
    type: default
    store_name: documents
    # [RECOMMENDATION]: Always try to specify file extensions when using glob pattern
    data_path: ${DGT_DATA_DIR}/public/instructlab/knowledge/textbook/science/**
    # [OPTIONAL]: Due to glob limitation, in case of multiple file formats
    data_formats:
      - md
      - txt
      - jsonl
      - parquet

# Task fields
chunk_size: -1 # full document
loop_over: True # loops over documents after exhausting
```

> [!NOTE]

> - When specifying documents using `knowledge`, the documents can be iterated over multiple times. An example can be found [here](../../../../../tasks/public/instructlab/knowledge/textbook/science/task.yaml).

> - If using data formats other than `md` or `txt`, make sure the data records have the main text under the `content` key/column

Task YAML examples can be found [here](../../../../../tasks/public/instructlab/knowledge/textbook/).

### Additional Task Parameters

- `domain`: (str) Domain of the knowledge documents
- `chunk_size`: (int) Documents will be chunked to a maximum of this size (defaults to full document)
- `loop_over`: (bool) Flag controlling whether to loop over documents once exhausted.
- `num_icl_examples_per_prompt`: (int) Number of in-context learning (icl) examples to use per prompt. Defaults to `3`.
- `num_docs_per_iteration`: (int) Number of documents to use per iteration. Defaults to `100`.
- `question_style`: (str) Style of questions. Allowed values are `FRQ` (Free-Response Questions) and `MCQ` (Multiple-Choice Questions). Defaults to `FRQ`.
- `criteria`: (List[str]) Question-Answer validity criteria. Allowed values are `faithfulness`, `relevancy` and `question_verification`. By default, it uses all of them.

## Databuilder specification

#### Generators and Validators

- `generator`: `mistralai/mistral-medium-2505` via `watsonx`
- `validator`: `mistralai/mistral-medium-2505` via `lm_judge` and `watsonx`
- `tagger`: `mistralai/mistral-medium-2505` via `magpie_tag` and `watsonx` (see [Magpie Tagger](../../../blocks/magpie/tag/README.md) block)

#### Postprocessors:

- `dedup`: via `magpie_distance` (see [Magpie Distance](../../../blocks/magpie/distance/))
- `filter` via `magpie_filter` (see [Magpie Filter](../../../blocks/magpie/filter/README.md))

Default configuration for generator and validator used by the data builder is available [here](./knowledge.yaml).

## Usage

To try out the databuilder, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/instructlab/knowledge/textbook/history/ibm_history/task.yaml
```

This launches a data generation job by passing seed examples data using the `--task-paths` argument.

## Contributors

**Authors**: Siva Sankalp Patel, Maxwell Crouse, Kshitij Fadnis
