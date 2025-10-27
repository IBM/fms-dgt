# Knowledge Generation



Data builder used for generating instruction-response pairs driven by examples in the knowledge and foundational skills branches of InstructLab Taxonomy.



It generates data using the open-source Mixtral-7x8B model as the teacher (generator and evaluator) and prompt templates.



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



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/knowledge_sdg/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task

- `task_description`: (str) description of the task

- `domain`: (str) domain of the document

- `data_builder`: (str) must be `knowledge_sdg`

- `taxonomy_path`: (str, optional) used to indicate part of instruct-lab taxonomy data is produced from (defaults to data file path)



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/instructlab/knowledge/textbook/history/ibm_history/task.yaml).



## Data specification



Tasks executed by this data builder require seed examples and documents that use the following parameters



### Parameters



Seed examples can be provided through the `seed_examples` field with the following parameters:



- `question`: (str) question for model to answer

- `answer`: (str) answer that model should produce



And documents can be passed using in two ways:



1. Using the `include` directive with the `documents` key:



- `documents`: (Dict) key-value pairs where keys are document names or groups and values are file paths or glob patterns (supported files types are `.md`, `.jsonl`, `.txt`)

- `chunk_size`: (int) documents will be chunked to a maximum of this size (defaults to 600)



For example:



```yaml

include:

  documents:

    photosynthesis: $DGT_DATA_DIR/watsonx/instructlab/knowledge/textbook/science/biology/photosynthesis/photosynthesis.md

    structure_of_matter: $DGT_DATA_DIR/watsonx/instructlab/knowledge/textbook/science/physics/static_electricity/structure_of_matter.txt



chunk_size: 800

```



2. Using the `knowledge_datastore` key:



- `type`: (str) `default_knowledge_datastore` if passing documents from local filesystem or `lh_knowledge_datastore` if using lakehouse

- `store_name` (str): Name of datastore

- `data_path` (str): Path to file/folder containing data. Can be a glob pattern.

- `data_format`: (str) File format. Supported formats - `md`, `txt`, `jsonl`, `parquet`.

- `fields_to_extract` (list): List of field/columns names to extract from file records. This is required when `data_format` is `jsonl` or `parquet`.

- `content_key` (str): Field name to use as main content. Required when `data_format` is `jsonl` or `parquet`.



For example:



```yaml

# from local filesystem

knowledge_datastore:

  type: default_knowledge_datastore

  store_name: documents

  data_path: $DGT_DATA_DIR/watsonx/instructlab/knowledge/textbook/science/**

  data_format: md

```



```yaml

# from lakehouse

knowledge_datastore:

  type: lh_knowledge_datastore

  namespace: ibmdatapile.internet

  store_name: angular_data_request_angular_dev_website

  environment_name: PROD

  fields_to_extract:

    - document_id

    - contents

  content_key: contents

```



- If using lakehouse, make sure to install additional dependencies using `pip install ".[dmf-lakehouse]"`. Also add `LAKEHOUSE_TOKEN` to .env.



An example can be found in the `seed_examples` and `include` sections in [here](../../../../../tasks/watsonx/instructlab/knowledge/textbook/).



## Databuilder specification



Default configuration for generator and validator used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/knowledge_sdg/knowledge_sdg.yaml).



- `generator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `watsonx`

- `validator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `lm_judge` and `watsonx`



In addition, we also pass the following parameters:



- `templates_path` (Optional[str]): Path to folder containing prompt templates.

- `num_prompt_instructions` (Optional[int]): Number of ICL examples per prompt.

- `num_docs_per_iteration` (Optional[int]): Number of documents to use per iteration of SDG run. Defaults to using all documents.



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/instructlab/knowledge/textbook/history/ibm_history/task.yaml

```



This launches a data generation job by passing seed examples data using the `--task-paths` argument.



By default, the generation engine used is WatsonX. To use RITS, we override the databuilder config:



```

python -m fms_dgt.research --task-paths ./tasks/watsonx/instructlab/knowledge/textbook/history/ibm_history/task.yaml --config-path ./configs/knowledge_sdg_rits.yaml

```



## Contributors



**Authors**: Siva Sankalp Patel, Maxwell Crouse
