# Magpie Transformation

Data builder used to transform existing datasets with Magpie tags, deduplication and/or filtering.

## Setup

Please ensure you have installed the DGT repo by doing

```bash
pip install -e ".[all]"
```

## Task specifications

### Required in task.yaml [here](../../../../tasks/public/magpie/task.yaml)

- data:
  - `type`: default
  - `data_path`: input data file path in .jsonl format

## Databuilder specifications

### Required in data builder config [here](./magpie.yaml).

By default the magpie.yaml config has all three steps of Magpie ie Tagging, Deduplication and Filtering based on the previous steps enabled. The three corresponding blocks are namely tagger, dedup and filter respectively. You can choose to keep any one of them or all of them.

The most important part that needs to be changed in each of the block that is going to be used in this config, is the `input_map`

> **_NOTE:_** Do not use reserved keywords as field names 'magpie_input', 'magpie_output', 'magpie_mt_input' or 'magpie_tags' in your data file.
> These are used within Magpie and will be overriden.

**Use of the `input_map` is show in each scenario below:**

#### Tagger

##### Single Input and Output

- `name`: tagger
- `input_map` :
  - `question`: magpie_input
  - `answer`: magpie_output

shows that in the taggging step, the code is expecting to find `question` and `answer` in the data (`seed_datastore`) and will treat them as input and its corresponding output respectively in order to assess their quality.

##### Multi Turn Conversations

In the case of multi-turn conversations, the conversations should follow OpenAI chat format, for example

```
messages : {'role': 'user', 'content': 'What is 1+1', 'role': 'asistant', 'content': '1+1 is 2'}
```

The input_map should look like this:

- `name`: tagger
- `input_map` :
  - `messages`: magpie_mt_input

> **_NOTE:_** In the case of multi-turn only messages with the field 'content'/'text'/'value' are supported by Magpie. Others will be skipped
> **_NOTE:_** In the case of multi-turn only 'user' and 'assistant' messages are supported by Magpie. Others will be skipped

`lm_config` under tagger can be changed to change the source of model of llm (eg. ollama) and batch_size for processing.

`tasks : ["quality", "sample_quality", "difficulty"] ` are the default tagging prompts . Any of these can be removed by overriding `tasks:[]` under the magpie tagger block .

"classification" is an optional task as well.

#### Dedup

Similarly `input_map` in the `dedup` block needs the name of the field for the input, the field containing a unique id (optional) and the field containing the tags from the tagger step (optional).
If either of the optional fields are not present then it will create the fields 'id' and 'magpie_tags'.

- `name`: dedup
- `input_map` :
  - `question`: magpie_input
  - `sample_id`: id (optional)
  - `magpie_tags`: magpie_tags (optional)

#### Filter

For the filter block, it needs the names of field for input, output, tags and unique id.

- `name`: filter
- `input_map` :
  - `question`: magpie_input
  - `answer`: magpie_output
  - `magpie_tags`: field
  - `sample_id`: id

If `id` does not exist in the data and was created in the dedup step, make sure to update the input map for filter as follows:

- `input_map` :
  - `question`: magpie_input
  - `answer`: magpie_output
  - `magpie_tags`: field
  - `id`: id

`filter_criteria:` can be used to change the filter criteria
`remove_duplicate` True/False (default=True)

## How to run

To execute this databuilder, run the following command

```bash
python -m fms_dgt.public --task-path ./tasks/public/magpie/task.yaml --output-dir output/magpie_output
```

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

## Output

Under the output folder, final_data.jsonl will contain the data file in the same format as input data file , tagged, deduped and filtered if enabled.

The file filter.jsonl under the blocks folder inside output will contain the data samples that were filtered along with their tags and dedup info.

## Filtering

### Tagging

Tagging the input and output (in case of single turn) or the conversation (in case of multi turn) in terms of :

```
quality (question) : [
"very poor",
"poor",
"average",
"good",
"excellent",
]

sample_quality score(question and response) : ["1", "2", "3", "4", "5"]

difficulty : [
"very easy",
"easy",
"medium",
"hard",
"very hard",
]

classification of task: []

knowledge : []
```

### Deduplication

It calculates the most similar sample based on distance using sentence transformer and outputs the id of the most similar to the sample.

### Filtering

Filters the magpie tagged data based on certain criteria. These are mentioned in `filter_criteria` under the filter block.

```
`input_quality` in ["good", "excellent"]
`judge_quality_score` in ["4","5"]
`min_similar_uuid` is None or `min_similar_uuid`=`id`
```

## Notice

Modified version of [**Magpie**](https://magpie-align.github.io/) enabled to use opensource models.
