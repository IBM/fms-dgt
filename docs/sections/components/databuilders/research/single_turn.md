# Retrieval Augmented Generation (RAG)



**[RAG Question Patterns](#rag-question-patterns) | [Output](#output) | [Retrieval](#retrieval) | [Task Specification](#task-specification) | [Generators](#generators) | [Rating Engines](#rating-engines) | [Post-Processors](#post-processors) | [Usage](#usage) | [Contributors](#contributors)**



Data builder used for generating single turn question answer pairs driven by examples.



It generates data using the open-source Mixtral-7x8B model as the teacher (generator), prompt templates and magpie based validators.



> [!NOTE]  

> Please make sure you install additional `granite` & `rag` depedencies.

>

> ```bash

> pip install -e ".[granite,rag]"

> ```



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



## RAG Question Patterns



Current implementation supports generation of following question types. The question-answer generating prompt templates for these question types are available [here](./templates).



- Direct - Simple questions that are easily answerable from documents. This includes question types such as factoid, binary, instructional, etc.

- Keyword - Underspecified short questions that are commonly asked by users who associate with chatbots.

- Comparative - Questions that ask the agent to compare/contrast two entities or events found in the document.

- Unanswerable - In-domain questions that are unanswerable from a given document.

- Chitchat - Informal questions that we see in out day-to-day conversations.



These patterns can be extended as required by any application, by adding your own template into the directory [here](./templates).



### Output



By default, the output (`*.jsonl`) is generated in sub-directories under `output/rag`. Please see a sample output (output/rag/data->rag->direct/generated_instructions.jsonl) for a `direct` question type task on dummy documents.



```JSON

{"task_name": "data->rag->direct", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "messages": [{"role": "user", "content": "Can employees work from home according to the company's policy? If so, on which day of the week is it?", "timestamp": 1723562489, "metadata": {"question type": "direct"}}, {"role": "assistant", "content": "Yes, employees can work from home on Fridays according to the company's policy.", "timestamp": 1723562489, "contexts": [{"document_id": "document 5", "text": "This is a sample document 5."}]}]}

{"task_name": "data->rag->direct", "task_description": "Generate single turn direct style grounded conversation over one or more documents.", "messages": [{"role": "user", "content": "What are the steps to bake a cake and at what temperature should the oven be preheated?", "timestamp": 1723562537, "metadata": {"question type": "direct"}}, {"role": "assistant", "content": "To bake a cake, preheat the oven to 350 degrees Fahrenheit. Gather all necessary ingredients, including flour, sugar, baking powder, salt, milk, eggs, and vanilla extract. Mix the dry ingredients together in one bowl and the wet ingredients in another. Gradually combine the two mixtures, being careful not to overmix. Pour the batter into a greased cake pan and bake for 25-30 minutes, or until a toothpick inserted into the center comes out clean.", "timestamp": 1723562537, "contexts": [{"document_id": "document 4", "text": "This is a sample document 4."}]}]}



```



Each line on the output file is a single RAG data point stored in a JSON containing following fields



- `task_name`: specifies task responsible for producing this particular instance

- `task_description`: description of the task from the corresponding task's `qna.yaml`

- `messages`: array containing alternating user and assistant turns. Each message have following fields

  - `role`: Either `user` or `assistant`

  - `content`: question is the speaker is `user` else answer for the previous user question

  - `timestamp`: timestamp when message was generated

  - `contexts`: Only available on `agent` messages. It contains information about documents used to generate the current turn's user and agent texts.

  - `metadata`: additional information relevant to particular turn. For example, in case of `user` message, `question_type` is added to `metadata` field.



## Retrieval



This implementation currently supports two retriever blocks.



- [JSONRetriever](../../../retrievers/unstructured_text/json_retriever.py) (`json`) - Supports retrieving documents from a json file that contains all the documents in the following format



  ```JSON

  [

      {

      "id": "document 1",

      "text": "This is a sample document 1."

  },

  {

      "id": "document 2",

      "text": "This is a sample document 2."

  },

  {

      "id": "document 3",

      "text": "This is a sample document 3."

  },

  .

  .

  .

  ]



  ```



- [ElasticRetriever](../../../retrievers/unstructured_text/elastic_retriever.py) (`elastic`) - Supports retrieving documents that are indexed and stored at an Elasticsearch index



  - You will need to provide a valid endpoint to your hosted Elasticsearch in `retriever` portion of the task YAML (qna.yaml) or via environment variable (`ES_ENDPOINT`)



  - You can authenticate either using `username` and `password` or `api_key` configuration. You must specify them in `retriever` portion of the task YAML (qna.yaml) or via environment variables (`ES_USERNAME`, `ES_PASSWORD`, `ES_API_KEY`) as per your authentication preference



  - You must specify `index_name` property in `retriever` portion of the task YAML



  - Additionally, if you have SSL fingerprint enabled in Elasticsearch configuration, then please set `ES_SSL_FINGERPRINT` environment variable with appropriate value for a secured connection



⚠️ NOTE: Currently, we only support randomly selecting a document using retriever for generating a RAG data point.



## Task specification



This data builder supports generation defining the following parameters:



### Required



- `created_by`: creator of the task

- `task_description`: description of the task

- `data_builder`: rag/single_turn

- `pattern`: question type. For example, `direct`, `comparative`, `unanswerable` etc.

- `retriever`: retriever configuration



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/rag/single_turn/direct/task.yaml).



## Generators, rating engines and post-processors



Default configuration for generator and validator used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/rag/single_turn/rag.yaml).



### Generators



- `mistralai/mixtral-8x22B-instruct-v0.1` via `rits`.



### Rating Engines



- `mistralai/mixtral-8x22B-instruct-v0.1` via `rits`: Magpie tagger for rating generated questions and answers



### Post-Processors



- `Magpie Distance`: Deduplicator that removes elements of generated data that are too similar.

- `Magpie Filter`: Removes elements of generated data that does not meet quality criteria.



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/research/rag/single_turn/direct/task.yaml --restart-generation

```



## Contributors



**Author and Maintainer**: Kshitij Fadnis, Chulaka Gunasekara
