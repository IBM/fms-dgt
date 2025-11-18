# Loading seed examples from a file

In synthetic data generation, a common pattern involves using seed examples (also known as in-context learning (ICL) examples) in the prompt provided to the teacher model. To support this, DGT offers a base class called `GenerationTask`, which allows databuilder developers to specify seed examples either directly in the task YAML file or through external files in formats such as .jsonl, .json, or .parquet.

Continuing with our earlier example of generating geography-related question-answer pairs, you can find the seed examples defined in the task YAML file.

Let’s take a closer look at its contents:

```{.yaml .no-copy title="tasks/public/examples/qa/task.yaml" hl_lines="13-53" }
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/geography_qa
task_description: A task for geography question-answering
created_by: IBM

data_builder: public/examples/geography_qa

######################################################
#                   RESERVED FIELDS
######################################################
seed_examples:
  - question: What is the name of the tallest mountain in the world?
    answer: Mount Everest
  - question: What are the names of the five oceans of the world?
    answer: Atlantic, Pacific, Indian, Arctic, and the Antarctic
  - question: What is the largest desert in the world?
    answer: The Antarctic Desert
  - question: What is the longest river in Africa?
    answer: The Nile River
  - question: What is the smallest country in the world by land area?
    answer: Vatican City
  - question: What is the capital of Australia?
    answer: Canberra
  - question: What is the longest mountain range in South America?
    answer: The Andes mountain range
  - question: What are well known dense forests around the world?
    answer: Amazon Rainforest, Congo Basin, La Mosquitia jungle are few examples of dense rainforests with thick, nearly impenetrable vegetation.
  - question: Which country has the largest population in the world?
    answer: China
  - question: What American city is the Golden Gate Bridge located in?
    answer: San Francisco
  - question: What is the capital of Mexico?
    answer: Mexico City
  - question: What is the name of the largest ocean in the world?
    answer: The Pacific Ocean
  - question: What country has the most natural lakes?
    answer: Canada
  - question: What continent is Britain part of?
    answer: Europe
  - question: Which European country is closest to Africa?
    answer: Spain
  - question: In what country is the Taj Mahal located?
    answer: India
  - question: What do you call a chain of mountains?
    answer: A range
  - question: How many time zones does Russia have?
    answer: 11
  - question: What is the name of the only tropical rainforest in the United States?
    answer: Puerto Rico’s El Yunque National Forest
  - question: What country formerly ruled Iceland?
    answer: Denmark
```

Instead of defining seed examples directly in the task YAML file, let’s specify them in a separate .jsonl file. This approach improves modularity, making the seed examples easier to manage and reuse across different tasks.
Save the following file as seed_examples.jsonl in the data/public/examples/qa directory.

```{.json title="seed_examples.jsonl"}
{"question": "What is the name of the tallest mountain in the world?", "answer": "Mount Everest"}
{"question": "What are the names of the five oceans of the world?", "answer": "Atlantic, Pacific, Indian, Arctic, and the Antarctic"}
{"question": "What is the largest desert in the world?", "answer": "The Antarctic Desert"}
{"question": "What is the longest river in Africa?", "answer": "The Nile River"}
{"question": "What is the smallest country in the world by land area?", "answer": "Vatican City"}
{"question": "What is the capital of Australia?", "answer": "Canberra"}
{"question": "What is the longest mountain range in South America?", "answer": "The Andes mountain range"}
{"question": "What are well known dense forests around the world?", "answer": "Amazon Rainforest, Congo Basin, La Mosquitia jungle are few examples of dense rainforests with thick, nearly impenetrable vegetation."}
{"question": "Which country has the largest population in the world?", "answer": "China"}
{"question": "What American city is the Golden Gate Bridge located in?", "answer": "San Francisco"}
{"question": "What is the capital of Mexico?", "answer": "Mexico City"}
{"question": "What is the name of the largest ocean in the world?", "answer": "The Pacific Ocean"}
{"question": "What country has the most natural lakes?", "answer": "Canada"}
{"question": "What continent is Britain part of?", "answer": "Europe"}
{"question": "Which European country is closest to Africa?", "answer": "Spain"}
{"question": "In what country is the Taj Mahal located?", "answer": "India"}
{"question": "What do you call a chain of mountains?", "answer": "A range"}
{"question": "How many time zones does Russia have?", "answer": "11"}
{"question": "What is the name of the only tropical rainforest in the United States?", "answer": "Puerto Rico’s El Yunque National Forest"}
{"question": "What country formerly ruled Iceland?", "answer": "Denmark"}
```

Now, we can reference the newly created seed_examples.jsonl file in our task YAML as shown below:

```{.yaml title="tasks/public/examples/qa/task.yaml" hl_lines="13-15" }
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/geography_qa
task_description: A task for geography question-answering
created_by: IBM

data_builder: public/examples/geography_qa

######################################################
#                   RESERVED FIELDS
######################################################
seed_datastore:
    type: default
    data_path: ${DGT_DATA_DIR}/public/examples/qa/seed_examples.jsonl
```
