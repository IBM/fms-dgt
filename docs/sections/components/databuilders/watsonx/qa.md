# Generation of Table QA pairs



**[Task Specification](#task-specification) | [Generators](#generators) | [Validators](#validators) | [Usage](#usage) | [Contributors](#contributors)**



Data builder used for generating instruction-response pairs for Table QA task, given a set of tables.



## Task specification



This data builder supports generation defining the following parameters:



### Required



- `created_by`: creator of the task.

- `task_description`: description of the task.

- `data_builder`: table_qa

- `questions`: list of relevant questions for the given table

- `answers`: list of answers for the corresponding questions

- `table`: table for the corresponding question and answer pairs

- `dir_path`: path where tables are stored (we assume tables are stored as CSV files)

- `prompt_format`: format to serialize tables while prompting (can be one of `pandas_dataframe`, `json`, `html`, `xml`, `latex`, `to_string`, `nl_sep`)

- `output_format`: format to serialize tables while saving in generated instructions

- `instruction_format`: format in which the generated QA pairs for a given table, are to be serialised.

- `max_num_questions_per_table`: maximum number of QA pairs to be generated per table.



An example can be found [here](../../../../../tasks/research/tables/qa/task.yaml).



## Generators and validators



Default configuration for generator and validator used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/tables/qa/table_qa.yaml).



### Generators



- `mistralai/mixtral-8x22B-instruct-v0.1` via `rits`.



### Validators



- `mistralai/mixtral-8x22B-instruct-v0.1` via `rits`: LLM Judge for validating generated questions and answers

- `rouge_scorer`: Deduplicator that removes elements of generated data that are too rouge-similar.



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/watsonx/tables/qa/task.yaml --restart-generation

```



This launches a data generation job by passing seed examples data using the `--task-paths` argument.



### Explanation



As you can see there's a `data_builder` field in the [task.yaml](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/tables/qa/task.yaml) file that points to the databuilder to be used for this task.



```yaml

created_by: IBM Research

data_builder: table_qa

tables:

  dir_path: ...

  prompt_format: json

  output_format: json



instruction_format: ...

max_num_questions_per_table: ...

seed_examples:

  - table_format: ...

    table: ...

    questions: ...

    answers: ...

  - table_format: ...

    table: ...

    questions: ...

    answers: ...

```



This particular task, given a table and seed examples, generates QA pairs. More specifically, the seed examples are passed to the `__call__` method in [`generate.py`](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/tables/qa/generate.py).



By default, the output (\*.jsonl) is generated in sub-directories under output/tables/qa. Here's a sample output:



```json

{

  "sys_prompt": "You are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.",

  "input": "Write your answer to the question given the information in the following table table: pd.DataFrame({\n{\"Date\": [\"February 9\", \"February 11\", \"February 18\", \"March 11\", \"March 26\", \"May 10\", \"May 13\", \"May 30\", \"May 30\", \"June 23\", \"July 5\", \"July 10\", \"September 12\", \"October 3\", \"October 10\"], \"Race\": [\"Tour of Qatar, Stage 3\", \"Tour of Qatar, Stage 5\", \"Tour of Oman, Stage 5\", \"Tirreno\\u2013Adriatico, Stage 2\", \"Volta a Catalunya, Stage 5\", \"Giro d'Italia, Stage 3\", \"Giro d'Italia, Stage 5\", \"Giro d'Italia, Premio della Fuga\", \"Tour of Belgium, Overall\", \"Halle\\u2013Ingooigem\", \"Tour de France, Stage 2\", \"Tour de France, Stage 7\", \"Vuelta a Espa\\u00f1a, Stage 15\", \"Circuit Franco-Belge, Stage 4\", \"G.P. Beghelli\"], \"Competition\": [\"UCI Asia Tour\", \"UCI Asia Tour\", \"UCI Asia Tour\", \"UCI World Ranking\", \"UCI ProTour\", \"UCI World Ranking\", \"UCI World Ranking\", \"UCI World Ranking\", \"UCI Europe Tour\", \"UCI Europe Tour\", \"UCI World Ranking\", \"UCI World Ranking\", \"UCI World Ranking\", \"UCI Europe Tour\", \"UCI Europe Tour\"], \"Rider\": [\"Tom Boonen (BEL)\", \"Tom Boonen (BEL)\", \"Tom Boonen (BEL)\", \"Tom Boonen (BEL)\", \"Davide Malacarne (ITA)\", \"Wouter Weylandt (BEL)\", \"J\\u00e9r\\u00f4me Pineau (FRA)\", \"J\\u00e9r\\u00f4me Pineau (FRA)\", \"Stijn Devolder (BEL)\", \"Jurgen Van de Walle (BEL)\", \"Sylvain Chavanel (FRA)\", \"Sylvain Chavanel (FRA)\", \"Carlos Barredo (ESP)\", \"Wouter Weylandt (BEL)\", \"Dario Cataldo (ITA)\"], \"Country\": [\"Qatar\", \"Qatar\", \"Oman\", \"Italy\", \"Spain\", \"Netherlands\", \"Italy\", \"Italy\", \"Belgium\", \"Belgium\", \"Belgium\", \"France\", \"Spain\", \"Belgium\", \"Italy\"], \"Location\": [\"Mesaieed\", \"Madinat Al Shamal\", \"Sultan Qaboos Sports Complex (Muscat)\", \"Montecatini Terme\", \"Cabac\\u00e9s\", \"Middelburg\", \"Novi Ligure\", NaN, NaN, \"Ingooigem\", \"Spa\", \"Les Rousses\", \"Lagos de Covadonga\", \"Tournai\", \"Monteveglio\"]}},\nindex=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) question: which race did Sylvain Chavanel (FRA) win in Belgium?",

  "label": "Tour de France, Stage 2",

  "is_truncated": "False"

}

```



## Evaluation



TBD



## Contributors



**Author and Maintainer**: Deepak Vijaykeerthy



**DiGiT**: Siva Sankalp Patel, Kshitij Fadnis
