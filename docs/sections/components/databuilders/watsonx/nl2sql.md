# SQL Data Generator



The `nl2sql` package provides a data builder class, **`Nl2SqlDataBuilder`**, to generate synthetic SQL data triplets: natural language utterances, corresponding SQL queries, and the relevant database schema. It optionally leverages query logs, ground-truth samples, and additional metadata.



## Data Specification



This data builder supports generation by defining the following parameters:



### Required



- `created_by`: Creator of the task.

- `task_description`: Description of the task.

- `data_builder`: Set to `nl2sql`.

- `generators`: Type of generators to use (currently supported: lm, components, template)

- `database -> schema`: DDL statement representing the database schema. Comments within the DDL schema to describe valid values or metadata are supported.



### Optional



- `database -> connection`: Metadata about the database. Each `connection` field must have `type` field representing type of databases to connect to (currently supported: sqlite, postgres)



> [!IMPORTANT]

>

> - If `database -> connection -> type` is set to `sqlite` then `SQLITE_DATABASE_PATH` environment variable must be set.

>

> - If `components` and/or `template` generators are used then `database -> connection` field must be configured.

>

> - If `template` generator is used then `NL2SQL_TEMPLATE_FILE_PATH` environment variable must be set.



An example of the input data configuration can be found



- [Standard](../../../../data/watsonx/nl2sql/standard/qna.yaml)

- [Advanced](../../../../data/watsonx/nl2sql/advanced/qna.yaml)

- [Template Based](../../../../data/watsonx/nl2sql/template_based/qna.yaml)



## Generators and Validators



Default configurations for generators and validators used by the data builder are specified in the [nl2sql.yaml](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/nl2sql/nl2sql.yaml) file.



### Generators



The data generation process uses a pre-configured Language Model (LLM) to create synthetic natural language utterances and corresponding SQL queries. The generator is defined in the configuration under the blocks section and can be customized for different models and parameters.



### Validators



Validators are applied to generated data samples to ensure quality and correctness. Examples of validators include:



- **`sql_syntax_validator`:** Validates the syntax of generated SQL queries against the provided database schema to ensure they are syntactically correct.

- **`sql_execution_validator`:** Tests the correctness of the generated SQL by executing it against an in-memory test database. Queries that fail to execute successfully are filtered out.



## Installation and Getting Started



This section will help you quickly set up and run a sample.



### Prerequisites



Ensure your environment is set up as described in the [main README](../../../../../README.md).  

Next, install the optional SQL dependency:



```console

pip install ".[nl2sql]"

```



### Running a Basic Sample



To run a basic sample using watsonx, set the following environment variables:



```yaml

WATSONX_API_KEY=<watsonx api key goes here>

WATSONX_PROJECT_ID=<watsonx project id goes here>

```



These environment variables are required to interact with the LLM service and must be set before running the sample.



Then execute the following command:



```console

python -m fms_dgt.watsonx \

    --data-paths data/watsonx/nl2sql/standard/qna.yaml \

    --restart-generation

```



- **`--data-paths`**: Specifies the path to the YAML data configuration file.

- **`--restart-generation`** (optional): Include this flag to overwrite existing results.



This command processes the default YAML configuration and generates utterances and corresponding SQL queries based on the provided schema and metadata.



The generated output is saved in the folder `output/nl2sql_standard/data.jsonl`, along with the corresponding task card.
