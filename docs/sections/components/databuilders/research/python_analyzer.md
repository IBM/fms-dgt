# Python Analyzer



Databuilder for executing Python code.



## Setup



This databuilder requires guardx to be set up (https://github.ibm.com/sec-watsonx/guardx). The library container images must be built before importing and using the library.



```shell

pip install ".[guardx]"



# optional

podman machine init

podman machine start



# initialize guardx

guardx init --client [docker|podman] #sudo guardx init --client [docker|podman]

```



## Data specification



This data builder supports generation defining the following parameters:



### Parameters



- `prompt`: The label of the prompt within python_analyzer/prompts to use for generation.



### Seed data required fields



- `question`: Question to generate code for.

- `answer`: The code that can be evaluated to get an answer to the question.



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/python_analyzer/datetime/task.yaml).



## Evaluation



TBD



## How to run



To execute this databuilder, run the following command



```bash

python -m fms_dgt.research --task-paths ./tasks/research/python_analyzer/

```
