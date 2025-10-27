# Salesforce Email Generation



Data builder used to generate cold-call emails for various products. The slight twist with this databuilder is how the email introductions were handled.



We found that most LLMs would use a very generic introduction (often "I hope this email finds you well", but other similar introductions as well), even when explicitly instructed not to, which made the emails sound very fake. The rest of the email following the generic introduction was fairly good, however, so we decided to lean into this trait.



We explicitly instructed the LLM to start each email with "I hope this email finds you well.", with the assumption that it would then start each email with exactly those words. Then, in the LLM-generated email we stripped that opening sentence and simply returned the remainder of the email. This led to fairly good results, where the emails generated practically never started with a generic introduction.



## Data specification



This data builder supports generation defining the following parameters:



### Parameters



- `task_name`: name of task.

- `created_by`: creator of the task.

- `task_description`: description of the task.

- `data_builder`: data builder name.

- `seed_examples`: examples used as seed data for generation.

- `instruction_format`: format to be used to produce instruction-tuning pairs for training.



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/granite/salesforce_email/task.yaml)



### Seed data required fields



- `company`: name of company.

- `product_name`: name of product.

- `product_description`: description of the product being sold by the company



Refer to [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/granite/databuilders/salesforce_email/task.py)



## Evaluation



TBD



## How to run



To execute this databuilder, run the following command



```bash

python -m fms_dgt.granite --task-path ./tasks/granite/salesforce_email/task.yaml --restart --seed-batch-size 5

```
