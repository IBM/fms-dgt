# Try-it

Once you have successfully installed DiGiT, let's move on to creating your first synthetic data. 

In this example, we will be generating question answering (QA) pairs demonstrating logical reasoning. Try running the following command from the DiGiT source code directory

```bash
# If you have set up a WATSONX_API_KEY
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --restart

# If you have set up a RITS_API_KEY
python -m fms_dgt.core --task-paths ./tasks/core/logical_reasoning/causal --restart --include-namespace granite --config ./configs/rits_simple_db.yaml

# Alternatively, since the `research` namespace loads the granite namespace as well, you can run with "fms_dgt.research" or "fms_dgt.granite", e.g.,
python -m fms_dgt.research --task-paths ./tasks/core/logical_reasoning/causal --restart --config ./configs/rits_simple_db.yaml
```

> **IMPORTANT**
> - This example uses the `SimpleInstructDataBuilder` as defined in `./fms-dgt/fms_dgt/databuilders/generation/simple/generate.py`. 
>
> - The `SimpleInstructDataBuilder` relies on large language model (LLM) hosted on WatsonX.AI to generate data (`./fms-dgt/fms_dgt/databuilders/generation/simple/simple.yaml`).

You should see the following messages in your terminal
```shell
2025-01-27 12:50:15,558 INFO worker.py:1832 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265 
2025-01-27:12:50:20,745 INFO     [rouge_scorer.py:83] Using default tokenizer.
2025-01-27:12:50:20,745 INFO     [utils.py:109] Cannot find prompt.txt. Using default prompt depending on model-family.
Running generate_batch requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.33s/it]
2025-01-27:12:50:29,452 INFO     [generate.py:106] Request 1 took 8.67s, post-processing took 0.01s█████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.20s/it]
2025-01-27:12:50:29,452 INFO     [generate.py:131] Assessing generated samples took 0.00s, discarded 0 instances
2025-01-27:12:50:29,467 INFO     [databuilder.py:275] Generated 2 data in this iteration, 2 data overall
Running generate_batch requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.93s/it]
2025-01-27:12:50:38,275 INFO     [generate.py:106] Request 2 took 8.79s, post-processing took 0.00s█████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.72s/it]
2025-01-27:12:50:38,276 INFO     [generate.py:131] Assessing generated samples took 0.00s, discarded 0 instances
2025-01-27:12:50:38,278 INFO     [databuilder.py:275] Generated 3 data in this iteration, 5 data overall
Running generate_batch requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.52s/it]
2025-01-27:12:50:44,393 INFO     [generate.py:106] Request 3 took 6.09s, post-processing took 0.00s                                                         | 1/4 [00:06<00:18,  6.08s/it]
2025-01-27:12:50:44,394 INFO     [generate.py:131] Assessing generated samples took 0.00s, discarded 0 instances
Running generation tasks: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:23<00:00, 23.64s/it]2025-01-27:12:50:44,397 INFO     [databuilder.py:275] Generated 4 data in this iteration, 9 data overall
2025-01-27:12:50:44,397 INFO     [databuilder.py:287] Launch postprocessing
2025-01-27:12:50:44,402 INFO     [databuilder.py:391] Postprocessing completed with 9 instances remaining for task causal_logical_reasoning
2025-01-27:12:50:44,402 INFO     [databuilder.py:289] Postprocessing completed
Running generation tasks: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:23<00:00, 23.64s/it]
2025-01-27:12:50:44,403 INFO     [databuilder.py:308] Generation took 23.66s
```

Once generation is complete, let's examine the outputs which are saved in the file (`output/causal_logical_reasoning/data.jsonl`)

```json
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "If John drives to work in 30 minutes when there is no traffic, how long will it take if there is heavy traffic and he travels at 75% of his normal speed?", "input": "Assume John's normal driving speed is 40 miles per hour.", "output": "With heavy traffic and a reduced speed of 75%, John's speed would be 30 miles per hour. Given the same distance, this would increase his travel time to 40 minutes.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Given the causal relationship that \"eating too much junk food leads to obesity,\" is it valid to infer that \"every obese person eats too much junk food?\"", "input": "", "output": "No, the inference is not valid. Not every obese individual may eat too much junk food; other factors such as genetics and metabolism may contribute to obesity.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "When someone cooks, they must follow a recipe. What if a chef invents a new dish on the spot? Would the chef be breaking the rule?", "input": "A creative chef decides to make up a brand new dish without a recipe.", "output": "No, in this case the chef would not be breaking the rule, as they are inventing a dish rather than following a preexisting recipe.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Considering the causal relationship between \"smoking cigarettes causing lung cancer,\" are there any potential confounding factors?", "input": "", "output": "Yes, genetic factors and exposure to radiation, among other things, could act as confounding factors in the relationship between smoking and lung cancer.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Suppose that dropping a ball from the Empire State Building will cause it to fall towards the ground. If you drop a bowling ball from the Eiffel Tower, which direction will it go?", "input": "", "output": "Just like the ball dropped from the Empire State Building, the bowling ball will also fall towards the ground.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "In general, people tend to take an umbrella when it's raining outside. If it starts raining while you are shopping, would you need an umbrella?", "input": "", "output": "Yes, if it starts raining while you are shopping, you would need an umbrella to keep yourself dry.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "If water boils at 100 degrees Celsius at sea level, at what temperature does it boil on top of Mount Everest, which is approximately 8,848 meters above sea level?", "input": "", "output": "Water's boiling point decreases as altitude increases. At the reduced atmospheric pressure of Mount Everest's 8,848 meters, water boils around 70 degrees Celsius.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Determine the logical cause-and-effect relationship between the following facts: \"Since Mary watered the plants yesterday, they wilted today.\"", "input": "Mary watered the plants on Monday, but they didn't wilt until Tuesday.", "output": "The wilting of the plants on Tuesday was not directly caused by Mary watering the plants on Monday. Mary's watering on Monday might have been excessive, and the wilting could have been due to over-watering. Other factors such as sunlight, temperature, and soil quality could also possibly have contributed to the plants' wilting.", "document": null}
{"task_name": "causal_logical_reasoning", "taxonomy_path": "causal_logical_reasoning", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Imagine a scenario where \"Plant A\" is watered daily and it thrives, while \"Plant B\" is watered daily but wilts. Can water be both the cause of growth for Plant A and the cause of wilting for Plant B?", "input": "", "output": "Yes, water could be the cause of growth for Plant A and cause of wilting for Plant B due to the difference in preferences between two species.", "document": null}
```


