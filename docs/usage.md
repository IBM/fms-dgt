Once you have successfully installed DiGiT, let's move on to creating your first synthetic data.

In this example, we will be generating question answering (QA) pairs demonstrating logical reasoning. Try running the following command from the DiGiT source code directory

```bash
python -m fms_dgt.core --task-paths tasks/core/simple/logical_reasoning/causal/task.yaml --restart
```

??? info
This example uses the `SimpleDataBuilder` as defined [here](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/core/databuilders/simple/generate.py).

    The `SimpleDataBuilder` relies on large language model (LLM) hosted via Ollama to generate data as specified [here](https://github.com/IBM/fms-dgt/blob/main/fms_dgt/core/databuilders/simple/simple.yaml).

You should see the following messages in your terminal

```{ .shell .no-copy hl_lines="32" }
2025-10-28:23:20:11,301 INFO     [utils.py:109] Cannot find prompt.txt. Using default prompt depending on model-family.
2025-10-28:23:20:11,302 INFO     [databuilder.py:540] ***************************************************************************************************
2025-10-28:23:20:11,302 INFO     [databuilder.py:541] 				EPOCH: 1
2025-10-28:23:20:11,302 INFO     [databuilder.py:542] ***************************************************************************************************
Running completion requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.23s/it]
2025-10-28:23:20:18,615 INFO     [generate.py:92] Request 1 took 7.31s, post-processing took 0.00s
2025-10-28:23:20:18,629 INFO     [generate.py:117] Assessing generated samples took 0.01s, discarded 0 instances
2025-10-28:23:20:18,630 INFO     [databuilder.py:581] ***************************************************************************************************
2025-10-28:23:20:18,630 INFO     [databuilder.py:582] 	[EPOCH 1]	GENERATION RESULTS AFTER ATTEMPT 1 (TOTAL ATTEMPTS: 1) # (1)!
2025-10-28:23:20:18,630 INFO     [databuilder.py:588] ***************************************************************************************************
2025-10-28:23:20:18,630 INFO     [databuilder.py:589] Task                                    	Current			Total
2025-10-28:23:20:18,630 INFO     [databuilder.py:595] core/simple/logical_reasoning/causal    	    1     	         1
2025-10-28:23:20:18,630 INFO     [databuilder.py:597] ***************************************************************************************************
2025-10-28:23:20:18,630 INFO     [default.py:189] No more rows left in dataloader. Resetting index to 0.
Running completion requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.26s/it] # (2)!
2025-10-28:23:20:25,907 INFO     [generate.py:92] Request 2 took 7.27s, post-processing took 0.00s
2025-10-28:23:20:25,915 INFO     [generate.py:117] Assessing generated samples took 0.01s, discarded 0 instances
2025-10-28:23:20:25,915 INFO     [databuilder.py:581] ***************************************************************************************************
2025-10-28:23:20:25,915 INFO     [databuilder.py:582] 	[EPOCH 1]	GENERATION RESULTS AFTER ATTEMPT 2 (TOTAL ATTEMPTS: 2) # (3)!
2025-10-28:23:20:25,915 INFO     [databuilder.py:588] ***************************************************************************************************
2025-10-28:23:20:25,915 INFO     [databuilder.py:589] Task                                    	Current			Total
2025-10-28:23:20:25,915 INFO     [databuilder.py:595] core/simple/logical_reasoning/causal    	    1     	         2
2025-10-28:23:20:25,915 INFO     [databuilder.py:597] ***************************************************************************************************
2025-10-28:23:20:25,915 INFO     [databuilder.py:625] Launch postprocessing
2025-10-28:23:20:25,926 INFO     [databuilder.py:371] ***************************************************************************************************
2025-10-28:23:20:25,926 INFO     [databuilder.py:372] 	[EPOCH 1]	POST-PROCESSING RESULTS
2025-10-28:23:20:25,926 INFO     [databuilder.py:373] ***************************************************************************************************
2025-10-28:23:20:25,927 INFO     [databuilder.py:374] Task                                    	Before			After
2025-10-28:23:20:25,927 INFO     [databuilder.py:380] core/simple/logical_reasoning/causal    	    2     	         2
2025-10-28:23:20:25,927 INFO     [databuilder.py:382] ***************************************************************************************************
2025-10-28:23:20:25,927 INFO     [databuilder.py:632] Postprocessing completed
2025-10-28:23:20:25,927 INFO     [task.py:418] Saving final data to output/core/simple/logical_reasoning/causal/final_data.jsonl # (4)!
2025-10-28:23:20:25,927 INFO     [databuilder.py:671] ***************************************************************************************************
2025-10-28:23:20:25,927 INFO     [databuilder.py:674] Generation took 14.63s
2025-10-28:23:20:25,928 INFO     [databuilder.py:288] ***************************************************************************************************
2025-10-28:23:20:25,928 INFO     [databuilder.py:289] 		EXECUTION PROFILER FOR DATABUILDER "simple" # (5)!
2025-10-28:23:20:25,928 INFO     [databuilder.py:290] ***************************************************************************************************
2025-10-28:23:20:25,928 INFO     [databuilder.py:291] Block               	Time (mean ± std)	Peak Memory (mean ± std)	Tokens (Completion Tokens) (Prompt)
2025-10-28:23:20:25,928 INFO     [databuilder.py:270] generator           	7.25 ± 0.01      	117.26 KB ± 15.77 KB    	        152                 2,087
2025-10-28:23:20:25,928 INFO     [databuilder.py:270] validator           	0.0 ± 0.0        	256.0 B ± 0B            	         -                    -
2025-10-28:23:20:25,928 INFO     [databuilder.py:270] rouge_val           	0.0 ± 0.0        	2.47 KB ± 0B            	         -                    -
2025-10-28:23:20:25,928 INFO     [databuilder.py:298] ***************************************************************************************************
```

1.  Results summary after 1st generation attempt
2.  Start of the 2nd generation attempt
3.  Results summary after 2nd generation attempt
4.  Default location for generated synthetic data
5.  Execution summary

Once generation is complete, let's examine the outputs which are saved in the `output/core/simple/logical_reasoning/causal/final_data.jsonl`

```json
{"task_name": "core/simple/logical_reasoning/causal", "is_seed": true, "taxonomy_path": "core/simple/logical_reasoning/causal", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Explain the causal relationship between smoking and lung cancer.", "input": "", "output": "Smoking is a major causal factor in the development of lung cancer. The chemicals in tobacco smoke can damage the DNA in lung cells, leading to uncontrolled cell growth and the formation of tumors.", "document": null}
{"task_name": "core/simple/logical_reasoning/causal", "is_seed": true, "taxonomy_path": "core/simple/logical_reasoning/causal", "task_description": "To teach a language model about Logical Reasoning - causal relationships", "instruction": "Identify the causal relationship in the following scenario and explain it.", "input": "\"After increasing the number of police patrols in a neighborhood, the crime rate decreased significantly.\"", "output": "The causal relationship is that increasing the number of police patrols led to a decrease in the crime rate. The presence of more patrols likely acted as a deterrent to potential criminals, resulting in fewer crimes being committed.", "document": null}
```
