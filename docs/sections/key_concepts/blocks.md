# Blocks

As mentioned previously, in the data builder `__call__` function, you will make use of blocks, which are the components in DGT that do the heavy lifting or contain specialized algorithms that would be useful to share across teams. While there are few constraints as to what types of programs can be written as blocks, for the most part, we have found that most blocks will fall under the general category of "generators" and "validators". We have provided a number of LLM-based generators (e.g., [OpenAI](https://github.com/openai/openai-python), [vLLM](https://github.com/vllm-project/vllm), etc.). To use a specific block in your databuilder, you need to specify it in both the YAML and in the `generate.py` file as a attribute of the task's class. From a design standpoint, we aim to keep all multiprocessing and parallelism contained to the generators and the validators, i.e., **not** in the `__call__` function. By defining these ahead of time and restricting heavy operations to these objects, we can allow for better performance optimizations in terms of speed and resource allocation.

To define a new block, first take a look at the base classes that the concrete implementation will inherit from. These are found in [here](https://github.ibm.com/DGT/fms-dgt/blob/main/fms_dgt/base/block.py). All blocks must define a `execute` function which contains their main logic.

Blocks are designed to be composable and specifiable through both config and code. A block will take as its main inputs an iterable of dictionaries, a huggingface dataset, or a pandas dataframe (see `./fms-dgt/fms_dgt/base/block.py`). Internally, it operates over a dataclass that is linked in its class definition under the `DATA_TYPE` field (e.g., see [LMBlockData](https://github.ibm.com/DGT/fms-dgt/blob/5d6226fa2fa19aeb6dedffa9c58a6b17a53c9699/fms_dgt/core/blocks/generators/llm/llm.py#L32)). When the list of inputs (e.g., dictionaries) is passed to the block, the [`__call__` function](https://github.ibm.com/DGT/fms-dgt/blob/5d6226fa2fa19aeb6dedffa9c58a6b17a53c9699/fms_dgt/base/block.py#L285) extracts the elements of the dictionary and maps them to the fields of the specified dataclass. This can be explicitly performed by specifying `input_map` or `output_map` in either the \_\_init\_\_ of the block or in the call to the block. When neither `input_map` or `output_map` are specified, the DGT assumes the required fields (e.g., the fields of the internal block dataclass) will be present in the input / output objects.

The core computation of the block (e.g., an LLM call in `./fms-dgt/fms_dgt/core/blocks/generators/llm/llm.py`) is then run on those extracted dataclass objects and the results are written back to the original dictionaries (using the `output_map`).

For example, within a databuilder one might call an llm block with:

```python
inp = {
  "llm_input": "Respond 'yes' or 'no'",
  "llm_params" : {"stop": ["yes", "no"], "temperature": 1.0},
}
inp_lst = [inp]
llm_outputs = llm_class(inp, input_map={"llm_input": "input", "llm_params": "gen_kwargs"}, output_map={"result": "llm_result"})
```

and the output may be extracted with

```python
for llm_output in llm_outputs:
  print("Original prompt: " + llm_output["input"])
  print(f"Result: {llm_output['llm_result']}")
```

Importantly, the `output_map` will specify what is _written_ onto the object that is passed in. Hence, if you want drag along additional elements in the dictionary, you just add those as fields to the input. Typically, in the process of SDG you are building up some object to return. This could be passed to through block as

```python
inp = {
  "input": "Respond 'yes' or 'no'",
  "gen_kwargs": {"stop": ["yes", "no"], "temperature": 1.0},
  "data": SdgObjectBeingGenerated
}
inp_lst = [inp]
```
