# Standard
from typing import Any, Dict, List
import copy
import os
import re
import time

# Third Party
import pytest

# Local
from fms_dgt.base.registry import get_block
from fms_dgt.core.blocks.llm import CachingLM, LMProvider

# Skip vLLM tests, if necessary
SKIP_VLLM = False
try:
    # Third Party
    import vllm  # noqa: F401
except ModuleNotFoundError:
    SKIP_VLLM = True

# ===========================================================================
#                       CONFIGURATIONS
# ===========================================================================
LM_CFG = {
    "temperature": 0.0,
    "max_tokens": 5,
    "n": 3,
}
LM_VLLM_CFG = {
    "type": "vllm",
    "model_id_or_path": "ibm/granite-3-3-8b-instruct",
    "tensor_parallel_size": 1,
    **LM_CFG,
    "n": 1,
}
LM_VLLM_SERVER_CFG = {
    "type": "vllm-server",
    "model_id_or_path": "ibm/granite-3-3-8b-instruct",
    "tensor_parallel_size": 1,
    **LM_CFG,
}
LM_OPENAI_CFG = {
    "type": "openai",
    "model_id_or_path": "gpt-3.5-turbo",
    **LM_CFG,
}


# ===========================================================================
#                       CONSTANTS
# ===========================================================================
PROMPTS = [f"Question: x = {i} + 1\nAnswer: x =" for i in range(25)]


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
def execute_completion_flow(model_cfg, prompts: List[str] = PROMPTS):
    # Step 1: Initialize LM block
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # Step 2: Configure inputs
    inputs: List[Dict] = [{"input": prompt} for prompt in prompts]

    # Step 3: Invoke "__call__" with "generate" method
    outputs = lm(inputs)

    # Step 4: Verify
    for idx, output in enumerate(outputs):
        assert (
            output["input"] == inputs[idx]["input"]
        ), f"Input list has been rearranged at index {idx}"

        assert isinstance(output["result"], str) or (
            isinstance(output["result"], list) and len(output["result"]) == model_cfg["n"]
        )


def execute_chat_completion_flow(
    model_cfg,
    conversations: List[Dict[str, Any]] = [
        [
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in PROMPTS
    ],
):
    # Step 1: Initialize LM block
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # Step 2: Configure inputs
    inputs: List[Dict] = [{"input": conversation} for conversation in conversations]

    # Step 3: Invoke "__call__" with "chat" method
    outputs = lm(inputs, method=LMProvider.CHAT_COMPLETION)

    # Step 4: Verify
    for idx, output in enumerate(outputs):
        assert (
            output["input"] == inputs[idx]["input"]
        ), f"Input list has been rearranged at index {idx}"

        assert (
            (isinstance(output["result"], dict) and isinstance(output["result"]["content"], str))
            or isinstance(output["result"], str)
            or (isinstance(output["result"], list) and len(output["result"]) == model_cfg["n"])
        )


def auto_chat_template_test(model_cfg):
    model_type = model_cfg.get("type")
    model_cfg["init_tokenizer"] = True
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # check it passes through for a simple string
    prompt = [
        {"role": "system", "content": "You are the system"},
        {"role": "user", "content": "Hello world"},
    ]

    prompt = lm.tokenizer.apply_chat_template(prompt, tokenize=False)

    assert isinstance(prompt, str)


def lm_caching_test(model_cfg):
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_cache.db")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    non_cache_inputs: List[Dict] = []
    for prompt in PROMPTS:
        non_cache_inputs.append({"input": prompt})

    pre_cache_inputs = copy.deepcopy(non_cache_inputs)
    post_cache_inputs = copy.deepcopy(non_cache_inputs)

    non_cache_time = time.time()
    lm(non_cache_inputs)
    non_cache_time = time.time() - non_cache_time

    cache_lm = CachingLM(
        type(lm),
        force_cache=False,
        lm_cache=cache_path,
        name=f"test_{model_type}",
        **model_cfg,
    )

    pre_cache_time = time.time()
    cache_lm(pre_cache_inputs)
    pre_cache_time = time.time() - pre_cache_time

    post_cache_time = time.time()
    cache_lm(post_cache_inputs)
    post_cache_time = time.time() - post_cache_time

    os.remove(cache_path)

    assert (
        post_cache_time < pre_cache_time and post_cache_time < non_cache_time
    ), f"Caching led to increased execution time {(post_cache_time, pre_cache_time, non_cache_time)}"

    for i, (non, pre, post) in enumerate(
        zip(non_cache_inputs, pre_cache_inputs, post_cache_inputs)
    ):
        assert (
            non["input"] == pre["input"] == post["input"]
        ), f"Input list has been rearranged at index {i}: {(non['input'], pre['input'], post['input'])}"
        assert (
            non["result"] == pre["result"] == post["result"]
        ), f"Different results detected at index {i}: {(non['result'], pre['result'], post['result'])}"


# ===========================================================================
#                       TESTS (vLLM)
# ===========================================================================


@pytest.mark.skipif(SKIP_VLLM, reason='requires "vllm" library')
@pytest.mark.parametrize("model_cfg", [LM_VLLM_CFG])
def test_generate(model_cfg):
    execute_completion_flow(model_cfg)


@pytest.mark.skipif(SKIP_VLLM, reason='requires "vllm" library')
@pytest.mark.parametrize("model_cfg", [LM_VLLM_CFG])
def test_chat(model_cfg):
    execute_chat_completion_flow(model_cfg)


@pytest.mark.skipif(SKIP_VLLM, reason='requires "vllm" library')
@pytest.mark.parametrize("model_cfg", [LM_VLLM_SERVER_CFG])
def test_lm_caching(model_cfg):
    lm_caching_test(model_cfg)


@pytest.mark.skipif(SKIP_VLLM, reason='requires "vllm" library')
@pytest.mark.parametrize("model_cfg", [LM_VLLM_SERVER_CFG])
def test_auto_chat_template(model_cfg):
    auto_chat_template_test(model_cfg)


@pytest.mark.skipif(SKIP_VLLM, reason='requires "vllm" library')
def test_vllm_tensor_parallel():
    """

    replace "model_id_or_path" with suitably large model and ensure you have 2 GPUs of sufficient size, e.g. 2 of the a100_80gb

    """
    model_cfg = dict(LM_VLLM_CFG)
    model_cfg["type"] = "vllm"
    model_cfg["tensor_parallel_size"] = 2
    model_cfg["model_id_or_path"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"input": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["input"] == inputs_copy[i]["input"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str)


# ===========================================================================
#                       TESTS (vLLM Remote)
# ===========================================================================
# Test openai API with batch prompts
def test_vllm_remote_batch():
    """
    start server with

    python -m vllm.entrypoints.openai.api_server --model ibm/granite-3-3-8b-instruct

    """
    model_cfg = dict(LM_VLLM_CFG)
    model_cfg["type"] = "vllm-remote"
    model_cfg["base_url"] = "http://0.0.0.0:8000/v1"
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"input": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["input"] == inputs_copy[i]["input"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str)


def test_vllm_remote_chat_template():
    """
    start server with
    python -m vllm.entrypoints.openai.api_server --model ibm/granite-3-3-8b-instruct
    """
    model_cfg = dict(LM_VLLM_CFG)
    model_cfg["type"] = "vllm-remote"
    model_cfg["base_url"] = "http://0.0.0.0:8000/v1"
    model_cfg["init_tokenizer"] = True
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        message = [{"role": "user", "content": prompt}]
        inp = lm.tokenizer.apply_chat_template(message, tokenize=False)
        inputs.append({"input": inp})

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    correct = 0
    for i, inp in enumerate(inputs):
        assert (
            inp["input"] == inputs_copy[i]["input"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str)
        try:
            f = float(inp["result"])
            if f == i + 1:
                correct += 1
        except Exception:
            pass  # nonfloat string is not correct
    assert correct > 5, f"Poor generation only {correct} of 25 correct"


# Test openai API with batch prompts applying a chat template under the covers
def test_vllm_remote_auto_chat():
    """
    start server with
    python -m vllm.entrypoints.openai.api_server --model ibm/granite-3-3-8b-instruct
    """
    model_cfg = dict(LM_VLLM_CFG)
    model_cfg["type"] = "vllm-remote"
    model_cfg["base_url"] = "http://0.0.0.0:8000/v1"
    model_type = model_cfg.get("type")
    lm: LMProvider = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        message = [{"role": "user", "content": prompt}]
        inputs.append({"input": message})

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs, method=LMProvider.CHAT_COMPLETION)

    correct = 0
    for i, inp in enumerate(inputs):
        assert (
            inp["input"] == inputs_copy[i]["input"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"]["content"], str)
        # This is a loose check, just making sure that the answer has
        # digits in it.
        if re.search(r"\d", inp["result"]["content"]):
            correct += 1
    assert correct > 10, f"Poor generation only {correct} of 25 correct"
