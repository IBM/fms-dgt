# Third Party
import pytest

# Local
from tests.core.blocks.llm.test_llm import (
    LM_CFG,
    auto_chat_template_test,
    execute_chat_completion_flow,
    execute_completion_flow,
)

LM_WATSONX_CFG = {
    "type": "watsonx",
    "model_id_or_path": "ibm/granite-3-8b-instruct",
    **LM_CFG,
}


@pytest.mark.parametrize("model_cfg", [LM_WATSONX_CFG])
def test_completion(model_cfg):
    execute_completion_flow(model_cfg)


@pytest.mark.parametrize("model_cfg", [LM_WATSONX_CFG])
def test_chat_completion(model_cfg):
    execute_chat_completion_flow(model_cfg)


@pytest.mark.parametrize("model_cfg", [LM_WATSONX_CFG])
def test_auto_chat_template(model_cfg):
    with pytest.raises(NotImplementedError) as exc_info:
        auto_chat_template_test(model_cfg)

    assert exc_info.value.args[0] == "WatsonX.AI does not support tokenizer initialization"
