# Third Party
import pytest

# Local
from tests.core.blocks.llm.test_llm import (
    LM_CFG,
    auto_chat_template_test,
    execute_chat_completion_flow,
    execute_completion_flow,
)

LM_OLLAMA_CFG = {
    "type": "ollama",
    "base_url": "http://localhost:11434",
    "model_id_or_path": "llama3.2",
    "stop": ["\n"],
    **LM_CFG,
}


@pytest.mark.skipif(True, "requires hardware support to run models")
@pytest.mark.parametrize("model_cfg", [LM_OLLAMA_CFG])
def test_completion(model_cfg):
    execute_completion_flow(model_cfg)


@pytest.mark.skipif(True, "requires hardware support to run models")
@pytest.mark.parametrize("model_cfg", [LM_OLLAMA_CFG])
def test_chat_completion(model_cfg):
    execute_chat_completion_flow(model_cfg)


@pytest.mark.parametrize("model_cfg", [LM_OLLAMA_CFG])
def test_auto_chat_template(model_cfg):
    with pytest.raises(NotImplementedError) as exc_info:
        auto_chat_template_test(model_cfg)

    assert (
        exc_info.value.args[0]
        == 'Tokenization support is disabled for "Ollama". Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
    )
