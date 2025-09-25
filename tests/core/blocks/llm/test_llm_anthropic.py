# Standard
import os

# Third Party
import pytest

# Local
from tests.core.blocks.llm.test_llm import (
    LM_CFG,
    auto_chat_template_test,
    execute_chat_completion_flow,
    execute_completion_flow,
)

LM_ANTHROPIC_CFG = {
    **LM_CFG,
    "type": "anthropic",
    "model_id_or_path": "claude-3-haiku-20240307",
    "max_tokens": 25,
}


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_completion(model_cfg):
    with pytest.raises(RuntimeError) as exc_info:
        execute_completion_flow(model_cfg, prompts=["Question: x = 0 + 1\nAnswer: x ="])

    assert (
        exc_info.value.args[0]
        == 'Support for "completion" method for newer models has been deprecated as per "Anthropic" documentation: https://docs.anthropic.com/en/api/complete'
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_chat_completion(model_cfg):
    execute_chat_completion_flow(
        model_cfg,
        conversations=[
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello! How were you trained?"},
            ]
        ],
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_auto_chat_template(model_cfg):
    with pytest.raises(NotImplementedError) as exc_info:
        auto_chat_template_test(model_cfg)

    assert (
        exc_info.value.args[0]
        == 'Tokenization support is disabled for "Antropic" provider. Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
    )
