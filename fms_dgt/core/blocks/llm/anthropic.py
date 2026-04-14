# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Literal, Tuple
import logging

# Third Party
import anthropic

# Local
from fms_dgt.base.block import get_row_name
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.base.telemetry import (
    payload_max_chars,
    payload_recording_enabled,
    record_llm_payload,
)
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters, ToolChoice
from fms_dgt.core.blocks.llm.executor import AsyncLLMExecutor
from fms_dgt.core.blocks.llm.utils import remap, retry
from fms_dgt.core.resources.api import ApiKeyResource

# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass(kw_only=True)
class AnthropicCompletionParameters(Parameters):
    max_tokens_to_sample: int | NotGiven = NOT_GIVEN
    stop_sequences: List[str] | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    top_k: int | NotGiven = NOT_GIVEN
    top_p: float | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "max_tokens_to_sample": [
                    "max_tokens",
                    "max_completion_tokens",
                ],
                "stop_sequences": ["stop"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)


@dataclass(kw_only=True)
class AnthropicChatCompletionParameters(Parameters):
    max_tokens: int
    stop_sequences: List[str] | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    top_k: int | NotGiven = NOT_GIVEN
    top_p: float | NotGiven = NOT_GIVEN
    response_format: Dict | NotGiven = NOT_GIVEN
    output_config: Dict | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "max_tokens": [
                    "max_tokens_to_sample",
                    "max_completion_tokens",
                ],
                "stop_sequences": ["stop"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
async def invoke_completion(
    client: anthropic.AsyncAnthropic,
    model: str,
    prompt: str,
    **kwargs,
) -> anthropic.types.completion.Completion:
    """
    Invoke Anthropic legacy completion endpoint.

    Args:
        client (anthropic.AsyncAnthropic): Async Anthropic client
        model (str): Anthropic model name
        prompt (str): prompt requests

    Returns:
        anthropic.types.completion.Completion: Completion response
    """
    return await client.completions.create(
        model=model, prompt=f"\n\nHuman: {prompt}\n\nAssistant: ", **kwargs
    )


async def invoke_chat_completion(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Literal["none", "auto", "required"] | ToolChoice = "auto",
    **kwargs,
) -> anthropic.types.Message:
    """
    Invoke Anthropic chat completion function.

    Args:
        client (anthropic.AsyncAnthropic): Async Anthropic client
        model (str): Anthropic model name
        messages (List[Dict[str, Any]]): list of messages in the conversation
        tools: List[Dict[str, Any]] | None: list of tool definitions available to the model
        tool_choice: controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools.`required` means the model must call one or more tools.Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool. `none` is the default when no tools are present. `auto` is the default if tools are present.


    Returns:
        anthropic.types.Message: Message response
    """
    # Extract system messages from messages list as per Anthropic requirements
    system_messags = []
    remaining_messages = []
    for message in messages:
        if message["role"] in "system" or message["role"] == "developer":
            system_messags.append(message)
        else:
            remaining_messages.append(message)

    # Adjust tool_choice as per Anthropic requirements
    if tools:
        if isinstance(tool_choice, ToolChoice):
            tool_choice = {"type": "tool", "name": tool_choice.function.name}
        elif tool_choice == "required":
            tool_choice = {"type": "any"}
        elif tool_choice == "none":
            tool_choice = {"type": "none"}
        else:
            tool_choice = {"type": "auto"}

    # Map response_format (OpenAI convention) to output_config as per Anthropic requirements.
    # output_config takes precedence if both are set (caller used native Anthropic API).
    # Anthropic only supports json_schema; json_object and text have no equivalent and are dropped.
    if "response_format" in kwargs and "output_config" not in kwargs:
        rf = kwargs.pop("response_format")
        if rf.get("type") == "json_schema":
            # OpenAI wraps the schema under json_schema.schema — Anthropic expects it directly.
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": rf.get("json_schema", {}).get("schema", {}),
                }
            }
        # type == "json_object" or "text": no Anthropic equivalent, omit output_config entirely.
    else:
        kwargs.pop("response_format", None)

    # Invoke completion
    return await client.messages.create(
        model=model,
        messages=remaining_messages,
        system=(
            " ".join([message["content"] for message in system_messags])
            if system_messags
            else anthropic.NOT_GIVEN
        ),
        tools=tools if tools else anthropic.NOT_GIVEN,
        tool_choice=tool_choice if tools else anthropic.NOT_GIVEN,
        **kwargs,
    )


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("anthropic")
class Anthropic(LMProvider):
    def __init__(
        self,
        call_limit: int = 10,
        init_tokenizer: bool = False,
        timeout: float = 300,
        **kwargs: Any,
    ):

        # Intialize parent
        super().__init__(init_tokenizer=init_tokenizer, **kwargs)

        # Set batch size, if None
        if not self.batch_size or self.batch_size > 1:
            self._batch_size = 1

        # Set call limit
        self._call_limit = call_limit

        # LM provider connection arguments
        api_resource: ApiKeyResource = get_resource(
            "api", key_name="ANTHROPIC_API_KEY", call_limit=call_limit
        )

        # Initialize Anthropic async client
        self.async_client = anthropic.AsyncAnthropic(api_key=api_resource.key, timeout=timeout)

        # Register with the credential-based semaphore pool
        self._init_semaphore(credential=api_resource.key, max_concurrent_requests=call_limit)

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_length(self):
        # If max length manually set, return it
        if self._parameters.max_length:
            return self._parameters.max_length

        # Default max length is set to 200k as per https://docs.anthropic.com/en/docs/about-claude/models/overview
        return int(2e6)

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(
        self, **kwargs
    ) -> Tuple[AnthropicCompletionParameters, AnthropicChatCompletionParameters]:
        return AnthropicCompletionParameters.from_dict(
            kwargs
        ), AnthropicChatCompletionParameters.from_dict(kwargs)

    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        raise NotImplementedError(
            'Tokenization support is disabled for "Antropic" provider. Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
        )

    def _extract_choice_content(self, choice: Any, method: str) -> str | Dict:
        # If choice is generated via chat completion
        if method == self.CHAT_COMPLETION:
            response = {"role": "assistant"}
            tool_calls = []
            text = []
            for entry in choice.content:
                if isinstance(entry, anthropic.types.TextBlock):
                    text.append(entry.text)
                elif isinstance(entry, anthropic.types.ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": entry.id,
                            "type": "function",
                            "function": {"arguments": entry.input, "name": entry.name},
                        }
                    )

            if text:
                response["content"] = " ".join(text)

            if tool_calls:
                response["tool_calls"] = tool_calls

            return response

        # If choice is generated via text completion
        return choice.completion

    def _extract_token_log_probabilities(self, choice, method: str) -> List[Any] | None:
        # If choice is generated via text completion for vLLM Remote
        if method == self.COMPLETION:
            return (
                [x for x in choice.logprobs.top_logprobs if x is not None]
                if choice.logprobs
                else None
            )
        # If choice is generated via chat completion for vLLM Remote
        elif method == self.CHAT_COMPLETION:
            top_logprobs = None
            if choice.logprobs and choice.logprobs.content:
                top_logprobs = []
                for entry in choice.logprobs.content:
                    if entry.top_logprobs:
                        top_logprobs.append(
                            {top_token.token: top_token.logprob for top_token in entry.top_logprobs}
                        )
                    else:
                        top_logprobs.append({})

            return top_logprobs

    async def _process_item(
        self,
        instance: LMBlockData,
        update_progress: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """Process one LMBlockData instance via the Anthropic API.

        Args:
            instance: The request to process.
            update_progress: Zero-argument callable to advance the progress bar.
            method: ``LMProvider.COMPLETION`` or ``LMProvider.CHAT_COMPLETION``.
        """
        params = (
            self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
        ).to_params(instance.gen_kwargs)

        invoke_with_retry = retry(
            on_exceptions=(anthropic.AnthropicError,),
            max_retries=3,
            on_exception_callback=lambda e, t: self.logger.warning(
                "Retrying in %d seconds due to %s: %s", t, type(e).__name__, e.args[0]
            ),
        )

        _task_name = get_row_name(instance)
        async with self._llm_span(
            method=method,
            batch_size=1,
            params=params,
            task_names=[_task_name] if _task_name is not None else None,
        ) as span_attrs:
            if method == self.CHAT_COMPLETION:
                messages = self._prepare_input(
                    instance,
                    method=method,
                    max_tokens=params.get("max_tokens", None),
                )
                response = await invoke_with_retry(invoke_chat_completion)(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    messages=messages,
                    tools=instance.tools,
                    tool_choice=instance.tool_choice,
                    **params,
                )
            elif method == self.COMPLETION:
                prompt = self._prepare_input(
                    instance,
                    method=self.COMPLETION,
                    max_tokens=params.get("max_tokens_to_sample", None),
                )
                response = await invoke_with_retry(invoke_completion)(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    prompt=prompt,
                    **params,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            span_attrs["prompt_tokens"] = response.usage.input_tokens
            span_attrs["completion_tokens"] = response.usage.output_tokens

            if payload_recording_enabled():
                if method == self.CHAT_COMPLETION:
                    rc = self._extract_choice_content(response, method=method)
                else:
                    rc = response.completion
                record_llm_payload(
                    span_attrs,
                    method,
                    payload_max_chars(),
                    prompt=prompt if method == self.COMPLETION else None,
                    messages=messages if method == self.CHAT_COMPLETION else None,
                    response_completion=rc,
                )

        self.update_instance_with_result(
            method,
            self._extract_choice_content(response, method=method),
            instance,
            params.get("stop", None),
            {
                "completion_tokens": response.usage.output_tokens,
                "prompt_tokens": response.usage.input_tokens,
                "token_logprobs": [],
            },
        )

        update_progress(1)

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        await AsyncLLMExecutor.run(
            work_items=requests,
            process_item=lambda item, upd: self._process_item(item, upd, method=method),
            call_limit=self._call_limit,
            total_requests=len(requests),
            method=method,
            disable_tqdm=disable_tqdm,
        )

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def completion(self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs) -> None:
        raise RuntimeError(
            'Support for "completion" method for newer models has been deprecated as per "Anthropic" documentation: https://docs.anthropic.com/en/api/complete'
        )

    def chat_completion(
        self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs
    ) -> None:
        self.run_async(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.CHAT_COMPLETION,
                **kwargs,
            )
        )
