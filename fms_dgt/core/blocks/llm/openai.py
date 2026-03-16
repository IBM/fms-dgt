# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple
import logging

# Third Party
import openai
import tiktoken

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.base.telemetry import (
    payload_max_chars,
    payload_recording_enabled,
    record_llm_payload,
)
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters, ToolChoice
from fms_dgt.core.blocks.llm.executor import AsyncLLMExecutor
from fms_dgt.core.blocks.llm.utils import Grouper, chunks, remap, retry

# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass(kw_only=True)
class OpenAICompletionParameters(Parameters):
    max_tokens: int | NotGiven = NOT_GIVEN
    n: int | NotGiven = NOT_GIVEN
    seed: int | NotGiven = NOT_GIVEN
    stop: List[str] | NotGiven = NOT_GIVEN
    top_p: int | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    skip_special_tokens: bool | NotGiven = NOT_GIVEN
    spaces_between_special_tokens: bool | NotGiven = NOT_GIVEN
    echo: bool | NotGiven = NOT_GIVEN
    logprobs: bool | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "stop": ["stop_sequences"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)


@dataclass(kw_only=True)
class OpenAIChatCompletionParameters(Parameters):
    max_completion_tokens: int | NotGiven = NOT_GIVEN
    n: int | NotGiven = NOT_GIVEN
    seed: int | NotGiven = NOT_GIVEN
    stop: List[str] | NotGiven = NOT_GIVEN
    top_p: int | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    logit_bias: Dict | NotGiven = NOT_GIVEN
    logprobs: bool | NotGiven = NOT_GIVEN
    frequency_penalty: float | NotGiven = NOT_GIVEN
    presence_penalty: float | NotGiven = NOT_GIVEN
    response_format: Dict | NotGiven = NOT_GIVEN
    top_logprobs: int | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "max_completion_tokens": ["max_tokens", "max_new_tokens"],
                "stop": ["stop_sequences"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)

    def __post_init__(self):
        if isinstance(self.logprobs, int) and self.logprobs != 0:
            self.top_logprobs = self.logprobs
            self.logprobs = True


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
async def invoke_completion(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: List[str | List[str]],
    **kwargs,
) -> openai.types.completion.Completion:
    """
    Invoke OpenAI legacy completion endpoint.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client
        model (str): OpenAI model name
        prompt (List[str | List[str]]): Batch prompt requests

    Returns:
        openai.types.completion.Completion: Completion response
    """
    return await client.completions.create(model=model, prompt=prompt, **kwargs)


async def invoke_chat_completion(
    client: openai.AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Literal["none", "auto", "required"] | ToolChoice = "auto",
    **kwargs,
) -> openai.types.completion.Completion:
    """
    Invoke OpenAI chat completion function.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client
        model (str): OpenAI model name
        messages (List[Dict[str, Any]]): list of messages in the conversation
        tools: List[Dict[str, Any]] | None: list of tool definitions available to the model
        tool_choice: controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools. `required` means the model must call one or more tools. Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool. `none` is the default when no tools are present. `auto` is the default if tools are present.

    Returns:
        openai.types.completion.Completion: Completion response
    """
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=(
            asdict(tool_choice)
            if isinstance(tool_choice, ToolChoice)
            else tool_choice if tools else "none"
        ),
        **kwargs,
    )


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("openai", "vllm-remote")
class OpenAI(LMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        call_limit: int = 10,
        base_url: str = None,
        init_tokenizer: bool = False,
        default_headers: Dict = None,
        timeout: float = 300,
        **kwargs: Any,
    ):
        # Step 1: Initialize parent
        super().__init__(init_tokenizer=init_tokenizer, **kwargs)

        # Step 2: Set batch size, if None
        if not self.batch_size:
            self._batch_size = 10

        # Step 3: Set call limit
        self._call_limit = call_limit

        # Step 4: Initialize OpenAI clients
        resolved_key = (
            api_key
            if api_key
            else get_resource("api", key_name="OPENAI_API_KEY", call_limit=call_limit).key
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=resolved_key,
            timeout=timeout,
            base_url=base_url,
            default_headers=default_headers,
        )

        # Step 5: Register with the credential-based semaphore pool so that all
        # blocks sharing the same API key share a single concurrency limit.
        self._init_semaphore(credential=resolved_key, max_concurrent_requests=call_limit)

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(
        self, **kwargs
    ) -> Tuple[OpenAICompletionParameters, OpenAIChatCompletionParameters]:
        return OpenAICompletionParameters.from_dict(
            kwargs
        ), OpenAIChatCompletionParameters.from_dict(kwargs)

    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        return tiktoken.encoding_for_model(model_id_or_path or self.model_id_or_path)

    def _extract_choice_content(self, choice: Any, method: str) -> str | Dict:
        # If choice is generated via chat completion for vLLM Remote
        if method == self.CHAT_COMPLETION:
            return (
                choice.message.to_dict()
                if isinstance(
                    choice.message,
                    openai.types.chat.chat_completion_message.ChatCompletionMessage,
                )
                else choice.message
            )

        # If choice is generated via text completion for vLLM Remote
        return choice.text

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
        chunk,
        update_progress: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """Process one work item (a batch for completion, a single instance for chat).

        Args:
            chunk: A list of ``LMBlockData`` for completion mode, or a single
                ``LMBlockData`` for chat-completion mode.
            update_progress: Callable ``(int) -> None`` to advance the progress bar.
            method: ``LMProvider.COMPLETION`` or ``LMProvider.CHAT_COMPLETION``.

        Raises:
            ValueError: If an unsupported method is passed.
            RuntimeError: If the number of API responses does not match expected.
        """
        # Fetch generation kwargs from 1st request (identical within a chunk)
        params = next(iter(chunk)).gen_kwargs if isinstance(chunk, Iterable) else chunk.gen_kwargs

        # Extract completion parameters from gen_kwargs
        params = (
            self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
        ).to_params(params)

        # Simplify downstream processing
        if params.get("logprobs") and params.get("top_logprobs") is None:
            params["top_logprobs"] = 1

        invoke_with_retry = retry(
            on_exceptions=(openai.OpenAIError,),
            max_retries=3,
            on_exception_callback=lambda e, t: self.logger.warning(
                "Retrying in %d seconds due to %s: %s", t, type(e).__name__, e.args[0]
            ),
        )

        batch_size = len(chunk) if isinstance(chunk, Iterable) else 1

        async with self._llm_span(
            method=method, batch_size=batch_size, params=params
        ) as span_attrs:
            if method == self.CHAT_COMPLETION:
                messages = self._prepare_input(
                    chunk,
                    method=self.CHAT_COMPLETION,
                    max_tokens=params.get("max_completion_tokens", None),
                )
                response = await invoke_with_retry(invoke_chat_completion)(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    messages=messages,
                    tools=chunk.tools,
                    tool_choice=chunk.tool_choice,
                    **params,
                )
            elif method == self.COMPLETION:
                prompts = [
                    self._prepare_input(
                        instance,
                        method=self.COMPLETION,
                        max_tokens=params.get("max_tokens", None),
                    )
                    for instance in chunk
                ]
                response = await invoke_with_retry(invoke_completion)(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    prompt=prompts,
                    **params,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            # Record token usage in span
            span_attrs["prompt_tokens"] = response.usage.prompt_tokens
            span_attrs["completion_tokens"] = response.usage.completion_tokens

            if payload_recording_enabled():
                if method == self.CHAT_COMPLETION:
                    rc = self._extract_choice_content(response.choices[0], method=method)
                else:
                    rc = " ".join(c.text for c in response.choices if hasattr(c, "text"))
                record_llm_payload(
                    span_attrs,
                    method,
                    payload_max_chars(),
                    prompt="\n---\n".join(prompts) if method == self.COMPLETION else None,
                    messages=messages if method == self.CHAT_COMPLETION else None,
                    response_completion=rc,
                )

        # Validate response count
        n = params.get("n", 1)
        if len(response.choices) != n * batch_size:
            raise RuntimeError(
                f"Number of responses does not match number of inputs * n, [{len(response.choices)}, {batch_size}, {n}]"
            )

        # Group N responses per input and write results back
        response_choices_per_input = [
            response.choices[i : i + n] for i in range(0, len(response.choices), n)
        ]
        total_outputs = sum(len(x) for x in response_choices_per_input)

        for response_choices, instance in zip(
            response_choices_per_input,
            chunk if isinstance(chunk, Iterable) else [chunk],
        ):
            outputs = []
            addtl = {
                "completion_tokens": (response.usage.completion_tokens // total_outputs),
                "prompt_tokens": response.usage.prompt_tokens // total_outputs,
                "token_logprobs": [],
            }
            for choice in response_choices:
                outputs.append(self._extract_choice_content(choice, method=method))

                token_logprobs = self._extract_token_log_probabilities(choice=choice, method=method)
                if token_logprobs:
                    addtl["token_logprobs"].append(token_logprobs)
                    addtl["completion_tokens"] = len(token_logprobs)

            self.update_instance_with_result(
                method,
                outputs if len(outputs) > 1 else outputs[0],
                instance,
                params.get("stop", None),
                addtl,
            )

        update_progress(self._batch_size if method == self.COMPLETION else 1)

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        # Build work items: batches for completion, individual instances for chat
        work_items = []
        grouper = Grouper(requests, lambda x: str(x.gen_kwargs))
        for _, reqs in grouper.get_grouped().items():
            for chunk in chunks(reqs, n=self.batch_size):
                if method == self.CHAT_COMPLETION:
                    work_items.extend(chunk)
                else:
                    work_items.append(chunk)

        await AsyncLLMExecutor.run(
            work_items=work_items,
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
        self.run_async(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.COMPLETION,
                **kwargs,
            )
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
