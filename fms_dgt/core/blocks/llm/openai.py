# Standard
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple
import asyncio
import logging

# Third Party
from tqdm import tqdm
import openai
import tiktoken

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters, ToolChoice
from fms_dgt.core.blocks.llm.utils import Grouper, chunks, remap, retry
from fms_dgt.utils import dgt_logger

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
@retry(
    on_exceptions=(openai.OpenAIError,),
    max_retries=3,
    on_exception_callback=lambda e, sleep_timer: dgt_logger.warning(
        "Retrying in %d seconds due to %s: %s", sleep_timer, type(e).__name__, e.args[0]
    ),
)
async def invoke_completion(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: List[str | List[str]],
    **kwargs,
) -> openai.types.completion.Completion:
    """
    Invoke OpenAI legacy completion endopint.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client
        model (str): OpenAI model name
        prompt (List[str | List[str]]): Batch prompt requests

    Returns:
        openai.types.completion.Completion: Completion response
    """
    return await client.completions.create(model=model, prompt=prompt, **kwargs)


@retry(
    on_exceptions=(openai.OpenAIError,),
    max_retries=3,
    on_exception_callback=lambda e, sleep_timer: dgt_logger.warning(
        "Retrying in %d seconds due to %s: %s", sleep_timer, type(e).__name__, e.args[0]
    ),
)
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
        tool_choice: controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools.`required` means the model must call one or more tools. Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool. `none` is the default when no tools are present. `auto` is the default if tools are present.

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
        self.async_client = openai.AsyncOpenAI(
            api_key=(
                api_key
                if api_key
                else get_resource("api", key_name="OPENAI_API_KEY", call_limit=call_limit).key
            ),
            timeout=timeout,
            base_url=base_url,
            default_headers=default_headers,
        )

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

    async def async_executor(
        self,
        queue: asyncio.Queue,
        update_progress_tracker: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """
        Execute text completion or chat completion asynchronously.

        NOTE:
        - For "chat" completion, queue contains individual conversation to complete
        - For "text" completion, queue contains batches of prompts that can be passed in a single call

        Args:
            queue (asyncio.Queue): instances to complete.
            update_progress_tracker (Callable): progress tracker update function
            method (str, optional): Either "completion" or "chat_completion". Defaults to LMProvider.COMPLETION.

        Raises:
            ValueError: If method outside "completion" or "chat_completion" is passed
            RuntimeError: If number of responses does not match number of inputs * n
        """
        while not queue.empty():
            # Step 1: Get a "work item" out of the queue.
            chunk = await queue.get()

            # Step 2: Fetch generation kwargs from 1st request since generation kwargs within a chunk are identical
            params = (
                next(iter(chunk)).gen_kwargs if isinstance(chunk, Iterable) else chunk.gen_kwargs
            )

            # Step 3: Extract completion parameters from gen_kwargs
            params = (
                self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
            ).to_params(params)

            # adding this here to simplify downstream processing
            if params.get("logprobs") and params.get("top_logprobs") is None:
                params["top_logprobs"] = 1

            # Step 4: Trigger OpenAI completion functions
            if method == self.CHAT_COMPLETION:
                response = await invoke_chat_completion(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    messages=self._prepare_input(
                        chunk,
                        method=self.CHAT_COMPLETION,
                        max_tokens=params.get("max_completion_tokens", None),
                    ),
                    tools=chunk.tools,
                    tool_choice=chunk.tool_choice,
                    **params,
                )
            elif method == self.COMPLETION:
                response = await invoke_completion(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    prompt=[
                        self._prepare_input(
                            instance,
                            method=self.COMPLETION,
                            max_tokens=params.get("max_tokens", None),
                        )
                        for instance in chunk
                    ],
                    **params,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            # Step 5: If multiple choices requested per input
            # Step 5.a: Get requested choices count from parameters
            n = params.get("n", 1)

            # Step 5.b: Verify enough responses are generated per input
            if len(response.choices) != n * (len(chunk) if isinstance(chunk, Iterable) else 1):
                raise RuntimeError(
                    f"Number of responses does not match number of inputs * n, [{len(response.choices)}, {len(chunk) if isinstance(chunk, Iterable) else 1}, {n}]"
                )

            # Step 5.c: Group N responses for each request
            response_choices_per_input = [
                response.choices[i : i + n] for i in range(0, len(response.choices), n)
            ]
            total_outputs = sum([len(x) for x in response_choices_per_input])

            # Step 6: Iterate over each grouped response
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

                    token_logprobs = self._extract_token_log_probabilities(
                        choice=choice, method=method
                    )
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

            # Step 7: Notify the queue that the "work item" has been processed.
            queue.task_done()

            # Step 8: Update progress tracker
            update_progress_tracker()

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        # Step 1: Initialize necessary variables
        queue = asyncio.Queue()

        # Step 2: Group requests by their generation_kwargs
        grouper = Grouper(requests, lambda x: str(x.gen_kwargs))

        # Step 3: Iterate over each group
        for _, reqs in grouper.get_grouped().items():
            # Step 3.a: Create request chunks based on maximum allowed batch size and add to queue
            for chunk in chunks(reqs, n=self.batch_size):
                if method == self.CHAT_COMPLETION:
                    for instance in chunk:
                        queue.put_nowait(instance)
                else:
                    queue.put_nowait(chunk)

        # Step 4: Initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc=f"Running {method} requests",
        )

        # Step 5: Create generate tasks
        executors = []
        for _ in range(min(queue.qsize(), self._call_limit)):
            executors.append(
                self.async_executor(
                    queue,
                    update_progress_tracker=lambda: pbar.update(
                        self._batch_size if method == self.COMPLETION else 1
                    ),
                    method=method,
                )
            )

        # Step 6: Wait until all worker are finished
        await asyncio.gather(*executors, return_exceptions=True)

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def completion(self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs) -> None:
        asyncio.run(
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
        asyncio.run(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.CHAT_COMPLETION,
                **kwargs,
            )
        )
