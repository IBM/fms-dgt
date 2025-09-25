# Standard
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Literal, Tuple
import asyncio
import logging

# Third Party
from tqdm import tqdm
import anthropic

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters, ToolChoice
from fms_dgt.core.blocks.llm.utils import remap, retry
from fms_dgt.core.resources.api import ApiKeyResource
from fms_dgt.utils import dgt_logger

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
@retry(
    on_exceptions=(anthropic.AnthropicError,),
    max_retries=3,
    on_exception_callback=lambda e, sleep_timer: dgt_logger.warning(
        "Retrying in %d seconds due to %s: %s", sleep_timer, type(e).__name__, e.args[0]
    ),
)
async def invoke_completion(
    client: anthropic.AsyncAnthropic,
    model: str,
    prompt: str,
    **kwargs,
) -> anthropic.types.completion.Completion:
    """
    Invoke Anthropic legacy completion endopint.

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


@retry(
    on_exceptions=(anthropic.AnthropicError,),
    max_retries=3,
    on_exception_callback=lambda e, sleep_timer: dgt_logger.warning(
        "Retrying in %d seconds due to %s: %s", sleep_timer, type(e).__name__, e.args[0]
    ),
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

        # Initialize OpenAI clients
        self.async_client = anthropic.AsyncAnthropic(api_key=api_resource.key, timeout=timeout)

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

    async def async_executor(
        self,
        queue: asyncio.Queue,
        update_progress_tracker: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """
        Execute text completion or chat completion asynchronously.

        Args:
            queue (asyncio.Queue): instances to complete.
            update_progress_tracker (Callable): progress tracker update function
            method (str, optional): Either "completion" or "chat_completion". Defaults to LMProvider.COMPLETION.

        Raises:
            ValueError: If method outside "completion" or "chat_completion" is passed
            RuntimeError: If number of responses does not match number of inputs * n
        """
        while not queue.empty():
            # Get a "work item" out of the queue.
            instance = await queue.get()

            # Extract parameters
            params = (
                self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
            ).to_params(instance.gen_kwargs)

            # Trigger appropriate functions
            if method == self.CHAT_COMPLETION:
                response = await invoke_chat_completion(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    messages=self._prepare_input(
                        instance,
                        method=method,
                        max_tokens=params.get("max_tokens", None),
                    ),
                    tools=instance.tools,
                    tool_choice=instance.tool_choice,
                    **params,
                )
            elif method == self.COMPLETION:
                response = await invoke_completion(
                    client=self.async_client,
                    model=self.model_id_or_path,
                    prompt=self._prepare_input(
                        instance,
                        method=self.COMPLETION,
                        max_tokens=params.get("max_tokens_to_sample", None),
                    ),
                    **params,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            # Add output
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

            # Notify the queue that the "work item" has been processed.
            queue.task_done()

            # Update progress tracker
            update_progress_tracker()

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        # Initialize necessary variables
        queue = asyncio.Queue()

        # Add to queue
        for instance in requests:
            queue.put_nowait(instance)

        # Initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc=f"Running {method} requests",
        )

        # Create generate tasks
        executors = []
        for _ in range(min(queue.qsize(), self._call_limit)):
            executors.append(
                self.async_executor(
                    queue,
                    update_progress_tracker=lambda: pbar.update(1),
                    method=method,
                )
            )

        # Wait until all worker are finished
        await asyncio.gather(*executors, return_exceptions=True)

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
        asyncio.run(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.CHAT_COMPLETION,
                **kwargs,
            )
        )
