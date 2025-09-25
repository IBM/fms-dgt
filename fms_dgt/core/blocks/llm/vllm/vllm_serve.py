"""
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard
from importlib.util import find_spec
from typing import Any, Tuple
import os
import subprocess
import time
import uuid

# Third Party
from dotenv import load_dotenv
import psutil
import requests

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.blocks.llm.openai import (
    OpenAI,
    OpenAIChatCompletionParameters,
    OpenAICompletionParameters,
)
from fms_dgt.utils import dgt_logger, get_open_port

try:
    # Third Party
    import vllm  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "attempted to use 'vllm' LM type, but package `vllm` not installed. ",
        "please install these via `pip install -r fms_dgt[vllm]`",
    ) from err


# TODO: this can be made more efficient for our purposes by rewriting the async code ourselves
@register_block("vllm-server")
class vLLMServer(LMProvider):
    """vLLM Generator"""

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        swap_space: int = 0,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        check_interval: int = 10,
        lora_modules: str = None,
        host: str = "0.0.0.0",
        port: int = None,
        pid: int = None,
        api_key: str = None,
        **kwargs: Any,
    ):
        # Step 1: Verify necessary resources are available
        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install fms_dgt[vllm]`"
            )

        if device and "cuda" not in device:
            raise ValueError("vLLM only supports CUDA")

        # Step 2: Intialize parent
        super().__init__(**kwargs)

        # Step 3: Load enviroment
        # FIXME: Should we remove this?
        load_dotenv()

        # Step 4: Configure necessary variables
        if not self.batch_size:
            self._batch_size = "auto"

        self._check_interval = check_interval
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._data_parallel_size = int(data_parallel_size)
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._swap_space = int(swap_space)
        self._lora_modules = lora_modules

        self._pid = pid if pid is not None else os.getpid()
        self._api_key = api_key if api_key is not None else str(uuid.uuid4())

        self._host = host
        self._port = get_open_port(host) if port is None else port
        self._base_url = f"http://{self._host}:{self._port}/v1/"
        self._vllm = OpenAI(api_key=self._api_key, base_url=self._base_url, **kwargs)

        self._vllm_process = None

        # Step 5: serve vLLM model
        self.serve()

    @property
    def base_url(self):
        return self._base_url

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(
        self, **kwargs
    ) -> Tuple[OpenAICompletionParameters, OpenAIChatCompletionParameters]:
        return OpenAICompletionParameters.from_dict(
            kwargs
        ), OpenAIChatCompletionParameters.from_dict(kwargs)

    def serve(self, model_id_or_path: str = None):
        model_id_or_path = self.model_id_or_path if model_id_or_path is None else model_id_or_path
        cmd = [
            [
                "python",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"),
            ],
            ["--pid", self._pid],
            ["--check-interval", self._check_interval],
            ["--api-key", self._api_key],
            ["--host", self._host],
            ["--port", self._port],
            ["--model", model_id_or_path],
            ["--tensor-parallel-size", self._tensor_parallel_size],
            ["--gpu-memory-utilization", self._gpu_memory_utilization],
            ["--swap-space", self._swap_space],
            (["--lora-modules", self._lora_modules] if self._lora_modules else []),
            ["--disable-log-requests"],
            # ["--enable-prefix-caching"],
        ]
        cmd = [str(x) for entry in cmd for x in entry]

        dgt_logger.info("Starting vllm server with command:\n\t%s", " ".join(cmd))

        self._vllm_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        lines = []
        while True:
            time.sleep(1)

            try:
                status_code = requests.get(
                    f"http://{self._host}:{self._port}/ping",
                    timeout=300,
                ).status_code
            except requests.exceptions.ConnectionError:
                status_code = None

            if status_code == 200:
                dgt_logger.info("VLLM server has been initialized")
                break
            elif self._vllm_process.poll() is not None:
                lines.append(
                    "\n".join(
                        [
                            proc.read().decode("utf-8").strip()
                            for proc in [
                                self._vllm_process.stdout,
                                self._vllm_process.stderr,
                            ]
                        ]
                    ).strip()
                )
                # if process has error'd out, kill it
                dgt_logger.error(
                    "Error in vllm server instance. The full traceback is provided below:\n\n%s\n\n%s\n\n%s",
                    "*" * 50,
                    "\n".join([line for line in lines if line]),
                    "*" * 50,
                )
                raise SystemError("Underlying vllm process has terminated!")

    def release_model(self):
        dgt_logger.info("Releasing model by killing process %d", self._vllm_process.pid)
        base_proc = psutil.Process(self._vllm_process.pid)
        for child_proc in base_proc.children(recursive=True):
            child_proc.kill()
        base_proc.kill()

    def completion(self, *args, **kwargs) -> None:
        return self._vllm.completion(*args, **kwargs)

    def chat_completion(self, *args, **kwargs) -> None:
        return self._vllm.chat_completion(*args, **kwargs)
