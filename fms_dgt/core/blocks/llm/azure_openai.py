# Standard
from typing import Any
import logging

# Third Party
import openai

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.core.blocks.llm.openai import OpenAI
from fms_dgt.core.resources.api import ApiKeyResource

# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("azure-openai")
class AzureOpenAI(OpenAI):
    def __init__(
        self,
        base_url: str,
        api_version: str | None = None,
        call_limit: int = 10,
        init_tokenizer: bool = False,
        timeout: float = 300,
        **kwargs: Any,
    ):

        # Step 1: LM provider connection arguments
        api_resource: ApiKeyResource = get_resource(
            "api", key_name="AZURE_OPENAI_API_KEY", call_limit=call_limit
        )

        # Step 2: Initialize parent
        super().__init__(
            init_tokenizer=init_tokenizer,
            api_key=api_resource.key,
            call_limit=call_limit,
            base_url=base_url,
            **kwargs,
        )

        # Step 3: Initialize OpenAI clients
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_resource.key,
            azure_endpoint=base_url,
            api_version=api_version,
            timeout=timeout,
        )
