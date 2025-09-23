# Standard
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import re

# Third Party
import jinja2

# Local
from fms_dgt.utils import dgt_logger


class PromptTemplate(ABC):

    @abstractmethod
    def encode(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )


class JinjaPromptTemplate(PromptTemplate):
    """
    Base class for Jinja2 based prompt templates
    """

    def __init__(
        self,
        template: str | None = None,
        template_path: str | None = None,
        stop: List[str] | None = None,
        jinja_globals: Any | None = None,
    ):
        if template is None and template_path is None:
            raise ValueError('Either "template" or "template_path" must be provided.')

        self.environment = jinja2.Environment()
        if template and template_path:
            dgt_logger.warning(
                'Both "template" and "template_path" are provided. Using "template".'
            )
            self._prompt_template = self.environment.from_string(
                template, globals=globals
            )
        elif template:
            self._prompt_template = self.environment.from_string(
                template, globals=jinja_globals
            )
        else:
            # Convert template path to string, if necessary
            if isinstance(template_path, Path):
                template_path = str(template_path)

            if not template_path.endswith("txt"):
                raise ValueError(
                    f"Provided non text file ({template_path}) for template path"
                )
            self._prompt_template = self.environment.from_string(
                open(template_path, "r", encoding="utf-8").read(), globals=jinja_globals
            )

        self._stop = stop if stop else ["[</s>]"]

    @property
    def stop(self) -> List[str]:
        return self._stop

    def encode(
        self,
        render_dict: Dict,
    ):
        return self._prompt_template.render(**render_dict)


class CustomPromptTemplate(PromptTemplate):

    def __init__(
        self,
        template: str,
        stop: List[str] = None,
    ):
        if template is None or not template:
            raise ValueError('"template" must be specified.')
        self._template = template

        # Identify all variables specified in {{VARIABLE_NAME}} format
        self._variables = re.findall(r"\{\{(.*?)\}\}", self._template)
        self._stop = stop

    @property
    def prompt(self):
        return self._template

    @property
    def stop(self) -> List[str]:
        return self._stop

    def encode(self, **kwargs: Any) -> str:
        """Format the prompt to string with the optional variables and return"""
        prompt_str = self._template
        for k, v in kwargs.items():
            prompt_str = prompt_str.replace("{{" + k + "}}", v)
        return prompt_str
