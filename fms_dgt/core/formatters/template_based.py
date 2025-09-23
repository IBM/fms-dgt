# Standard
from dataclasses import asdict, is_dataclass
from typing import Dict

# Third Party
from jinja2 import Environment

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter


@register_formatter("formatters/template")
class TemplateBasedFormatter(Formatter):
    def __init__(self, format: Dict[str, str] | str):
        self._environment = Environment()
        self._format = format

    def apply(self, data: DataPoint, *args, **kwargs):
        # Convert to dictionary
        data_to_format = asdict(data) if is_dataclass(data) else data

        # Initialize formatted data
        if isinstance(self._format, dict):
            formatted_data = {}
            for key, value in self._format.items():
                template = self._environment.from_string(value)
                formatted_data[key] = template.render(data_to_format)

            return formatted_data
        elif isinstance(self._format, str):
            template = self._environment.from_string(self._format)
            return template.render(data_to_format)
        else:
            raise ValueError(
                f'Unsupported format type ({type(self._format)}). "format" must be specified either as a string or dictionary.'
            )
