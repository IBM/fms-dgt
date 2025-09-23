# Standard
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Dict

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter


@register_formatter("formatters/native")
class NativeFormatter(Formatter):
    def __init__(self, format: Dict[str, str] | str):
        self._format = format

    def apply(self, data: DataPoint | Dict, *args, **kwargs):
        # Convert to dictionary
        data_to_format = asdict(data) if is_dataclass(data) else data

        # Initialize formatted data
        formatted_data = deepcopy(self._format)
        if isinstance(self._format, dict):
            for key in formatted_data:
                for df_k, df_v in data_to_format.items():
                    formatted_data[key] = formatted_data[key].replace(
                        "{{" + df_k + "}}", str(df_v)
                    )

        elif isinstance(self._format, str):
            for df_k, df_v in data_to_format.items():
                formatted_data = formatted_data.replace("{{" + df_k + "}}", str(df_v))

        else:
            raise ValueError(
                f'Unsupported format type ({type(self._format)}). "format" must be specified either as a string or dictionary.'
            )

        return formatted_data
