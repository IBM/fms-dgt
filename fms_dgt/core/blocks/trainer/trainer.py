# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict
import abc
import json
import os

# Third Party
import torch

# Local
from fms_dgt.base.block import Block
from fms_dgt.constants import DATASET_TYPE


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass
class TrainerData:
    input: str
    output: str

    def to_dict(self):
        return asdict(self)


def get_model_dir(output_path: str):
    return os.path.join(output_path, "model")


# ===========================================================================
#                       EXCEPTION CLASS
# ===========================================================================
class TrainingException(Exception):
    pass


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
class Trainer(Block):
    def __init__(
        self,
        model_id_or_path: str | None = None,
        config_path: str = None,
        num_gpus: int = None,
        logging_steps: int = 100,
        save_steps: int = 50,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_steps: int = None,
        learning_rate: float = 0.00001,
        num_train_epochs: int = 1,
        log_level: str = "debug",
        save_total_limit: int = 1,
        # known good settings
        max_seq_length: int = 4096,
        torch_dtype: str = "bfloat16",
        optim: str = "adamw_torch_fused",
        optim_args: str = "lr=5.0e-5,weight_decay=0.1,eps=1e-10",
        **kwargs: Any,
    ) -> None:
        """Initialize a trainer that trains a model on a dataset input.

        Args:
            config_path (Any): path to config used for trainer
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._model_id_or_path = model_id_or_path
        self._config_path = config_path

        self._num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus

        training_args = {
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "save_total_limit": save_total_limit,
            "log_level": log_level,
            # known good settings
            "max_seq_length": max_seq_length,
            "torch_dtype": torch_dtype,
            "optim": optim,
            "optim_args": optim_args,
        }
        self._training_args = {k: v for k, v in training_args.items() if v is not None}

        self._kwargs = kwargs

    @property
    def model_id_or_path(self) -> str:
        """Returns the model_id_or_path.

        Returns:
            str: model_id_or_path
        """
        return self._model_id_or_path

    @property
    def training_args(self) -> Dict:
        """Returns a dictionary of training arguments.

        Returns:
            Dict: A dictionary of training arguments
        """
        return self._training_args

    def save_dataset(self, data: DATASET_TYPE, file_path: str, encoding: str = "utf-8"):
        # Maket ancestral directories, if required
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write dataset
        with open(file_path, mode="w", encoding=encoding) as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> str:
        return self.train(*args, **kwargs)

    @abc.abstractmethod
    def train(
        self,
        output_dir: str,
        data: DATASET_TYPE,
        *args,
        model_id_or_path: str | None = None,
        **kwargs,
    ) -> str:
        """Run training and return a model

        Args:
            output_dir (str): Directory to output model checkpoints
            data (DATASET_TYPE): All training data from one or more tasks
            model_id_or_path (str | None): Model to initialize from
            kwargs (Any): Additional keyword arguments to pass to the base class.

        Returns:
            str: Path to model that was trained
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    def release_model(self):
        pass

    def close(self):
        self.release_model()
