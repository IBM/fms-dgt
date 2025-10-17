# Standard
from typing import Dict
import os
import random
import shutil
import sys

# Third Party
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import numpy as np
import pandas as pd
import torch

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.core.blocks.trainer.trainer import Trainer as TrainerBlock
from fms_dgt.core.blocks.trainer.trainer import TrainingException, get_model_dir
from fms_dgt.public.databuilders.time_series.utils import embeddings_to_text
from fms_dgt.utils import dgt_logger


@register_block("public/trainers/sdforger-tuning")
class SDForgerTuningBlock(TrainerBlock):
    def __init__(
        self,
        seed=42,
        **kwargs,
    ):
        """Create SDForger trainer instance.

        Args:
            seed (int, optional): Seed value for training. Defaults to 42.
        """
        super().__init__(**kwargs)
        self.set_seed(seed)

    @property
    def compute_device(self) -> str:
        if sys.platform == "darwin":
            return "mps"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def set_seed(self, seed):
        self.seed = seed
        os.environ["PYTHONHASHSEED"] = "0"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_quantization_config(self, dtype="float32", k_bit=None) -> BitsAndBytesConfig | None:
        if k_bit:
            if k_bit == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype
                )
            elif k_bit == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("k_bit must be either 4 or 8")
        else:
            quantization_config = None

        return quantization_config

    def get_model(self, model_id_or_path, model_args) -> AutoModel:
        """Load Huggingface model from the path.

        Args:
            model_id_or_path (str): The path to load the language model.
            model_args (dict): model args for the llm.

        Returns:
            AutoModel: Huggingface model
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=model_args["dtype"],
                quantization_config=self.get_quantization_config(
                    dtype=model_args["dtype"], k_bit=model_args["k_bit"]
                ),
                trust_remote_code=model_args["trust_remote_code"],
                ignore_mismatched_sizes=model_args["ignore_mismatched_sizes"],
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=model_args["trust_remote_code"],
            )
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
        except Exception as err:
            raise ValueError(f"Error creating Model: {str(err)}") from err

        model.to(self.compute_device)
        return model, tokenizer

    def train(
        self,
        output_dir: str,
        dataset: DATASET_TYPE,
        model_args: Dict,
        sdforger_params: Dict,
        *args,
        iteration: int = 1,
        model_id_or_path: str | None = None,
        **kwargs,
    ):
        """Train LLM on the given dataset.

        Args:
            output_dir (str): Directory to store train results.
            dataset (DATASET_TYPE): Training dataset.
            model_args (Dict): Arguments for model initialization.
            sdforger_params (Dict): Dictionary of SDForger params.
            iteration (int): Iteration number.
            model_id_or_path (str | None): Optional model path or id.

        Returns:
            str: Path to the best saved model.
        """
        model_id_or_path = self.model_id_or_path or model_id_or_path
        if model_id_or_path is None:
            raise ValueError(
                "Must provide `model_id_or_path` during initializing SDForgerTuningBlock or when calling train()"
            )

        try:
            # Load model and tokenizer
            model_args["k_bit"] = sdforger_params.get("k_bit", None)
            model, tokenizer = self.get_model(model_id_or_path, model_args)

            # Validate and convert dataset
            if isinstance(dataset, np.ndarray):
                # assigning fantom column names
                columns = [f"column_{i+1}" for i in range(dataset.shape(1))]
                dataset = pd.DataFrame(dataset, columns=columns)
            elif not isinstance(dataset, pd.DataFrame):
                raise TypeError("Dataset must be a pandas dataframe or numpy array")

            # Shuffle and convert to HuggingFace dataset
            dataset = dataset.sample(frac=1, random_state=self.seed)
            hf_dataset = Dataset.from_pandas(dataset)

            # Add text column from embeddings
            # pylint: disable=no-value-for-parameter
            hf_dataset = hf_dataset.add_column(
                "data2text",
                [
                    embeddings_to_text(
                        row=row,
                        columns=dataset.columns.to_list().copy(),
                        eos_token=tokenizer.eos_token,
                        input_tokens_precision=sdforger_params["input_tokens_precision"],
                    )
                    for row in tqdm(hf_dataset, desc="Generating text from embeddings")
                ],
            )

            # Tokenization
            def preprocess_function(row):
                model_inputs = tokenizer(row["data2text"], padding=True)
                model_inputs["labels"] = model_inputs["input_ids"]
                return model_inputs

            tokenized_dataset = hf_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=hf_dataset.column_names,
            )

            # Split into train (80%) and validation (20%)
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=self.seed)
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]

            # Training arguments
            train_args = {
                "learning_rate": self.training_args.get("learning_rate", 5e-5),
                "num_train_epochs": self.training_args.get("num_train_epochs", 2),
                "per_device_train_batch_size": self.training_args.get(
                    "per_device_train_batch_size", 4
                ),
            }

            # tuned model path
            tuned_model_path = os.path.join(get_model_dir(output_dir) + f"_iter-{iteration}")
            if os.path.exists(
                tuned_model_path
            ):  # Delete what's already present in the model directory
                for file_name in os.listdir(tuned_model_path):
                    file_path = os.path.join(tuned_model_path, file_name)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                dgt_logger.info(
                    "Model files were found at %s. All files have been deleted, and the new model will replace the old one.",
                    tuned_model_path,
                )

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tuned_model_path,
                    **train_args,
                    adam_epsilon=1e-04,
                    logging_strategy="steps",
                    logging_steps=10,
                    eval_strategy="steps",
                    eval_steps=5,
                    save_strategy="steps",
                    save_steps=100,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                ),
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=val_data,
                callbacks=[EvalLossEarlyStopping(patience=5)],
            )

            trainer.train()

            # Save best model
            best_model_path = os.path.join(tuned_model_path, "best")
            trainer.save_model(best_model_path)

        except Exception as err:
            raise TrainingException(f"Finetuning LLM failed: {str(err)}") from err

        return best_model_path


class EvalLossEarlyStopping(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        if self.best_loss is None or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True

        return control
