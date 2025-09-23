# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import uuid

# Third Party
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import torch

# Local
from fms_dgt.base.block import Block, BlockData
from fms_dgt.base.registry import register_block
from fms_dgt.utils import dgt_logger


@dataclass(kw_only=True)
class MagpieDistanceBlockData(BlockData):
    """
    Data type for Magpie tagging block
    """

    # ===========================================================================
    #                       INPUT FIELDS
    # ===========================================================================
    magpie_input: Optional[str] = None

    # For multi-turn setting, the input must be array of dictionaries
    # with each item in dictionary containing "text"/"content" field
    magpie_mt_input: Optional[List[Dict[str, Any]]] = None

    # UUID for each entry
    id: Optional[str] = None

    # ===========================================================================
    #                       OUTPUT FIELDS
    # ===========================================================================
    magpie_tags: Optional[Dict] = None


@register_block("magpie_distance")
class MagpieDistance(Block):
    r"""Class for Magpie Distance based duplicate identification

    Args:
        sentence_model (Optional[str]): sentence model to use for encoding. Defaults to `sentence-transformers/all-mpnet-base-v2`
        distance_threshold (Optional[float]): similarity measure in distance. Defaults to `0.05`
        search_space_size (Optional[int]): FAISS search space size. Defaults to `500`
        search_batch_size (Optional[int]): FAISS search batch size. Defaults to `1024`
        encoding_batch_size (Optional[int]): number of entries to encode in a batch. Defaults to `65536`


    .. code-block:: python

        # Initialize dedup calculator
        dedup_calculator = MagpieDistance()

        # Sample data
        data = [
                {
                    "question": "what is capital of the United States of America?",
                    "answer": "Washington D.C"
                },
                {
                    "question": "What is biggest star in our solar system?",
                    "answer": "Sun is the biggest star in our solar system."
                }
            ]

        # Invoke dedup calculator
        dedup_calculator(data)
    """

    DATA_TYPE = MagpieDistanceBlockData

    def __init__(
        self,
        sentence_model: str = "sentence-transformers/all-mpnet-base-v2",
        distance_threshold: float = 0.05,
        search_space_size: int = 500,
        search_batch_size: int = 1024,
        encoding_batch_size: int = 65536,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        **kwargs: Any,
    ) -> None:
        # Set default values for "input_map" & "output_map", if necessary
        if input_map is None:
            input_map = {
                "id": "id",
                "input": "magpie_input",
            }

        if output_map is None:
            output_map = {
                "id": "id",
                "magpie_tags": "magpie_tags",
            }

        # Initialize parent
        super().__init__(input_map=input_map, output_map=output_map, **kwargs)

        # Set additional necessary variables
        self._distance_threshold = distance_threshold
        self._search_space_size = search_space_size
        self._search_batch_size = search_batch_size
        self._encoding_batch_size = encoding_batch_size

        self._model = SentenceTransformer(sentence_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device=device, dtype=torch.float32)

        dgt_logger.info("The model is loaded on device: %s", self._model.device)

    def index(self, dataset: Dataset):
        inputs = dataset["text"]

        # Encode the sentences in the dataset into vectors
        embeddings = []
        for i in range(0, len(inputs), self._encoding_batch_size):
            batch_sentences = inputs[i : i + self._encoding_batch_size]
            batch_embeddings = self._model.encode(
                batch_sentences, convert_to_tensor=True, show_progress_bar=True
            )
            embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate the embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        # Add the encoded vectors to the dataset
        dataset = dataset.add_column("embeddings", embeddings.tolist())

        # Build the Faiss index on the dataset
        dataset.add_faiss_index(column="embeddings", index_name="embeddings")

        return dataset, embeddings

    def compute(
        self,
        dataset: Dataset,
        embeddings: np.ndarray,
        instances: List[MagpieDistanceBlockData],
    ):

        n_batches = (
            len(dataset) + self._search_batch_size - 1
        ) // self._search_batch_size

        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * self._search_batch_size
            end_idx = min((batch_idx + 1) * self._search_batch_size, len(dataset))

            batch_indices = list(range(start_idx, end_idx))

            # Obtain the embeddings for the current batch
            batch_embeddings = embeddings[batch_indices]

            # Search for the most similar examples
            search_results = dataset.search_batch(
                index_name="embeddings",
                queries=batch_embeddings,
                k=min(self._search_space_size, len(dataset)),
            )
            total_scores = search_results.total_scores
            total_indices = search_results.total_indices

            for idx, indices in enumerate(total_indices):
                scores = total_scores[idx]
                indices = total_indices[idx]
                min_distance = float(scores[1])  # should exclude itself
                dataset[start_idx + idx]["min_distance"] = min_distance

                filtered_indices = [
                    index
                    for index, score in zip(indices, scores)
                    if score < self._distance_threshold
                ]
                # Should remove itself
                filtered_indices = [
                    index for index in filtered_indices if index != start_idx + idx
                ]

                if len(filtered_indices) == 0:
                    repeat_count = 0
                    min_similar_uuid = None
                    dataset[start_idx + idx]["repeat_count"] = repeat_count
                    dataset[start_idx + idx]["min_similar_uuid"] = min_similar_uuid
                else:
                    min_similar_uuidx = int(min(filtered_indices))
                    if min_similar_uuidx >= start_idx + idx:
                        min_similar_uuid = dataset[start_idx + idx]["id"]
                    else:
                        min_similar_uuid = dataset[min_similar_uuidx]["id"]

                    repeat_count = len(filtered_indices)

                    dataset[start_idx + idx]["repeat_count"] = repeat_count
                    dataset[start_idx + idx]["min_similar_uuid"] = min_similar_uuid

                # Initialize results container, if necessary
                if not instances[start_idx + idx].magpie_tags:
                    instances[start_idx + idx].magpie_tags = {}

                # Store results
                instances[start_idx + idx].magpie_tags[
                    "min_neighbor_distance"
                ] = min_distance
                instances[start_idx + idx].magpie_tags["repeat_count"] = repeat_count
                instances[start_idx + idx].magpie_tags[
                    "min_similar_uuid"
                ] = min_similar_uuid

    def execute(
        self, inputs: List[MagpieDistanceBlockData]
    ) -> List[MagpieDistanceBlockData]:
        # Cast to list, if necessary
        inputs = [entry for entry in inputs] if isinstance(inputs, map) else inputs

        # Convert to HuggingFace dataset object
        data = []
        for entry in inputs:
            # Add "id" field, if not present already
            if entry.id is None:
                entry.id = str(uuid.uuid4())

            # Initialize record
            record = {"id": entry.id}

            # Flatten input, if in Multi-turn settings
            if entry.magpie_mt_input:
                user_utterance_texts = []
                for utterance in entry.magpie_mt_input:
                    # Identify role field
                    if "from" in utterance:
                        role_field = "from"
                    elif "role" in utterance:
                        role_field = "role"
                    elif "speaker" in utterance:
                        role_field = "speaker"
                    else:
                        raise ValueError(
                            "\"magpie_mt_input\" should have a 'from' field or a 'role' field or a 'speaker' field to signify whether it was a user or assistant utterance"
                        )

                    # Identify text/content field
                    if "value" in utterance:
                        txt_field = "value"
                    elif "content" in utterance:
                        txt_field = "content"
                    elif "text" in utterance:
                        txt_field = "text"
                    else:
                        # Skipping messages without content
                        continue
                    if utterance[role_field] == "user":
                        user_utterance_texts.append(utterance[txt_field])

                record["text"] = ("\n\n".join(user_utterance_texts)).strip()
            else:
                record["text"] = entry.magpie_input.strip()

            # Add record
            data.append(record)

        # Run indexing and dedup, only if data is available
        if data:
            dataset = Dataset.from_list(data)

            # Index data
            dataset, embeddings = self.index(dataset)

            # Add similarity scores
            self.compute(dataset, embeddings=embeddings, instances=inputs)

        # Return
        return inputs
