# Standard
from typing import Iterable, Optional

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.public.blocks.magpie.distance import MagpieDistance
from fms_dgt.public.blocks.magpie.filter import MagpieFilter
from fms_dgt.public.blocks.magpie.tag import MagpieTagger
from fms_dgt.public.databuilders.magpie.task import (
    MagpieTransformData,
    MagpieTransformTask,
)
from fms_dgt.utils import dgt_logger

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
_MP_INPUT_DATA = "MP_INP_DATA"

# ===========================================================================
#                       MAIN CLASS
# ===========================================================================


@register_data_builder("magpie")
class MagpieTransformDataBuilder(TransformationDataBuilder):
    """Class for Magpie Transformation"""

    TASK_TYPE: MagpieTransformTask = MagpieTransformTask

    # tagger is the Magpie based tagger to rate synthetic examples
    tagger: Optional[MagpieTagger]
    dedup: Optional[MagpieDistance]
    filter: Optional[MagpieFilter]

    def __call__(
        self,
        data_points: MagpieTransformData,
    ) -> Iterable[dict]:
        # Step 1: Initialize outputs from inputs
        outputs = [{**data_point.input, _MP_INPUT_DATA: data_point} for data_point in data_points]

        # Step 2: Invoke magpie tagger, if requested
        if hasattr(self, "tagger"):
            dgt_logger.info(
                'Tagging %d data points with "%s"...',
                len(outputs),
                ".".join([self.tagger.__module__, self.tagger.__class__.__name__]),
            )

            # Step 2.a: Tag
            tagged_outputs = self.tagger(outputs)

            # Step 2.b: Re-build outputs using tagger outputs
            outputs = [
                {
                    **tagged_output[_MP_INPUT_DATA].input,
                    "magpie_tags": tagged_output["magpie_tags"],
                    _MP_INPUT_DATA: tagged_output[_MP_INPUT_DATA],
                }
                for tagged_output in tagged_outputs
            ]

        # Step 3: Invoke magpie deduplicator, if requested
        if hasattr(self, "dedup"):
            dgt_logger.info(
                'Deduping %d data points with "%s"...',
                len(outputs),
                ".".join([self.dedup.__module__, self.dedup.__class__.__name__]),
            )

            # Step 3.a: Deduplicate
            dedup_outputs = self.dedup(outputs)

            # Step 3.b: Re-build outputs using dedup outputs
            outputs = [
                {
                    **dedup_output[_MP_INPUT_DATA].input,
                    "magpie_tags": dedup_output["magpie_tags"],
                    "id": dedup_output["id"],
                    _MP_INPUT_DATA: dedup_output[_MP_INPUT_DATA],
                }
                for dedup_output in dedup_outputs
            ]

        # Step 4: Invoke magpie filteration, if requested
        if hasattr(self, "filter"):
            dgt_logger.info(
                'Filtering %d data points with "%s"...',
                len(outputs),
                ".".join([self.filter.__module__, self.filter.__class__.__name__]),
            )

            # Step 4.a: Filter
            filtered_outputs = self.filter(
                [
                    {
                        **output,
                        "store_names": self.get_block_store_names(
                            block_name=self.filter.name, task_name=output["task_name"]
                        ),
                    }
                    for output in outputs
                ]
            )

            # Build outputs using filtered outputs
            outputs = [
                {
                    **filtered_output[_MP_INPUT_DATA].input,
                    "magpie_tags": filtered_output["magpie_tags"],
                    _MP_INPUT_DATA: filtered_output[_MP_INPUT_DATA],
                }
                for filtered_output in filtered_outputs
            ]

        # Step 5: Return
        return [
            MagpieTransformData(
                task_name=output[_MP_INPUT_DATA].task_name,
                is_seed=False,
                input=output[_MP_INPUT_DATA].input,
                magpie_tags=output["magpie_tags"],
            )
            for output in outputs
        ]
