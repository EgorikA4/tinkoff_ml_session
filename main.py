"""Module that provides FullPipeline."""

import numpy as np
from PIL import Image

import consts
from description.utils import DescriptionPipeline
from recognition.utils import RecognitionPipeline
from segmentation.utils import SegmentationPipeline
from utils import change_bg


class FullPipeline:
    """Process the entire image."""

    def __init__(self) -> None:
        """Initialize Recognition, Segmentation and Description pipelines."""
        self.recognize = RecognitionPipeline()
        self.segment = SegmentationPipeline()
        self.describe = DescriptionPipeline()

    def forward(
        self,
        image_path: str,
        bg_color: int | tuple[int],
    ) -> tuple[Image.Image, str]:
        """Get an image with a replaced background and its description.

        Args:
            image_path: str - path to source image.
            bg_color: str - background color in RGB format.

        Returns:
            An image with replaced background and description.
        """
        image = Image.open(image_path)
        product_category = self.recognize.forward(
            image,
            consts.ITEMS_RECOGNITION_CONVERSATION,
        )
        masks = self.segment.forward(
            image_path,
            consts.GROUNDING_DINO_PROMPT.format(product_category),
        )
        image_with_changed_bg = Image.fromarray(change_bg(
            masks,
            np.array(image),
            bg_color,
        ))
        description = self.describe.forward(
            image_with_changed_bg,
            consts.OMNIFUSION_QUERY,
            self.describe.default_gen_params(),
        )
        return image_with_changed_bg, description
