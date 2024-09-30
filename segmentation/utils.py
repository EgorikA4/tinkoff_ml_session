"""Module that provides SegmentationPipeline."""
from typing import Any

import numpy as np
import torch

import consts

from . import setup
from .GroundingDINO.groundingdino.util import box_ops, inference
from .segment_anything.segment_anything.predictor import SamPredictor


class SegmentationPipeline:
    """Pipeline for segment and detect objects on image."""

    def __init__(self) -> None:
        """Initialize segmentation and detection models."""
        self._setup()
        self.groundingdino: Any = setup.get_groundingdino()
        self.sam: SamPredictor = setup.get_sam()

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> torch.Tensor:
        """Detect boxes on image by prompt.

        Args:
            image_path: str - path to image.
            text_prompt: str - a prompt that determines objects on the image.
            box_threshold: float = 0.3 - the threshold of the boxes.
            text_threshold: float = 0.25 - the threshold of the text.

        Returns:
            Tensor with boxes.
        """
        _, image = self._preprocessing_image(image_path)

        boxes, _, _ = inference.predict(
            model=self.groundingdino,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return boxes

    def forward(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> torch.Tensor:
        """Segment objects on image by prompt.

        Args:
            image_path: str - path to image.
            text_prompt: str - a prompt that determines objects on the image.
            box_threshold: float = 0.3 - the threshold of the boxes.
            text_threshold: float = 0.25 - the threshold of the text.

        Returns:
            Tensor with masks.
        """
        image_source, _ = self._preprocessing_image(image_path)
        boxes = self.detect(
            image_path,
            text_prompt,
            box_threshold,
            text_threshold,
        )
        self.sam.set_image(image_source)
        height, width, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor(
            [width, height, width, height],
        )

        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes_xyxy.to(consts.DEVICE),
            image_source.shape[:2],
        )
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()

    def _setup(self) -> None:
        setup.setup_groundingdino(
            consts.DINO_CONFIG_PATH,
            consts.DINO_CKPT_PATH,
        )
        setup.setup_sam(
            consts.SAM_CKPT_PATH,
        )

    def _preprocessing_image(
        self,
        image_path: str,
    ) -> tuple[np.array, torch.Tensor]:
        return inference.load_image(image_path)
