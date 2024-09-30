"""Module that initializes models variables."""

from typing import Any

from GroundingDINO.groundingdino.util.inference import load_model
from segment_anything import SamPredictor, build_sam

import consts

groundingdino_model: Any
sam_predictor: SamPredictor


def setup_groundingdino(
    model_config_path: str,
    model_checkpoint_path: str,
) -> None:
    """Set a GroundingDINO model.

    Args:
        model_config_path: str - configuration file path.
        model_checkpoint_path: str - checkpoint file path.
    """
    global groundingdino_model
    groundingdino_model = load_model(
        model_config_path,
        model_checkpoint_path,
        consts.DEVICE,
    )


def setup_sam(sam_checkpoint: str) -> None:
    """Set a GroundedSAM model.

    Args:
        sam_checkpoint: str - checkpoint file path.
    """
    global sam_predictor
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(
        consts.DEVICE,
    ))


def get_groundingdino() -> Any:
    """Get a GroundingDINO model.

    Returns:
        GroundingDINO model.
    """
    global groundingdino_model
    return groundingdino_model


def get_sam() -> SamPredictor:
    """Get a GroundedSAM model.

    Returns:
        GroundedSAM model.
    """
    global sam_predictor
    return sam_predictor
