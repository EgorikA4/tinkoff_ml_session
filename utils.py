"""Module for replacing the image background."""

import numpy as np
import torch


def combine_masks(masks: torch.Tensor, image: np.array) -> torch.Tensor:
    """Combine masks obtained by the GroundedSAM model.

    Args:
        masks: torch.Tensor - masks of goods.
        image: np.array - the image that needs to change the background.

    Returns:
        A single mask.
    """
    height, width = image.shape[:2]
    combined_mask = torch.zeros((height, width))
    for mask in masks:
        combined_mask = torch.logical_or(mask[0], combined_mask)
    return combined_mask


def change_bg(
    masks: torch.Tensor,
    image: np.array,
    bg_color: int | tuple[int],
) -> np.array:
    """Replace the background of the product.

    Args:
        masks: torch.Tensor - masks of goods.
        image: np.array - the image that needs to change the background.
        bg_color: int | tuple[int] - background color in RGB format.

    Returns:
        An image with a replaced background.
    """
    mask = combine_masks(masks, image).numpy()
    rgb_mask = np.stack(
        (mask,) * 3,
        axis=-1,
        dtype=np.uint8,
    )

    background = np.full(image.shape, bg_color, dtype=np.uint8)
    fg_masked = rgb_mask * image

    rgb_mask_inverted = np.logical_not(rgb_mask)
    bg_masked = background * rgb_mask_inverted

    return np.bitwise_or(fg_masked, bg_masked)
