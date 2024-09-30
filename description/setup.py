"""Module that initializes a model variables."""
import os
import sys

sys.path.append('../.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinkoff_ml_session import consts
from tinkoff_ml_session.description.models import CLIPVisionTower

tokenizer: AutoTokenizer
model: AutoModelForCausalLM
projection: Any
special_embs: Any
clip: CLIPVisionTower


def setup_tokenizer(repo_id: str) -> None:
    """Set a tokenizer.

    Args:
        repo_id: str - repository name.
    """
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        subfolder='OmniMistral-v1_1/tokenizer',
        use_fast=False,
    )


def setup_model(repo_id: str) -> None:
    """Set an OmniFusion model.

    Args:
        repo_id: str - repository name.
    """
    global model
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder='OmniMistral-v1_1/tuned-model',
        torch_dtype=torch.bfloat16,
        device_map=consts.DEVICE,
    )


def setup_projection() -> None:
    """Set a projection."""
    global projection
    projection = torch.load(
        consts.OMNIFUSION_PROJECTION,
        map_location=consts.DEVICE,
    )


def setup_special_embs() -> None:
    """Set a special embeddings."""
    global special_embs
    special_embs = torch.load(
        consts.OMNIFUSION_EMBEDDINGS,
        map_location=consts.DEVICE,
    )


def setup_clip() -> None:
    """Set a clip."""
    global clip
    clip = CLIPVisionTower('openai/clip-vit-large-patch14-336')
    clip.load_model()
    clip = clip.to(device=consts.DEVICE, dtype=torch.bfloat16)


def get_tokenizer() -> AutoTokenizer:
    """Get a tokenizer.

    Returns:
        AutoTokenizer object.
    """
    global tokenizer
    return tokenizer


def get_model() -> AutoModelForCausalLM:
    """Get an OmniFusion model.

    Returns:
        AutoModelForCausalLM object.
    """
    global model
    return model


def get_projection() -> Any:
    """Get a projection.

    Returns:
        projection for OmniFusion model.
    """
    global projection
    return projection


def get_special_embs() -> Any:
    """Get a special embeddings.

    Returns:
        special embeddings for OmniFusion model.
    """
    global special_embs
    return special_embs


def get_clip() -> CLIPVisionTower:
    """Get a clip.

    Returns:
        CLIPVisionTower object.
    """
    global clip
    return clip
