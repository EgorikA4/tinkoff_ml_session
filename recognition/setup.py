"""Module that initializes model with processor."""

import torch
import transformers

import consts

llava_model: transformers.LlavaForConditionalGeneration
llava_processor: transformers.AutoProcessor


def setup_llava_model() -> None:
    """Set the LLava-v1.5 4bit model."""
    global llava_model
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    llava_model = transformers.LlavaForConditionalGeneration.from_pretrained(
        consts.LLAVA_REPO_ID,
        cache_dir=consts.LLAVA_CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )


def setup_llava_processor() -> None:
    """Set the LLava-v1.5 processor."""
    global llava_processor
    llava_processor = transformers.AutoProcessor.from_pretrained(
        consts.LLAVA_REPO_ID,
    )


def get_llava_model() -> transformers.LlavaForConditionalGeneration:
    """Get the LLava-v1.5 model.

    Returns:
        LlavaForConditionalGeneration object.
    """
    global llava_model
    return llava_model


def get_llava_processor() -> transformers.AutoProcessor:
    """Get the processor for LLava-v1.5 model.

    Returns:
        AutoProcessor object.
    """
    global llava_processor
    return llava_processor
