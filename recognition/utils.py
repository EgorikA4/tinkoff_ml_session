"""Module that provides RecognitionPipeline."""

import torch
from PIL import Image

import consts

from . import setup


class RecognitionPipeline:
    """Pipeline for recognizing products in the picture."""

    def __init__(self) -> None:
        """Initialize LLava model and processor."""
        self._setup()
        self.llava_processor = setup.get_llava_processor()
        self.llava_model = setup.get_llava_model()

    def forward(self, image: Image, conversation: dict) -> str:
        """Get the names of the objects in the image.

        Args:
            image: Image - an image.
            conversation: dict - prompt for getting the names of objects.

        Returns:
            The names of objects in the picture.
        """
        prompt = self.llava_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.llava_processor(
            images=image,
            text=prompt,
            return_tensors='pt',
        ).to(consts.DEVICE, torch.float16)

        output = self.llava_model.generate(
            **inputs,
            max_new_tokens=consts.LLAVA_MAX_NEW_TOKENS,
            do_sample=False,
        )
        output = self.llava_processor.decode(
            output[0][2:],
            skip_special_tokens=True,
        )
        return output.split('ASSISTANT:')[-1].strip()

    def _setup(self) -> None:
        setup.setup_llava_model()
        setup.setup_llava_processor()
