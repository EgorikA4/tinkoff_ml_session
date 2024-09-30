"""Module that provides DescriptionPipeline."""
import sys

sys.path.append('../.')

from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinkoff_ml_session import consts
from tinkoff_ml_session.description.models import CLIPVisionTower

from . import setup


class DescriptionPipeline:
    """Pipeline for describing an image by text."""

    def __init__(self) -> None:
        """Initialize OmniFusion parts."""
        self._setup()
        self.tokenizer: AutoTokenizer = setup.get_tokenizer()
        self.model: AutoModelForCausalLM = setup.get_model()
        self.projection: Any = setup.get_projection()
        self.special_embs: Any = setup.get_special_embs()
        self.clip: CLIPVisionTower = setup.get_clip()

    def default_gen_params(self) -> dict:
        """Create default generation params.

        Returns:
            The dictionary of parameters.
        """
        bad_words_ids = self.tokenizer(
            ['\n', '</s>', ':'],
            add_special_tokens=False,
        ).input_ids + [[13]]
        return {
            'do_sample': False,
            'max_new_tokens': 200,
            'early_stopping': True,
            'num_beams': 3,
            'repetition_penalty': 1.0,
            'remove_invalid_values': True,
            'eos_token_id': 2,
            'pad_token_id': 2,
            'forced_eos_token_id': 2,
            'use_cache': True,
            'no_repeat_ngram_size': 4,
            'bad_words_ids': bad_words_ids,
            'num_return_sequences': 1,
        }

    def forward(
        self,
        image: Image,
        query: str,
        gen_params: dict,
    ) -> str:
        """Get image descriptions.

        Args:
            image: Image - an image.
            query: str - a prompt.
            gen_params: dict - description generation parameters.

        Returns:
            Image description.
        """
        with torch.no_grad():
            image_features = self.clip.image_processor(
                image,
                return_tensors='pt',
            )
            image_embedding = self.clip(
                image_features['pixel_values'],
            ).to(
                device=consts.DEVICE,
                dtype=torch.bfloat16,
            )

            projected_vision_embeddings = self.projection(
                image_embedding,
            ).to(
                device=consts.DEVICE,
                dtype=torch.bfloat16,
            )
            prompt_ids = self.tokenizer.encode(
                f'{consts.OMNIFUSION_PROMPT}',
                add_special_tokens=False,
                return_tensors='pt',
            ).to(device=consts.DEVICE)

            question_ids = self.tokenizer.encode(
                query,
                add_special_tokens=False,
                return_tensors='pt',
            ).to(device=consts.DEVICE)

            prompt_embeddings = self.model.model.embed_tokens(
                prompt_ids,
            ).to(torch.bfloat16)
            question_embeddings = self.model.model.embed_tokens(
                question_ids,
            ).to(torch.bfloat16)

            embeddings = torch.cat(
                [
                    prompt_embeddings,
                    self.special_embs['SOI'][None, None, ...],
                    projected_vision_embeddings,
                    self.special_embs['EOI'][None, None, ...],
                    self.special_embs['USER'][None, None, ...],
                    question_embeddings,
                    self.special_embs['BOT'][None, None, ...],
                ],
                dim=1,
            ).to(dtype=torch.bfloat16, device=consts.DEVICE)
            out = self.model.generate(
                inputs_embeds=embeddings,
                **gen_params,
            )
        out = out[:, 0:]
        return self.tokenizer.batch_decode(out)[0]

    def _setup(self) -> None:
        setup.setup_tokenizer(
            consts.OMNIFUSION_REPO_ID,
        )
        setup.setup_model(
            consts.OMNIFUSION_REPO_ID,
        )
        setup.setup_projection()
        setup.setup_special_embs()
        setup.setup_clip()
