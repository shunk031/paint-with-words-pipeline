import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch as th
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from PIL.Image import Image as PilImage
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from paint_with_words.helper.aliases import RGB
from paint_with_words.models.attention import paint_with_words_forward

logger = logging.getLogger(__name__)


@dataclass
class SeparatedImageContext(object):
    word: str
    token_ids: List[int]
    color_map_th: th.Tensor


class PaintWithWordsPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        # replace scheduler to LMSDiscreteScheduler
        self.scheduler = LMSDiscreteScheduler.from_config(self.scheduler.config)

        # replace cross attention to the paint with words one
        self.replace_cross_attention()

    def replace_cross_attention(
        self, cross_attention_name: str = "CrossAttention"
    ) -> None:

        for m in self.unet.modules():
            if m.__class__.__name__ == cross_attention_name:
                m.__class__.__call__ = paint_with_words_forward

    def separate_image_context(self, img: PilImage, color_context: Dict[RGB, str]):

        assert img.width % 32 == 0 and img.height % 32 == 0, img.size

        separated_image_and_context: List[SeparatedImageContext] = []

        for rgb_color, word_with_weight in color_context.items():

            # e.g.,
            # rgb_color: (0, 0, 0)
            # word_with_weight: cat,1.0

            # cat,1.0 -> ["cat", "1.0"]
            word_and_weight = word_with_weight.split(",")
            # ["cat", "1.0"] -> 1.0
            word_weight = float(word_and_weight[-1])
            # ["cat", "1.0"] -> cat
            word = ",".join(word_and_weight[:-1])

            logger.info(
                f"input = {word_with_weight}; word = {word}; weight = {word_weight}"
            )

            word_input = self.tokenizer(
                word,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            word_as_tokens = word_input["input_ids"]

            img_where_color_np = (np.array(img) == rgb_color).all(axis=-1)
            if not img_where_color_np.sum() > 0:
                logger.warning(
                    f"Warning : not a single color {rgb_color} not found in image"
                )

            img_where_color_th = th.tensor(
                img_where_color_np,
                dtype=th.float32,
                device=self.device,
            )
            img_where_color_th = img_where_color_th * word_weight

            image_context = SeparatedImageContext(
                word=word,
                token_ids=word_as_tokens,
                color_map_th=img_where_color_th,
            )
            separated_image_and_context.append(image_context)

        if len(separated_image_and_context) == 0:
            image_context = SeparatedImageContext(
                word="",
                token_ids=[-1],
                color_map_th=th.zeros((img.width, img.height), dtype=th.float32),
            )
            separated_image_and_context.append(image_context)

        return separated_image_and_context

    def _tokens_img_attention_weight(
        img_context_seperated, tokenized_texts, ratio: int = 8
    ):

        token_lis = tokenized_texts["input_ids"][0].tolist()
        w, h = img_context_seperated[0][1].shape

        w_r, h_r = w // ratio, h // ratio

        ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)

        for v_as_tokens, img_where_color in img_context_seperated:
            is_in = 0

            for idx, tok in enumerate(token_lis):
                if token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                    is_in = 1

                    # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                    ret_tensor[:, idx : idx + len(v_as_tokens)] += (
                        _img_importance_flatten(img_where_color, ratio)
                        .reshape(-1, 1)
                        .repeat(1, len(v_as_tokens))
                    )

            if not is_in == 1:
                print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

        return ret_tensor
