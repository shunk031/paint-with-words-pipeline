import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as PilImage
from PIL.Image import Resampling
from transformers.tokenization_utils import PreTrainedTokenizer

from paint_with_words.helper.aliases import RGB, SeparatedImageContext

logger = logging.getLogger(__name__)


def load_image(image: Union[str, os.PathLike, PilImage]) -> PilImage:
    if isinstance(image, str) or isinstance(image, os.PathLike):
        image = Image.open(image)

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def get_resize_size(img: PilImage) -> Tuple[int, int]:
    w, h = img.size
    w, h = map(lambda x: x - x % 32, (w, h))
    return w, h


def resize_image(img: PilImage, w: int, h: int) -> PilImage:
    assert w % 32 == 0 and h % 32 == 0, (w, h)
    return img.resize((w, h), resample=Resampling.LANCZOS)


def flatten_image_importance(img_th: th.Tensor, ratio: int) -> th.Tensor:
    # shape: (h, w) -> (1, 1, h, w)
    img_th = img_th.view(1, 1, *img_th.size())

    # shape: (1, 1, w * scale_factor, h * scale_factor)
    scale_factor = 1 / ratio
    ret = F.interpolate(
        img_th,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=True,
    )
    # shape: (w * scale_factor, h * scale_factor)
    ret = ret.squeeze()

    return ret


def separate_image_context(
    tokenizer: PreTrainedTokenizer,
    img: PilImage,
    color_context: Dict[RGB, str],
    device: str,
) -> List[SeparatedImageContext]:
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

        word_input = tokenizer(
            word,
            max_length=tokenizer.model_max_length,
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
            device=device,
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


def calculate_tokens_image_attention_weight(
    tokenizer: PreTrainedTokenizer,
    input_prompt: str,
    separated_image_context_list: List[SeparatedImageContext],
    ratio: int,
    device: str,
) -> th.Tensor:
    prompt_token_ids = tokenizer(
        input_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    w, h = separated_image_context_list[0].color_map_th.shape
    w_r, h_r = w // ratio, h // ratio

    ret_tensor = th.zeros(
        (w_r * h_r, len(prompt_token_ids)), dtype=th.float32, device=device
    )

    for separated_image_context in separated_image_context_list:
        is_in = False
        context_token_ids = separated_image_context.token_ids
        context_image_map = separated_image_context.color_map_th

        for i, token_id in enumerate(prompt_token_ids):
            if prompt_token_ids[i : i + len(context_token_ids)] == context_token_ids:
                is_in = True

                # shape: (w * 1/ratio, h * 1/ratio)
                img_importance = flatten_image_importance(
                    img_th=context_image_map, ratio=ratio
                )
                # shape: ((w * 1/ratio) * (h * 1/ratio), 1)
                img_importance = img_importance.view(-1, 1)
                # shape: ((w * 1/ratio) * (h * 1/ratio), len(context_token_ids))
                img_importance = img_importance.repeat(1, len(context_token_ids))

                ret_tensor[:, i : i + len(context_token_ids)] += img_importance

        if not is_in:
            logger.warning(
                f"Warning ratio {ratio} : tokens {context_token_ids} not found in text"
            )

    # add dimension for the batch
    # shape: (w_r * h_r, len(prompt_token_ids)) -> (1, w_r * h_r, len(prompt_token_ids))
    ret_tensor = ret_tensor.unsqueeze(dim=0)

    return ret_tensor
