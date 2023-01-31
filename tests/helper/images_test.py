from typing import Dict

import pytest
import torch as th
from PIL import Image
from transformers import CLIPTokenizer

from paint_with_words.helper.aliases import RGB, SeparatedImageContext
from paint_with_words.helper.images import (
    calculate_tokens_image_attention_weight,
    get_resize_size,
    load_image,
    resize_image,
    separate_image_context,
)


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


def test_get_resize_size():

    image = Image.new(mode="RGB", size=(555, 555), color=(0, 0, 0))
    assert get_resize_size(img=image) == (544, 544)

    image = Image.new(mode="RGB", size=(543, 543), color=(0, 0, 0))
    assert get_resize_size(img=image) == (512, 512)

    image = Image.new(mode="RGB", size=(130, 130), color=(0, 0, 0))
    assert get_resize_size(img=image) == (128, 128)

    image = Image.new(mode="RGB", size=(70, 70), color=(0, 0, 0))
    assert get_resize_size(img=image) == (64, 64)


def test_resize_image():

    image = Image.new(mode="RGB", size=(512, 512), color="white")

    resized_image = resize_image(img=image, w=256, h=256)
    assert resized_image.size == (256, 256)

    resized_image = resize_image(img=image, w=128, h=128)
    assert resized_image.size == (128, 128)

    with pytest.raises(AssertionError):
        resized_image = resize_image(
            img=image,
            # Assert if the size is a multiple of 32
            w=129,
            h=129,
        )


def test_separate_image_context(
    model_name: str, color_context: Dict[RGB, str], color_map_image_path: str
):

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    color_map_image = load_image(color_map_image_path)

    ret_list = separate_image_context(
        tokenizer=tokenizer,
        img=color_map_image,
        color_context=color_context,
        device="cpu",
    )

    for ret in ret_list:
        assert isinstance(ret, SeparatedImageContext)
        assert isinstance(ret.word, str)
        assert isinstance(ret.token_ids, list)
        assert isinstance(ret.color_map_th, th.Tensor)

        token_ids = tokenizer(
            ret.word,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids
        assert ret.token_ids == token_ids


def test_calculate_tokens_image_attention_weight(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
):

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

    color_map_image = load_image(color_map_image_path)
    w, h = color_map_image.size

    separated_image_context_list = separate_image_context(
        tokenizer=tokenizer,
        img=color_map_image,
        color_context=color_context,
        device="cpu",
    )

    cross_attention_weight_8 = calculate_tokens_image_attention_weight(
        tokenizer=tokenizer,
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=8,
        device="cpu",
    )
    assert cross_attention_weight_8.size() == (
        int((w * 1 / 8) * (h * 1 / 8)),
        tokenizer.model_max_length,
    )

    cross_attention_weight_16 = calculate_tokens_image_attention_weight(
        tokenizer=tokenizer,
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=16,
        device="cpu",
    )
    assert cross_attention_weight_16.size() == (
        int((w * 1 / 16) * (h * 1 / 16)),
        tokenizer.model_max_length,
    )

    cross_attention_weight_32 = calculate_tokens_image_attention_weight(
        tokenizer=tokenizer,
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=32,
        device="cpu",
    )
    assert cross_attention_weight_32.size() == (
        int((w * 1 / 32) * (h * 1 / 32)),
        tokenizer.model_max_length,
    )

    cross_attention_weight_64 = calculate_tokens_image_attention_weight(
        tokenizer=tokenizer,
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=64,
        device="cpu",
    )
    assert cross_attention_weight_64.size() == (
        int((w * 1 / 64) * (h * 1 / 64)),
        tokenizer.model_max_length,
    )
