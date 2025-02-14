from typing import Dict, Tuple

import pytest
import torch
import torch as th
from diffusers.schedulers import LMSDiscreteScheduler
from PIL import Image

from paint_with_words.helper.aliases import RGB
from paint_with_words.pipelines import PaintWithWordsPipeline
from paint_with_words.pipelines.paint_with_words_pipeline import SeparatedImageContext


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


@pytest.fixture
def gpu_device() -> str:
    return "cuda"


EXAMPLE_SETTING_1 = {
    "color_context": {
        (0, 0, 0): "cat,1.0",
        (255, 255, 255): "dog,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
    "output_image_path": "contents/output_cat_dog.png",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (0, 0, 0): "dog,1.0",
        (255, 255, 255): "cat,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
    "output_image_path": "contents/output_dog_cat.png",
}


EXAMPLE_SETTING_3 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_2.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_image_path": "contents/aurora_2_output.png",
}

EXAMPLE_SETTING_4 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
    "output_image_path": "contents/aurora_1_output.png",
}


ARGNAMES = [
    "color_context",
    "color_map_image_path",
    "input_prompt",
    "output_image_path",
]
ARGVALUES = [
    [EXAMPLE[name] for name in ARGNAMES]
    for EXAMPLE in [
        EXAMPLE_SETTING_1,
        EXAMPLE_SETTING_2,
        EXAMPLE_SETTING_3,
        EXAMPLE_SETTING_4,
    ]
]


@pytest.mark.skipif(
    not th.cuda.is_available(),
    reason="No GPUs available for testing.",
)
@pytest.mark.parametrize(
    argnames=",".join(ARGNAMES),
    argvalues=ARGVALUES,
)
def test_pipeline(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
    output_image_path: str,
    gpu_device: str,
):
    # load pre-trained weight with paint with words pipeline
    pipe = PaintWithWordsPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None  # disable the safety checker
    pipe.to(gpu_device)

    # check the scheduler is LMSDiscreteScheduler
    assert isinstance(pipe.scheduler, LMSDiscreteScheduler), type(pipe.scheduler)

    # generate latents with seed-fixed generator
    generator = torch.manual_seed(0)
    latents = torch.randn((1, 4, 64, 64), generator=generator)

    # load color map image
    color_map_image = Image.open(color_map_image_path)

    # generate image using the pipeline
    with th.autocast("cuda"):
        image = pipe(
            prompt=input_prompt,
            color_context=color_context,
            color_map_image=color_map_image,
            latents=latents,
            num_inference_steps=30,
        ).images[0]

    # save the generated image
    image.save(output_image_path)


@pytest.mark.parametrize(
    argnames=",".join(ARGNAMES),
    argvalues=ARGVALUES,
)
def test_separate_image_context(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
    output_image_path: str,
):
    pipe = PaintWithWordsPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=torch.float16,
    )

    color_map_image = pipe.load_image(color_map_image_path)

    ret_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    for ret in ret_list:
        assert isinstance(ret, SeparatedImageContext)
        assert isinstance(ret.word, str)
        assert isinstance(ret.token_ids, list)
        assert isinstance(ret.color_map_th, torch.Tensor)

        token_ids = pipe.tokenizer(
            ret.word,
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids
        assert ret.token_ids == token_ids


@pytest.mark.parametrize(
    argnames=",".join(ARGNAMES),
    argvalues=ARGVALUES,
)
def test_calculate_attention_maps(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
    output_image_path: str,
    attention_ratios: Tuple[int, ...] = (8, 16, 32, 64),
):
    pipe: PaintWithWordsPipeline = PaintWithWordsPipeline.from_pretrained(  # type: ignore
        model_name,
        revision="fp16",
        torch_dtype=torch.float16,
    )

    color_map_image = pipe.load_image(color_map_image_path)
    w, h = color_map_image.size

    separated_image_context_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    attention_maps = pipe.calculate_attention_maps(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratios=attention_ratios,
    )
    assert len(attention_maps) == len(attention_ratios)
    for attention_map, ratio in zip(attention_maps, attention_ratios):
        expected_attention_map_size = (
            1,
            int((w * 1 / ratio) * (h * 1 / ratio)),
            pipe.tokenizer.model_max_length,
        )
        assert attention_map.size() == expected_attention_map_size
