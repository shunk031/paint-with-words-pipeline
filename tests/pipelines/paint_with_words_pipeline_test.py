from typing import Dict

import pytest
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
        torch_dtype=th.float16,
    )
    pipe.safety_checker = None  # disable the safety checker
    pipe.to(gpu_device)

    # check the scheduler is LMSDiscreteScheduler
    assert isinstance(pipe.scheduler, LMSDiscreteScheduler), type(pipe.scheduler)

    # generate latents with seed-fixed generator
    generator = th.manual_seed(0)
    latents = th.randn((1, 4, 64, 64), generator=generator)

    # load color map image
    color_map_image = Image.open(color_map_image_path)

    # generate image using the pipeline
    with th.autocast("cuda"):
        image = pipe(
            prompts=input_prompt,
            color_contexts=color_context,
            color_map_images=color_map_image,
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
        torch_dtype=th.float16,
    )

    color_map_image = pipe.load_image(color_map_image_path)

    ret_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    for ret in ret_list:
        assert isinstance(ret, SeparatedImageContext)
        assert isinstance(ret.word, str)
        assert isinstance(ret.token_ids, list)
        assert isinstance(ret.color_map_th, th.Tensor)

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
def test_calculate_tokens_image_attention_weight(
    model_name: str,
    color_context: Dict[RGB, str],
    color_map_image_path: str,
    input_prompt: str,
    output_image_path: str,
):
    pipe = PaintWithWordsPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=th.float16,
    )

    color_map_image = pipe.load_image(color_map_image_path)
    w, h = color_map_image.size

    separated_image_context_list = pipe.separate_image_context(
        img=color_map_image, color_context=color_context
    )

    cross_attention_weight_8 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=8,
    )
    assert cross_attention_weight_8.size() == (
        1,
        int((w * 1 / 8) * (h * 1 / 8)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_16 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=16,
    )
    assert cross_attention_weight_16.size() == (
        1,
        int((w * 1 / 16) * (h * 1 / 16)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_32 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=32,
    )
    assert cross_attention_weight_32.size() == (
        1,
        int((w * 1 / 32) * (h * 1 / 32)),
        pipe.tokenizer.model_max_length,
    )

    cross_attention_weight_64 = pipe.calculate_tokens_image_attention_weight(
        input_prompt=input_prompt,
        separated_image_context_list=separated_image_context_list,
        ratio=64,
    )
    assert cross_attention_weight_64.size() == (
        1,
        int((w * 1 / 64) * (h * 1 / 64)),
        pipe.tokenizer.model_max_length,
    )


@pytest.mark.skipif(
    not th.cuda.is_available(),
    reason="No GPUs available for testing.",
)
def test_batch_pipeline(model_name: str, gpu_device: str):
    # load pre-trained weight with paint with words pipeline
    pipe = PaintWithWordsPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=th.float16,
    )
    pipe.safety_checker = None  # disable the safety checker
    pipe.to(gpu_device)

    # check the scheduler is LMSDiscreteScheduler
    assert isinstance(pipe.scheduler, LMSDiscreteScheduler), type(pipe.scheduler)

    # generate latents with seed-fixed generator
    generator = th.manual_seed(0)
    latents = th.randn((1, 4, 64, 64), generator=generator)
    latents = latents.repeat(2, 1, 1, 1)  # shape: (1, 4, 64, 64) -> (2, 4, 64, 64)

    batch_examples = [EXAMPLE_SETTING_1, EXAMPLE_SETTING_3]

    with th.autocast("cuda"):
        pipe_output = pipe(
            prompts=[example["input_prompt"] for example in batch_examples],
            color_contexts=[example["color_context"] for example in batch_examples],
            color_map_images=[
                example["color_map_image_path"] for example in batch_examples
            ],
            latents=latents,
            num_inference_steps=30,
        )
    images = pipe_output.images

    for image, example in zip(images, batch_examples):
        content_dir, image_filename = example["output_image_path"].split("/")  # type: ignore
        image.save(f"{content_dir}/batch_{image_filename}")
