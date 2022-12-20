import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch as th
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
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
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from paint_with_words.helper.aliases import RGB
from paint_with_words.helper.images import (
    flatten_image_importance,
    get_resize_size,
    resize_image,
)
from paint_with_words.models.attention import paint_with_words_forward
from paint_with_words.weight_functions import (
    PaintWithWordsWeightFunction,
    UnconditionedWeightFunction,
    WeightFunction,
)

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
        config = self.scheduler.config  # type: ignore
        self.scheduler = LMSDiscreteScheduler.from_config(config)

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

    def calculate_tokens_image_attention_weight(
        self,
        input_prompt: str,
        separated_image_context_list: List[SeparatedImageContext],
        ratio: int,
    ) -> th.Tensor:

        prompt_token_ids = self.tokenizer(
            input_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        w, h = separated_image_context_list[0].color_map_th.shape
        w_r, h_r = w // ratio, h // ratio

        ret_tensor = th.zeros(
            (w_r * h_r, len(prompt_token_ids)), dtype=th.float32, device=self.device
        )

        for separated_image_context in separated_image_context_list:
            is_in = False
            context_token_ids = separated_image_context.token_ids
            context_image_map = separated_image_context.color_map_th

            for i, token_id in enumerate(prompt_token_ids):
                if (
                    prompt_token_ids[i : i + len(context_token_ids)]
                    == context_token_ids
                ):
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

        return ret_tensor

    def load_image(self, image: Union[str, os.PathLike, PilImage]) -> PilImage:
        if isinstance(image, str) or isinstance(image, os.PathLike):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @th.no_grad()
    def __call__(
        self,
        prompt: str,
        color_context: Dict[RGB, str],
        color_map_image: PilImage,
        weight_function: WeightFunction = PaintWithWordsWeightFunction(),
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0,
        generator: Optional[th.Generator] = None,
        latents: Optional[th.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, th.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ) -> StableDiffusionPipelineOutput:

        assert isinstance(prompt, str), type(prompt)
        assert guidance_scale > 1.0, guidance_scale

        # 0. Default height and width to unet and resize the color map image
        color_map_image = self.load_image(image=color_map_image)
        width, height = get_resize_size(img=color_map_image)
        color_map_image = resize_image(img=color_map_image, w=width, h=height)

        separated_image_context_list = self.separate_image_context(
            img=color_map_image, color_context=color_context
        )
        cross_attention_weight_8 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompt,
            separated_image_context_list=separated_image_context_list,
            ratio=8,
        )

        cross_attention_weight_16 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompt,
            separated_image_context_list=separated_image_context_list,
            ratio=16,
        )

        cross_attention_weight_32 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompt,
            separated_image_context_list=separated_image_context_list,
            ratio=32,
        )

        cross_attention_weight_64 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompt,
            separated_image_context_list=separated_image_context_list,
            ratio=64,
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        # Ensure classifier free guidance is performed and
        # the batch size of the text embedding is 2 (conditional + unconditional)
        assert do_classifier_free_guidance and text_embeddings.size(dim=0) == 2
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                assert isinstance(self.scheduler, LMSDiscreteScheduler)
                step_index = (self.scheduler.timesteps == t).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                assert latent_model_input.size() == (
                    1,
                    num_channels_latents,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                )

                # predict the noise residual
                noise_pred_text = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states={
                        "context_tensor": cond_embeddings,
                        f"cross_attention_weight_{height * width // (8*8)}": cross_attention_weight_8,
                        f"cross_attention_weight_{height * width // (16*16)}": cross_attention_weight_16,
                        f"cross_attention_weight_{height * width // (32*32)}": cross_attention_weight_32,
                        f"cross_attention_weight_{height * width // (64*64)}": cross_attention_weight_64,
                        "sigma": sigma,
                        "weight_function": weight_function,
                    },
                ).sample

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                noise_pred_uncond = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states={
                        "context_tensor": uncond_embeddings,
                        f"cross_attention_weight_{height * width // (8*8)}": 0.0,
                        f"cross_attention_weight_{height * width // (16*16)}": 0.0,
                        f"cross_attention_weight_{height * width // (32*32)}": 0.0,
                        f"cross_attention_weight_{height * width // (64*64)}": 0.0,
                        "sigma": sigma,
                        "weight_function": UnconditionedWeightFunction(),
                    },
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:  # type: ignore
                        callback(i, t, latents)  # type: ignore

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, text_embeddings.dtype
        )

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
