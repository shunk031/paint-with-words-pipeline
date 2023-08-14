import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.schedulers import (
    KarrasDiffusionSchedulers,
    LMSDiscreteScheduler,
)
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from paint_with_words.helper.aliases import RGB
from paint_with_words.helper.images import (
    flatten_image_importance,
    get_resize_size,
    resize_image,
)
from paint_with_words.models.attention_processor import PaintWithWordsAttnProcessor
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


class PaintWithWordsPipeline(StableDiffusionPipeline, TextualInversionLoaderMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
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

    def register_cross_attention_processor(self) -> None:
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            attn_procs[name] = PaintWithWordsAttnProcessor()
        self.unet.set_attn_processor(attn_procs)

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

    def _find_context_token_index(
        self,
        prompt_token_ids: List[int],
        context_token_ids: List[int],
    ) -> Optional[int]:
        for idx in range(len(prompt_token_ids)):
            cond = (
                prompt_token_ids[idx : idx + len(context_token_ids)]
                == context_token_ids
            )
            if cond:
                return idx
        return None

    def _calculate_attention_maps(
        self,
        prompt_token_ids: List[int],
        separated_image_context_list: List[SeparatedImageContext],
        color_map_w: int,
        color_map_h: int,
        ratio: int,
    ) -> th.Tensor:
        w_r, h_r = color_map_w // ratio, color_map_h // ratio

        attention_maps_size = (w_r * h_r, len(prompt_token_ids))
        attention_maps = th.zeros(
            attention_maps_size, dtype=self.text_encoder.dtype, device=self.device
        )

        for separated_image_context in separated_image_context_list:
            context_token_ids = separated_image_context.token_ids
            context_image_map = separated_image_context.color_map_th

            idx = self._find_context_token_index(
                prompt_token_ids=prompt_token_ids, context_token_ids=context_token_ids
            )
            if idx is not None:
                # shape: (w, h) -> (w * 1/ratio, h * 1/ratio)
                img_importance = flatten_image_importance(
                    img_th=context_image_map, ratio=ratio
                )
                # shape: ((w * 1/ratio) * (h * 1/ratio), 1)
                img_importance = img_importance.view(-1, 1)
                # shape: ((w * 1/ratio) * (h * 1/ratio), len(context_token_ids))
                img_importance = img_importance.repeat(1, len(context_token_ids))

                attention_maps[:, idx : idx + len(context_token_ids)] += img_importance
            else:
                logger.warning(
                    f"Warning ratio {ratio} : tokens {context_token_ids} not found in text"
                )

        # add dimension for the batch
        # shape: (w_r * h_r, len(prompt_token_ids)) -> (1, w_r * h_r, len(prompt_token_ids))
        attention_maps = attention_maps.unsqueeze(dim=0)
        assert attention_maps.size() == (1, w_r * h_r, len(prompt_token_ids))

        return attention_maps

    def calculate_attention_maps(
        self,
        input_prompt: str,
        separated_image_context_list: List[SeparatedImageContext],
        ratios: Tuple[int, ...],
    ) -> List[th.Tensor]:
        assert (
            len(set(c.color_map_th.size() for c in separated_image_context_list)) == 1
        )
        map_w, map_h = separated_image_context_list[0].color_map_th.size()

        prompt_token_ids = self.tokenizer(
            input_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        attention_maps = [
            self._calculate_attention_maps(
                prompt_token_ids=prompt_token_ids,
                separated_image_context_list=separated_image_context_list,
                color_map_w=map_w,
                color_map_h=map_h,
                ratio=ratio,
            )
            for ratio in ratios
        ]
        assert len(attention_maps) == len(ratios)
        return attention_maps

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
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[th.Generator, List[th.Generator]]] = None,
        latents: Optional[th.FloatTensor] = None,
        prompt_embeds: Optional[th.FloatTensor] = None,
        negative_prompt_embeds: Optional[th.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, th.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attention_ratios: Tuple[int, ...] = (8, 16, 32, 64),
    ) -> StableDiffusionPipelineOutput:
        cross_attention_kwargs = cross_attention_kwargs or {}
        cross_attention_kwargs["weight_function"] = weight_function

        # 0. Default height and width to unet and resize the color map image
        color_map_image = self.load_image(image=color_map_image)
        width, height = get_resize_size(img=color_map_image)
        color_map_image = resize_image(img=color_map_image, w=width, h=height)

        separated_image_context_list = self.separate_image_context(
            img=color_map_image, color_context=color_context
        )
        attention_maps = self.calculate_attention_maps(
            input_prompt=prompt,
            separated_image_context_list=separated_image_context_list,
            ratios=attention_ratios,
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # Ensure classifier free guidance is performed and
        # the batch size of the text embedding is 2 (conditional + unconditional)
        assert do_classifier_free_guidance and prompt_embeds.size(dim=0) == 2
        uncond_embeddings, cond_embeddings = prompt_embeds.chunk(2)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.register_cross_attention_processor()

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                assert isinstance(self.scheduler, LMSDiscreteScheduler)
                step_index = (self.scheduler.timesteps == t).nonzero().item()
                sigma_t = self.scheduler.sigmas[step_index]
                cross_attention_kwargs["sigma_t"] = sigma_t

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                expected_latent_size = (
                    1,
                    num_channels_latents,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                )
                assert latent_model_input.size() == expected_latent_size

                assert len(attention_ratios) == len(attention_maps)
                for i, ratio in enumerate(attention_ratios):
                    attn_img_size = height * width // (ratio * ratio)
                    cross_attention_kwargs[f"w_{attn_img_size}"] = attention_maps[i]

                # predict the noise residual
                noise_pred_text = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                for i, ratio in enumerate(attention_ratios):
                    attn_img_size = height * width // (ratio * ratio)
                    cross_attention_kwargs[f"w_{attn_img_size}"] = 0.0

                noise_pred_uncond = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
