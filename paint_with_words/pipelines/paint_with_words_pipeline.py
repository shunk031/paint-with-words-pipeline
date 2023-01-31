import logging
from typing import Callable, Dict, List, Optional, Union

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
from PIL.Image import Image as PilImage
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from paint_with_words.helper.aliases import RGB, ColorContext, SeparatedImageContext
from paint_with_words.helper.attention import replace_cross_attention
from paint_with_words.helper.images import (
    calculate_tokens_image_attention_weight,
    get_resize_size,
    load_image,
    resize_image,
    separate_image_context,
)
from paint_with_words.weight_functions import (
    PaintWithWordsWeightFunction,
    UnconditionedWeightFunction,
    WeightFunction,
)

logger = logging.getLogger(__name__)


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
        replace_cross_attention(
            unet=self.unet,
            cross_attention_name=cross_attention_name,
        )

    def separate_image_context(
        self, img: PilImage, color_context: ColorContext
    ) -> List[SeparatedImageContext]:
        return separate_image_context(
            tokenizer=self.tokenizer,
            img=img,
            color_context=color_context,
            device=self.device,
        )

    def calculate_tokens_image_attention_weight(
        self,
        input_prompt: str,
        separated_image_context_list: List[SeparatedImageContext],
        ratio: int,
    ) -> th.Tensor:

        return calculate_tokens_image_attention_weight(
            tokenizer=self.tokenizer,
            input_prompt=input_prompt,
            separated_image_context_list=separated_image_context_list,
            ratio=ratio,
            device=self.device,
        )

    @th.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        color_contexts: Union[ColorContext, List[ColorContext]],
        color_map_images: Union[PilImage, List[PilImage]],
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

        if not isinstance(prompts, list):
            prompts = [prompts]
        if not isinstance(color_contexts, list):
            color_contexts = [color_contexts]
        if not isinstance(color_map_images, list):
            color_map_images = [color_map_images]

        assert guidance_scale > 1.0, guidance_scale

        # 0. Default height and width to unet and resize the color map image
        color_map_images = [load_image(image=image) for image in color_map_images]
        sizes = [get_resize_size(img=img) for img in color_map_images]
        color_map_images = [
            resize_image(img=img, w=w, h=h)
            for img, (w, h) in zip(color_map_images, sizes)
        ]

        separated_image_context_list = self.separate_image_context(
            img=color_map_images, color_context=color_contexts
        )
        cross_attention_weight_8 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompts,
            separated_image_context_list=separated_image_context_list,
            ratio=8,
        )

        cross_attention_weight_16 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompts,
            separated_image_context_list=separated_image_context_list,
            ratio=16,
        )

        cross_attention_weight_32 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompts,
            separated_image_context_list=separated_image_context_list,
            ratio=32,
        )

        cross_attention_weight_64 = self.calculate_tokens_image_attention_weight(
            input_prompt=prompts,
            separated_image_context_list=separated_image_context_list,
            ratio=64,
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompts, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompts, str) else len(prompts)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompts,
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
