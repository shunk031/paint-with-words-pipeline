from typing import Optional

import torch as th
from diffusers.models.attention_processor import Attention, AttnProcessor

from paint_with_words.weight_functions import WeightFunction


class PaintWithWordsAttnProcessor(AttnProcessor):
    def __init__(self) -> None:
        super().__init__()

    def get_attention_scores(
        self,
        attn: Attention,
        query: th.Tensor,
        key: th.Tensor,
        sigma_t: th.Tensor,
        weight_function: WeightFunction,
        is_cross_attention: bool,
        attention_mask: Optional[th.Tensor] = None,
        **kwargs,
    ) -> th.Tensor:
        dtype = query.dtype
        if attn.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            attention_mask = th.zeros(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )

        attention_scores = th.matmul(query, key.transpose(-1, -2))

        if is_cross_attention:
            attn_img_size = attention_scores.size(dim=-2)
            weight = kwargs[f"w_{attn_img_size}"]

            # shape: (1, attn_img_size, tokenizer.max_model_length)
            cross_attention_weight = weight_function(
                w=weight, sigma=sigma_t, qk=attention_scores
            )
        else:
            cross_attention_weight = 0.0

        # shape: e.g., (8, 4096, 77) * (1, 4096, 77)
        attention_scores = (attention_scores + cross_attention_weight) * attn.scale
        attention_scores += attention_mask

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def __call__(
        self,
        attn: Attention,
        hidden_states: th.Tensor,
        sigma_t: th.Tensor,
        weight_function: WeightFunction,
        encoder_hidden_states: Optional[th.Tensor] = None,
        attention_mask: Optional[th.Tensor] = None,
        temb=None,
        **kwargs,
    ) -> th.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(
            attn=attn,
            query=query,
            key=key,
            sigma_t=sigma_t,
            weight_function=weight_function,
            attention_mask=attention_mask,
            is_cross_attention=is_cross_attention,
            **kwargs,
        )
        hidden_states = th.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
