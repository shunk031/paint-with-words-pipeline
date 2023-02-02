from typing import Optional

import torch as th
from diffusers.models.attention import CrossAttention

from paint_with_words.helper.aliases import PaintWithWordsHiddenStates
from paint_with_words.weight_functions import WeightFunction


def paint_with_words_forward(
    self: CrossAttention,
    hidden_states: th.Tensor,
    context: Optional[PaintWithWordsHiddenStates] = None,
    mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    is_dict_format = True
    if context is not None:
        try:
            context_tensor = context["context_tensor"]
        except KeyError:
            context_tensor = context
            is_dict_format = False

    else:
        context_tensor = hidden_states

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    # dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    # shape: (batch_size * self.heads, 64 * 64, 77)
    attention_scores = th.matmul(query, key.transpose(-1, -2))

    attention_size_of_img = attention_scores.shape[-2]
    if context is not None:
        if is_dict_format:
            weight_function: WeightFunction = context["weight_function"]  # type: ignore
            cross_attention_weight = weight_function(
                w=context[f"cross_attention_weight_{attention_size_of_img}"],  # type: ignore
                sigma=context["sigma"],  # type: ignore
                qk=attention_scores,
            )
        else:
            cross_attention_weight = 0.0
    else:
        cross_attention_weight = 0.0

    if not isinstance(cross_attention_weight, float):
        # shape: (batch_size, 64 * 64, 77) -> (batch_size * self.heads, 64 * 64, 77)
        #
        # example:
        # >>> x = torch.arange(20).reshape(2, 10)
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        #
        # >>> x.repeat_interleave(2, dim=0)
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
        #         [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #         [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        #
        cross_attention_weight = cross_attention_weight.repeat_interleave(
            self.heads, dim=0
        )

        # Example:
        # shape (attention_scores): (16, 4096, 77)
        #   scores1: (8, 4096, 77), scores2: (8, 4096, 77)
        # shape (cross_attention_weights): (2, 4096, 77)
        #   weights1: (1, 4096, 77), weights2: (1, 4096, 77)
        #
        # We want to calculate the following:
        #   scores1 + weights1
        #   scores2 + weights2

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = th.matmul(attention_probs, value)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states
