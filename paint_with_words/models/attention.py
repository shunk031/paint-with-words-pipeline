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

    # batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    # dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

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

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = th.matmul(attention_probs, value)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states
