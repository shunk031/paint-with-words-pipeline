from diffusers.models import UNet2DConditionModel

from paint_with_words.models.attention import paint_with_words_forward


def replace_cross_attention(
    unet: UNet2DConditionModel, cross_attention_name: str = "CrossAttention"
) -> None:
    for m in unet.modules():
        if m.__class__.__name__ == cross_attention_name:
            m.__class__.__call__ = paint_with_words_forward
