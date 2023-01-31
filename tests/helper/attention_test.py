import copy

import pytest
from diffusers.models import UNet2DConditionModel

from paint_with_words.helper.attention import replace_cross_attention


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


def test_replace_cross_attention(
    model_name: str, cross_attention_name: str = "CrossAttention"
):
    unet_original = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", revision="fp16"
    )

    unet_proposed = copy.deepcopy(unet_original)
    # unet_proposed = UNet2DConditionModel.from_pretrained(
    #     model_name, subfolder="unet", revision="fp16"
    # )
    unet_proposed = replace_cross_attention(unet=unet_proposed)

    for m_orig, m_prop in zip(unet_original.modules(), unet_proposed.modules()):
        cond1 = m_orig.__class__.__name__ == cross_attention_name
        cond2 = m_prop.__class__.__name__ == cross_attention_name

        if cond1 and cond2:
            breakpoint()
            assert m_orig.__class__.__call__.__name__ == "_call_impl"
            assert m_prop.__class__.__call__.__name__ == "paint_with_words_forward"
