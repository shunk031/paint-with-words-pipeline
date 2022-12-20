from typing import Tuple

import torch as th
import torch.nn.functional as F
from PIL.Image import Image as PilImage
from PIL.Image import Resampling


def get_resize_size(img: PilImage) -> Tuple[int, int]:
    w, h = img.size
    w, h = map(lambda x: x - x % 32, (w, h))
    return w, h


def resize_image(img: PilImage, w: int, h: int) -> PilImage:
    assert w % 32 == 0 and h % 32 == 0, (w, h)
    return img.resize((w, h), resample=Resampling.LANCZOS)


def flatten_image_importance(img_th: th.Tensor, ratio: int) -> th.Tensor:

    # shape: (h, w) -> (1, 1, h, w)
    img_th = img_th.view(1, 1, *img_th.size())

    # shape: (1, 1, w * scale_factor, h * scale_factor)
    scale_factor = 1 / ratio
    ret = F.interpolate(
        img_th,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=True,
    )
    # shape: (w * scale_factor, h * scale_factor)
    ret = ret.squeeze()

    return ret
