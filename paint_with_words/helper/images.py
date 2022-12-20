import torch as th
import torch.nn.functional as F


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
