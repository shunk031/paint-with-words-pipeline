import math
from typing import Union

import torch


class WeightFunction(object):
    def __call__(
        self, w: torch.Tensor, sigma: torch.Tensor, qk: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        raise NotImplementedError


class DefaultWeightFunction(WeightFunction):
    def __call__(
        self, w: torch.Tensor, sigma: torch.Tensor, qk: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        return 0.1


class UnconditionedWeightFunction(WeightFunction):
    def __call__(
        self, w: torch.Tensor, sigma: torch.Tensor, qk: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        return 0.0


class PaintWithWordsWeightFunction(WeightFunction):
    def __call__(
        self, w: torch.Tensor, sigma: torch.Tensor, qk: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        return 0.4 * w * math.log(1 + sigma) * qk.max()
