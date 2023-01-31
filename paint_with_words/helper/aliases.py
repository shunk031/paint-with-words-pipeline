from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch as th

from paint_with_words.weight_functions import WeightFunction

RGB = Tuple[int, int, int]

PaintWithWordsHiddenStates = Dict[str, Union[torch.Tensor, WeightFunction]]
ColorContext = Dict[RGB, str]


@dataclass
class SeparatedImageContext(object):
    word: str
    token_ids: List[int]
    color_map_th: th.Tensor
