from typing import Dict, Tuple, Union

import torch

from paint_with_words.weight_functions import WeightFunction

RGB = Tuple[int, int, int]

PaintWithWordsHiddenStates = Dict[str, Union[torch.Tensor, WeightFunction]]
