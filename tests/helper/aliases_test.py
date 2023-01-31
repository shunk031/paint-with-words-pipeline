from typing import Dict, Tuple, Union

import torch as th

from paint_with_words.helper.aliases import (
    RGB,
    PaintWithWordsHiddenStates,
    SeparatedImageContext,
)
from paint_with_words.weight_functions import WeightFunction


def test_rgb():
    assert RGB == Tuple[int, int, int]


def test_paint_with_words_hidden_states():
    assert PaintWithWordsHiddenStates == Dict[str, Union[th.Tensor, WeightFunction]]


def test_separated_image_context():

    separated_image_context = SeparatedImageContext(
        word="cat",
        token_ids=[2368],
        color_map_th=th.zeros((512, 512)),
    )
    assert isinstance(separated_image_context, SeparatedImageContext)
