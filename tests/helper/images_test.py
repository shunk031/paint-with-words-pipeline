EXAMPLE_SETTING_1 = {
    "color_context": {
        (0, 0, 0): "cat,1.0",
        (255, 255, 255): "dog,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
}

EXAMPLE_SETTING_2 = {
    "color_context": {
        (0, 0, 0): "dog,1.0",
        (255, 255, 255): "cat,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_image_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
}


EXAMPLE_SETTING_3 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_2.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
}

EXAMPLE_SETTING_4 = {
    "color_context": {
        (7, 9, 182): "aurora,0.5",
        (136, 178, 92): "full moon,1.5",
        (51, 193, 217): "mountains,0.4",
        (61, 163, 35): "a half-frozen lake,0.3",
        (89, 102, 255): "boat,2.0",
    },
    "color_map_image_path": "contents/aurora_1.png",
    "input_prompt": "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed.",
}


ARGNAMES = [
    "color_context",
    "color_map_image_path",
    "input_prompt",
]
ARGVALUES = [
    [EXAMPLE[name] for name in ARGNAMES]
    for EXAMPLE in [
        EXAMPLE_SETTING_1,
        EXAMPLE_SETTING_2,
        EXAMPLE_SETTING_3,
        EXAMPLE_SETTING_4,
    ]
]


def test_get_resize_size(color_map_image_path: str):

    raise NotImplementedError


def test_resize_image(color_map_image_path: str):
    raise NotImplementedError


def test_flatten_image_importance(color_map_image_path: str):
    raise NotImplementedError
