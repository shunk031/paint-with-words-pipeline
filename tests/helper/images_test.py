import pytest
from PIL import Image

from paint_with_words.helper.images import get_resize_size, resize_image


@pytest.fixture
def model_name() -> str:
    return "CompVis/stable-diffusion-v1-4"


def test_get_resize_size():

    image = Image.new(mode="RGB", size=(555, 555), color=(0, 0, 0))
    assert get_resize_size(img=image) == (544, 544)

    image = Image.new(mode="RGB", size=(543, 543), color=(0, 0, 0))
    assert get_resize_size(img=image) == (512, 512)

    image = Image.new(mode="RGB", size=(130, 130), color=(0, 0, 0))
    assert get_resize_size(img=image) == (128, 128)

    image = Image.new(mode="RGB", size=(70, 70), color=(0, 0, 0))
    assert get_resize_size(img=image) == (64, 64)


def test_resize_image():

    image = Image.new(mode="RGB", size=(512, 512), color="white")

    resized_image = resize_image(img=image, w=256, h=256)
    assert resized_image.size == (256, 256)

    resized_image = resize_image(img=image, w=128, h=128)
    assert resized_image.size == (128, 128)

    with pytest.raises(AssertionError):
        resized_image = resize_image(
            img=image,
            # Assert if the size is a multiple of 32
            w=129,
            h=129,
        )
