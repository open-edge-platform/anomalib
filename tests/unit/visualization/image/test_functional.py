from PIL import Image
import numpy as np
from anomalib.visualization.image.functional import add_bounding_boxes_to_image
def test_add_bounding_boxes_to_image():
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    anomaly_map = np.random.rand(256, 256).astype(np.float32)

    result = add_bounding_boxes_to_image(image, anomaly_map)

    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)


def test_add_bounding_boxes_with_score():
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    anomaly_map = np.random.rand(256, 256).astype(np.float32)

    result = add_bounding_boxes_to_image(
        image,
        anomaly_map,
        show_score=True,
    )

    assert isinstance(result, Image.Image)
    assert result.size == (256, 256)


def test_add_bounding_boxes_with_empty_map():
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    anomaly_map = np.zeros((256, 256), dtype=np.float32)

    result = add_bounding_boxes_to_image(
        image,
        anomaly_map,
        show_score=True,
    )

    assert isinstance(result, Image.Image)