# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for image visualization functions."""

import numpy as np
import pytest
from PIL import Image

from anomalib.visualization.image.functional import np_to_pil_image


@pytest.fixture
def rng() -> np.random.Generator:
    """Create a random number generator."""
    return np.random.default_rng(42)


class TestNumpyToPILImage:
    """Test suite for np_to_pil_image function."""

    @staticmethod
    def test_grayscale_conversion(rng: np.random.Generator) -> None:
        """Test converting grayscale numpy array to PIL Image."""
        # Test with 2D array
        array = rng.random((100, 100))
        image = np_to_pil_image(array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (100, 100)
        assert np.array(image).max() <= 255
        assert np.array(image).min() >= 0

        # Test with boolean array
        bool_array = np.array([[True, False], [False, True]])
        image = np_to_pil_image(bool_array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (2, 2)
        assert np.array(image).max() == 255
        assert np.array(image).min() == 0

        # Test with uint8 array
        uint8_array = np.array([[0, 128], [128, 255]], dtype=np.uint8)
        image = np_to_pil_image(uint8_array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (2, 2)
        assert np.array(image).max() == 255
        assert np.array(image).min() == 0

    @staticmethod
    def test_rgb_conversion(rng: np.random.Generator) -> None:
        """Test converting RGB numpy array to PIL Image."""
        # Test with 3D array (H, W, 3)
        array = rng.random((100, 100, 3))
        image = np_to_pil_image(array)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (100, 100)
        assert np.array(image).max() <= 255
        assert np.array(image).min() >= 0

        # Test with boolean RGB array
        bool_array = np.array([[[True, False, True], [False, True, False]]], dtype=bool)
        image = np_to_pil_image(bool_array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert np.array(image).max() == 255
        assert np.array(image).min() == 0

        # Test with uint8 RGB array
        uint8_array = np.array([[[0, 128, 255], [255, 128, 0]]], dtype=np.uint8)
        image = np_to_pil_image(uint8_array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert np.array(image).max() == 255
        assert np.array(image).min() == 0

    @staticmethod
    def test_singleton_dimensions(rng: np.random.Generator) -> None:
        """Test handling of arrays with singleton dimensions."""
        # Test with (1, H, W) array
        array = rng.random((1, 100, 100))
        image = np_to_pil_image(array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (100, 100)
        assert np.array(image).max() <= 255
        assert np.array(image).min() >= 0

        # Test with (H, W, 1) array
        array = rng.random((100, 100, 1))
        image = np_to_pil_image(array)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size == (100, 100)
        assert np.array(image).max() <= 255
        assert np.array(image).min() >= 0

        # Test with (1, H, W, 3) array
        array = rng.random((1, 100, 100, 3))
        image = np_to_pil_image(array)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (100, 100)
        assert np.array(image).max() <= 255
        assert np.array(image).min() >= 0

    @staticmethod
    def test_invalid_dimensions(rng: np.random.Generator) -> None:
        """Test handling of arrays with invalid dimensions."""
        # Test with 1D array
        array = rng.random(100)
        with pytest.raises(ValueError, match=r"Expected 2D array"):
            np_to_pil_image(array)

        # Test with 3D array that's not RGB
        array = rng.random((100, 100, 4))
        with pytest.raises(ValueError, match=r"Expected 2D array"):
            np_to_pil_image(array)
