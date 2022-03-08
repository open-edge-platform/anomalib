"""Test Dataset."""

import numpy as np
import pytest

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.data import get_datamodule
from anomalib.data.mvtec import MVTecDataModule
from anomalib.pre_processing.transforms import Denormalize, ToNumpy
from tests.helpers.dataset import TestDataset, get_dataset_path


@pytest.fixture(autouse=True)
def data_module():
    datamodule = MVTecDataModule(
        root=get_dataset_path(),
        category="leather",
        image_size=(256, 256),
        train_batch_size=1,
        test_batch_size=1,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule


@pytest.fixture(autouse=True)
def data_sample(data_module):
    _, data = next(enumerate(data_module.train_dataloader()))
    return data


class TestMVTecDataModule:
    def test_batch_size(self, data_module):
        """test_mvtec_datamodule [summary]"""
        _, train_data_sample = next(enumerate(data_module.train_dataloader()))
        _, val_data_sample = next(enumerate(data_module.val_dataloader()))
        assert train_data_sample["image"].shape[0] == 1
        assert val_data_sample["image"].shape[0] == 1

    def test_val_and_test_dataloaders_has_mask_and_gt(self, data_module):
        """Validation and Test dataloaders should return filenames, image, mask
        and label."""
        _, val_data = next(enumerate(data_module.val_dataloader()))
        _, test_data = next(enumerate(data_module.test_dataloader()))

        assert sorted(["image_path", "mask_path", "image", "label", "mask"]) == sorted(val_data.keys())
        assert sorted(["image_path", "mask_path", "image", "label", "mask"]) == sorted(test_data.keys())


class TestDenormalize:
    """Test Denormalize Util."""

    def test_denormalize_image_pixel_values(self, data_sample):
        """Test Denormalize denormalizes tensor into [0, 256] range."""
        denormalized_sample = Denormalize().__call__(data_sample["image"].squeeze())
        assert denormalized_sample.min() >= 0 and denormalized_sample.max() <= 256

    def test_denormalize_return_numpy(self, data_sample):
        """Denormalize should return a numpy array."""
        denormalized_sample = Denormalize()(data_sample["image"].squeeze())
        assert isinstance(denormalized_sample, np.ndarray)

    def test_denormalize_channel_order(self, data_sample):
        """Denormalize should return a numpy array of order [HxWxC]"""
        denormalized_sample = Denormalize().__call__(data_sample["image"].squeeze())
        assert len(denormalized_sample.shape) == 3 and denormalized_sample.shape[-1] == 3

    def test_representation(self):
        """Test Denormalize representation should return string
        Denormalize()"""
        assert str(Denormalize()) == "Denormalize()"


class TestToNumpy:
    """Test ToNumpy whether it properly converts tensor into numpy array."""

    def test_to_numpy_image_pixel_values(self, data_sample):
        """Test ToNumpy should return an array whose pixels in the range of [0,
        256]"""
        array = ToNumpy()(data_sample["image"])
        assert array.min() >= 0 and array.max() <= 256

    def test_to_numpy_converts_tensor_to_np_array(self, data_sample):
        """ToNumpy returns a numpy array."""
        array = ToNumpy()(data_sample["image"])
        assert isinstance(array, np.ndarray)

    def test_to_numpy_channel_order(self, data_sample):
        """ToNumpy() should return a numpy array of order [HxWxC]"""
        array = ToNumpy()(data_sample["image"])
        assert len(array.shape) == 3 and array.shape[-1] == 3

    def test_one_channel_images(self, data_sample):
        """One channel tensor should be converted to HxW np array."""
        data = data_sample["image"][:, 0, :, :].unsqueeze(0)
        array = ToNumpy()(data)
        assert len(array.shape) == 2

    def test_representation(self):
        """Test ToNumpy() representation should return string `ToNumpy()`"""
        assert str(ToNumpy()) == "ToNumpy()"


class TestConfigToDataModule:
    """Tests that check if the dataset parameters in the config achieve the desired effect."""

    @pytest.mark.parametrize(
        ["input_size", "effective_image_size"],
        [
            (512, (512, 512)),
            ((245, 276), (245, 276)),
            ((263, 134), (263, 134)),
            ((267, 267), (267, 267)),
        ],
    )
    @TestDataset(num_train=20, num_test=10)
    def test_image_size(self, input_size, effective_image_size, category="shapes", path=""):
        """Test if the image size parameter works as expected."""
        model_name = "stfpm"
        configurable_parameters = get_configurable_parameters(model_name)
        configurable_parameters.dataset.path = path
        configurable_parameters.dataset.category = category
        configurable_parameters.dataset.image_size = input_size
        configurable_parameters = update_input_size_config(configurable_parameters)

        data_module = get_datamodule(configurable_parameters)
        data_module.setup()
        assert iter(data_module.train_dataloader()).__next__()["image"].shape[-2:] == effective_image_size
