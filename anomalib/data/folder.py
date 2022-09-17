"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
from pandas.core.frame import DataFrame
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.base import AnomalibDataModule
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)

logger = logging.getLogger(__name__)


def _check_and_convert_path(path: Union[str, Path]) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (Union[str, Path]): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: Union[str, Path], path_type: str, extensions: Optional[Tuple[str, ...]] = None
) -> Tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (Union[str, Path]): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = [f for f in path.glob(r"**/*") if f.suffix in extensions and not f.is_dir()]
    if len(filenames) == 0:
        raise RuntimeError(f"Found 0 {path_type} images in {path}")

    labels = [path_type] * len(filenames)

    return filenames, labels


@DATAMODULE_REGISTRY
class Folder(AnomalibDataModule):
    """Folder Lightning Data Module."""

    def __init__(
        self,
        root: Union[str, Path],
        normal_dir: str = "normal",
        abnormal_dir: str = "abnormal",
        task: str = "classification",
        normal_test_dir: Optional[Union[Path, str]] = None,
        mask_dir: Optional[Union[Path, str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        split_ratio: float = 0.2,
        seed: Optional[int] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Folder Dataset PL Datamodule.

        Args:
            root (Union[str, Path]): Path to the root folder containing normal and abnormal dirs.
            normal_dir (str, optional): Name of the directory containing normal images.
                Defaults to "normal".
            abnormal_dir (str, optional): Name of the directory containing abnormal images.
                Defaults to "abnormal".
            task (str, optional): Task type. Could be either classification or segmentation.
                Defaults to "classification".
            normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                normal images for the test dataset. Defaults to None.
            mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                the mask annotations. Defaults to None.
            extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
                directory. Defaults to None.
            split_ratio (float, optional): Ratio to split normal training images and add to the
                test set in case test set doesn't contain any normal images.
                Defaults to 0.2.
            seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
            image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
                Defaults to None.
            train_batch_size (int, optional): Training batch size. Defaults to 32.
            test_batch_size (int, optional): Test batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 8.
            transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
                during training.
                Defaults to None.
            transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
                during validation.
                Defaults to None.
            create_validation_set (bool, optional):Boolean to create a validation set from the test set.
                Those wanting to create a validation set could set this flag to ``True``.

        Examples:
            Assume that we use Folder Dataset for the MVTec/bottle/broken_large category. We would do:
            >>> from anomalib.data import Folder
            >>> datamodule = Folder(
            ...     root="./datasets/MVTec/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     image_size=256
            ... )
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data["image"].shape
            torch.Size([16, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> test_data.keys()
            dict_keys(['image'])

            We could also create a Folder DataModule for datasets containing mask annotations.
            The dataset expects that mask annotation filenames must be same as the original filename.
            To this end, we modified mask filenames in MVTec AD bottle category.
            Now we could try folder data module using the mvtec bottle broken large category
            >>> datamodule = Folder(
            ...     root="./datasets/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256
            ... )

            >>> i , train_data = next(enumerate(datamodule.train_dataloader()))
            >>> train_data.keys()
            dict_keys(['image'])
            >>> train_data["image"].shape
            torch.Size([16, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([24, 3, 256, 256]) torch.Size([24, 256, 256])

            By default, Folder Data Module does not create a validation set. If a validation set
            is needed it could be set as follows:

            >>> datamodule = Folder(
            ...     root="./datasets/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256,
            ...     create_validation_set=True,
            ... )

            >>> i, val_data = next(enumerate(datamodule.val_dataloader()))
            >>> val_data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(val_data["image"].shape, val_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

        """
        if seed is None and normal_test_dir is None:
            raise ValueError(
                "Both seed and normal_test_dir cannot be None."
                " When seed is not set, images from the normal directory are split between training and test dir."
                " This will lead to inconsistency between runs."
            )

        if task == "segmentation" and mask_dir is None:
            warnings.warn(
                "Segmentation task is requested, but mask directory is not provided. "
                "Classification is to be chosen if mask directory is not provided."
            )
            self.task = "classification"
        else:
            self.task = task

        self.root = _check_and_convert_path(root)
        self.normal_dir = self.root / normal_dir
        self.abnormal_dir = self.root / abnormal_dir if abnormal_dir is not None else None
        self.normal_test_dir = normal_test_dir
        if normal_test_dir:
            self.normal_test_dir = self.root / normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions
        self.split_ratio = split_ratio

        self.create_validation_set = create_validation_set
        self.seed = seed

        super().__init__(
            task=task,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            transform_config_train=transform_config_train,
            transform_config_val=transform_config_val,
            image_size=image_size,
            create_validation_set=create_validation_set,
        )

    def _create_samples(self):
        """Create the dataframe with samples for the Folder dataset.

        The files are expected to follow the structure:
            path/to/dataset/normal_folder_name/normal_image_name.png
            path/to/dataset/abnormal_folder_name/abnormal_image_name.png


        This function creates a dataframe to store the parsed information based on the following format:
        |---|-------------------|--------|-------------|------------------|-------|
        |   | image_path        | label  | label_index | mask_path        | split |
        |---|-------------------|--------|-------------|------------------|-------|
        | 0 | path/to/image.png | normal | 0           | path/to/mask.png | train |
        |---|-------------------|--------|-------------|------------------|-------|

        Returns:
            DataFrame: an output dataframe containing the samples of the dataset.
        """

        filenames = []
        labels = []
        dirs = {"normal": self.normal_dir, "abnormal": self.abnormal_dir}

        if self.normal_test_dir:
            dirs = {**dirs, **{"normal_test": self.normal_test_dir}}

        for dir_type, path in dirs.items():
            if path is not None:
                filename, label = _prepare_files_labels(path, dir_type, self.extensions)
                filenames += filename
                labels += label

        samples = DataFrame({"image_path": filenames, "label": labels})

        # Create label index for normal (0) and abnormal (1) images.
        samples.loc[(samples.label == "normal") | (samples.label == "normal_test"), "label_index"] = 0
        samples.loc[(samples.label == "abnormal"), "label_index"] = 1
        samples.label_index = samples.label_index.astype(int)

        # If a path to mask is provided, add it to the sample dataframe.
        if self.mask_dir is not None:
            self.mask_dir = _check_and_convert_path(self.mask_dir)
            samples["mask_path"] = ""
            for index, row in samples.iterrows():
                if row.label_index == 1:
                    samples.loc[index, "mask_path"] = str(self.mask_dir / row.image_path.name)

        # Ensure the pathlib objects are converted to str.
        # This is because torch dataloader doesn't like pathlib.
        samples = samples.astype({"image_path": "str"})

        # Create train/test split.
        # By default, all the normal samples are assigned as train.
        #   and all the abnormal samples are test.
        samples.loc[(samples.label == "normal"), "split"] = "train"
        samples.loc[(samples.label == "abnormal") | (samples.label == "normal_test"), "split"] = "test"

        if not self.normal_test_dir:
            samples = split_normal_images_in_train_set(
                samples=samples, split_ratio=self.split_ratio, seed=self.seed, normal_label="normal"
            )

        # If `create_validation_set` is set to True, the test set is split into half.
        if self.create_validation_set:
            samples = create_validation_set_from_test_set(samples, seed=self.seed, normal_label="normal")

        return samples
