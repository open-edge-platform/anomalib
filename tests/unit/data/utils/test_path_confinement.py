# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for dataset metadata path confinement."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from anomalib.data import Visa
from anomalib.data.datasets.image.datumaro import make_datumaro_dataset
from anomalib.data.datasets.image.realiad import make_realiad_dataset
from anomalib.data.datasets.image.tabular import make_tabular_dataset
from anomalib.data.utils import LabelName, Split
from anomalib.data.utils.image import save_image
from anomalib.data.utils.path import resolve_path_under_root


class TestVisaPathConfinement:
    """VisA split CSV must not read files outside the dataset root."""

    @staticmethod
    def test_apply_cls1_split_rejects_out_of_root_image_path() -> None:
        """Malicious ``image_path`` in ``1cls.csv`` must not copy host files."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            root = base / "visa"
            (root / "candle").mkdir(parents=True)
            (root / "split_csv").mkdir()

            secret = base / "SECRET_host_file.txt"
            secret.write_text("TOP-SECRET host contents", encoding="utf-8")

            with (root / "split_csv" / "1cls.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["category", "split", "label", "image_path", "mask_path"])
                writer.writerow(["candle", "train", "normal", "../SECRET_host_file.txt", ""])

            datamodule = Visa(root=root, category="candle")
            with pytest.raises(ValueError, match="Access denied"):
                datamodule.prepare_data()

            leaked = root / "visa_pytorch" / "candle" / "train" / "good" / "SECRET_host_file.txt"
            assert not leaked.is_file()


class TestDatumaroPathConfinement:
    """Datumaro JSON image paths must stay under the dataset root."""

    @staticmethod
    def test_rejects_path_traversal_in_image_path() -> None:
        """Reject ``items[].image.path`` that escapes the dataset root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "datumaro"
            (root / "annotations").mkdir(parents=True)
            (root / "images" / "default").mkdir(parents=True)
            secret = Path(tmp_dir) / "secret.txt"
            secret.write_text("secret", encoding="utf-8")

            annotations = {
                "categories": {"label": {"labels": [{"name": "Normal"}, {"name": "Anomalous"}]}},
                "items": [
                    {
                        "image": {"path": "../../../secret.txt"},
                        "annotations": [{"label_id": 0}],
                    },
                ],
            }
            (root / "annotations" / "default.json").write_text(json.dumps(annotations), encoding="utf-8")

            with pytest.raises(ValueError, match="Access denied"):
                make_datumaro_dataset(root)


class TestTabularPathConfinement:
    """Tabular paths must resolve under ``root`` when ``root`` is set."""

    @staticmethod
    def test_rejects_relative_escape_when_root_set() -> None:
        """Reject ``../`` image paths when a root is provided."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            root.mkdir()
            samples = {
                "image_path": ["../outside.png"],
                "label_index": [LabelName.NORMAL],
                "split": [Split.TRAIN],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_tabular_dataset(samples=samples, root=root)

    @staticmethod
    def test_rejects_absolute_path_outside_root() -> None:
        """Reject absolute image paths outside the provided root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "data"
            root.mkdir()
            outside = Path(tmp_dir) / "outside.png"
            outside.write_bytes(b"x")
            samples = {
                "image_path": [str(outside)],
                "label_index": [LabelName.NORMAL],
                "split": [Split.TRAIN],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_tabular_dataset(samples=samples, root=root)


class TestRealIADPathConfinement:
    """RealIAD JSON paths must stay under the category root."""

    @staticmethod
    def test_rejects_path_traversal_in_metadata() -> None:
        """Reject metadata ``image_path`` values that escape the root."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "realiad"
            root.mkdir()
            metadata = {
                "train": [
                    {
                        "image_path": "../../secret.jpg",
                        "anomaly_class": "OK",
                    },
                ],
                "test": [],
            }
            with pytest.raises(ValueError, match="Access denied"):
                make_realiad_dataset(root=root, split=Split.TRAIN, metadata=metadata)


class TestKaputtStylePathConfinement:
    """Kaputt-style relative joins must use ``resolve_path_under_root``."""

    @staticmethod
    def test_resolve_under_query_subdir_rejects_escape() -> None:
        """Parquet-relative paths must not escape the query image subdirectory."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "kaputt"
            image_subdir = root / "query-image"
            image_subdir.mkdir(parents=True)
            (Path(tmp_dir) / "secret.jpg").write_bytes(b"x")
            with pytest.raises(ValueError, match="Access denied"):
                resolve_path_under_root(image_subdir, "../secret.jpg", should_exist=False)


class TestSaveImagePathConfinement:
    """``save_image`` must not write outside the provided root."""

    @staticmethod
    def test_rejects_parent_directory_escape() -> None:
        """Reject filenames that escape ``root`` via ``..``."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "results"
            root.mkdir()
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="Access denied"):
                save_image(filename="../escaped.png", image=image, root=root)
            assert not (Path(tmp_dir) / "escaped.png").exists()
