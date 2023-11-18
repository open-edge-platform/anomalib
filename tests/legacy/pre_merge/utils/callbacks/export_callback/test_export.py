import os
import tempfile
from unittest.mock import patch

import lightning.pytorch as pl
import pytest
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from tests.legacy.helpers.dataset import get_dataset_path
from tests.legacy.pre_merge.utils.callbacks.export_callback.dummy_lightning_model import DummyLightningModule

from anomalib.callbacks.export import ExportCallback
from anomalib.data.image.mvtec import MVTec
from anomalib.data.utils import random_split
from anomalib.deploy import ExportMode
from anomalib.engine import Engine


# TODO: This is temporarily here. Move it to conftest.py in integration tests.
@pytest.fixture()
def dummy_datamodule() -> MVTec:
    datamodule = MVTec(root=get_dataset_path("MVTec"), category="bottle", image_size=32)
    datamodule.setup()

    _, train_subset = random_split(dataset=datamodule.train_data, split_ratio=0.1, label_aware=True)
    _, test_subset = random_split(dataset=datamodule.test_data, split_ratio=0.1, label_aware=True)

    datamodule.train_data = train_subset
    datamodule.test_data = test_subset

    return datamodule


@pytest.mark.parametrize(
    "export_mode",
    [ExportMode.OPENVINO, ExportMode.ONNX],
)
def test_export_model_callback(dummy_datamodule: MVTec, export_mode):
    """Tests if an optimized model is created."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = DummyLightningModule()
        model.callbacks = [
            ExportCallback(
                input_size=(32, 32),
                dirpath=os.path.join(tmp_dir),
                filename="model",
                export_mode=export_mode,
            ),
            EarlyStopping(monitor="loss"),
        ]
        engine = Engine(
            accelerator="gpu",
            devices=1,
            callbacks=model.callbacks,
            logger=False,
            enable_checkpointing=False,
            max_epochs=1,
            default_root_dir=tmp_dir,
        )
        engine.fit(model, datamodule=dummy_datamodule)

        if export_mode == ExportMode.OPENVINO:
            assert os.path.exists(
                os.path.join(tmp_dir, "weights/openvino/model.bin")
            ), "Failed to generate OpenVINO model"
        elif export_mode == ExportMode.ONNX:
            assert os.path.exists(os.path.join(tmp_dir, "weights/onnx/model.onnx")), "Failed to generate ONNX model"
        else:
            raise ValueError(f"Unknown export_mode {export_mode}. Supported modes: onnx or openvino.")
