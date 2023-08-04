"""Test for ensemble visualizer"""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch import Tensor

from anomalib.models.ensemble.post_processing import EnsembleVisualization
from tests.helpers.dataset import get_dataset_path

mock_result = {
    "image_path": [Path(get_dataset_path()) / "bottle/test/broken_large/000.png"],
    "image": torch.rand((1, 3, 100, 100)),
    "mask": torch.zeros((1, 100, 100)),
    "anomaly_maps": torch.ones((1, 100, 100)),
    "label": torch.Tensor([0]),
    "pred_scores": torch.Tensor([0.5]),
    "pred_labels": torch.Tensor([0]),
    "pred_masks": torch.zeros((1, 100, 100)),
    "pred_boxes": [torch.rand(1, 4)],
    "box_labels": [torch.tensor([0.5])],
}


@pytest.mark.parametrize("task", ["segmentation", "classification", "detection"])
def test_save_image(task, get_config):
    config = get_config
    with TemporaryDirectory() as temp_dir:
        config.project.path = temp_dir
        config.dataset.task = task
        visualization = EnsembleVisualization(config)
        visualization.process(copy.deepcopy(mock_result))

        assert (Path(temp_dir) / "images/broken_large/000.png").exists()


def test_data_unchanged(get_config):
    config = get_config
    with TemporaryDirectory() as temp_dir:
        config.project.path = temp_dir
        visualization = EnsembleVisualization(config)
        vis_output = visualization.process(copy.deepcopy(mock_result))

        for name, values in vis_output.items():
            if isinstance(values, Tensor):
                assert values.equal(mock_result[name]), f"{name} changed"
            elif isinstance(values, list) and isinstance(values[0], Tensor):
                assert values[0].equal(mock_result[name][0]), f"{name} changed"
            else:
                assert values == mock_result[name], f"{name} changed"
