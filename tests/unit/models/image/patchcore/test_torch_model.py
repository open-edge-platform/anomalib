# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore torch model."""

import pytest
import torch

from anomalib.data.utils.tiler import Tiler
from anomalib.models.image.patchcore import torch_model
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


@pytest.fixture(params=[False, True], ids=["untiled", "tiled"])
def model(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> PatchcoreModel:
    """PatchCore with a random memory bank and a chunk size smaller than one image's patches."""
    # A 64x64 image gives 8x8 = 64 patches, so eager inference searches the memory bank in several chunks.
    monkeypatch.setattr(torch_model, "DEFAULT_CHUNK_SIZE", 16)
    monkeypatch.setattr(torch_model, "_DISTANCE_BLOCK_BYTES", 2048)
    torch.manual_seed(0)
    model = PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False, num_neighbors=3)
    model.memory_bank = torch.randn(32, 384)  # resnet18 layer2 + layer3 channels
    if request.param:
        model.tiler = Tiler(tile_size=32, stride=32)
    return model.eval()


@pytest.mark.parametrize("n_neighbors", [1, 5])
def test_large_bank_search(model: PatchcoreModel, n_neighbors: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound the distance blocks while retaining indices into the full bank, including ties."""
    budget = 1024
    monkeypatch.setattr(torch_model, "_DISTANCE_BLOCK_BYTES", budget)
    bank = model.memory_bank = torch.randint(-4, 5, (41, 16)).float()
    bank[-1] = bank[0]
    queries = torch.cat((bank[:1], torch.randint(-4, 5, (34, bank.size(1))).float()))
    distances = model.euclidean_dist(queries, bank)
    expected = distances.min(1) if n_neighbors == 1 else distances.topk(n_neighbors, largest=False)

    block_shapes = []
    matmul = torch.matmul

    def record_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        block_shapes.append((x.size(0), y.size(1)))
        return matmul(x, y)

    monkeypatch.setattr(torch, "matmul", record_block)
    actual, indices = model.nearest_neighbors(queries, n_neighbors)
    torch.testing.assert_close(actual, expected.values)
    if n_neighbors == 1:
        torch.testing.assert_close(indices, expected.indices)
        assert indices[0] == 0
    else:
        torch.testing.assert_close(actual, distances.gather(1, indices))
    assert sum(rows for rows, _ in block_shapes) == len(queries)
    assert all(rows * cols * bank.element_size() <= budget for rows, cols in block_shapes)


def test_export_keeps_batch_size_dynamic(model: PatchcoreModel) -> None:
    """Export searches all patches at once, so any batch size runs and matches chunked eager inference."""
    batch_size = torch.export.Dim("batch_size")
    exported = torch.export.export(model, (torch.rand(2, 3, 64, 64),), dynamic_shapes=({0: batch_size},)).module()

    for size in (1, 3):
        images = torch.rand(size, 3, 64, 64)
        actual, expected = exported(images), model(images)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
