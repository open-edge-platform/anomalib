# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AUPRO metric."""

import numpy as np
import pytest
import torch

from anomalib.metrics.aupro import _AUPRO as AUPRO

from .aupro_reference import calculate_au_pro


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate fixtures for the tests."""
    labels = [
        torch.tensor(
            [
                [
                    [0, 0, 0, 1, 0, 0, 0],
                ]
                * 400,
            ],
        ),
        torch.tensor(
            [
                [
                    [0, 1, 0, 1, 0, 1, 0],
                ]
                * 400,
            ],
        ),
    ]
    preds = torch.arange(2800) / 2800.0
    preds = preds.view(1, 1, 400, 7)

    preds = [preds, preds]

    fpr_limit = [1 / 3, 1 / 3]
    expected_aupro = [torch.tensor(1 / 6), torch.tensor(1 / 6)]

    # Also test that per-region aupros are averaged
    labels.append(torch.cat(labels))
    preds.append(torch.cat(preds))
    fpr_limit.append(float(np.mean(fpr_limit)))
    expected_aupro.append(torch.tensor(np.mean(expected_aupro)))

    if metafunc.function is test_aupro:
        vals = list(zip(labels, preds, fpr_limit, expected_aupro, strict=True))
        metafunc.parametrize(argnames=("labels", "preds", "fpr_limit", "expected_aupro"), argvalues=vals)


def test_aupro(labels: torch.Tensor, preds: torch.Tensor, fpr_limit: float, expected_aupro: torch.Tensor) -> None:
    """Test if the AUPRO metric is computed correctly."""
    aupro = AUPRO(fpr_limit=fpr_limit)
    aupro.update(preds, labels)
    computed_aupro = aupro.compute()

    tmp_labels = [label.squeeze().numpy() for label in labels]
    tmp_preds = [pred.squeeze().numpy() for pred in preds]
    ref_aupro = torch.tensor(calculate_au_pro(tmp_labels, tmp_preds, integration_limit=fpr_limit)[0], dtype=torch.float)

    tolerance = 0.001
    assert torch.allclose(computed_aupro, expected_aupro, atol=tolerance)
    assert torch.allclose(computed_aupro, ref_aupro, atol=tolerance)


@pytest.mark.parametrize("fpr_limit", [0.05, 0.3, 1.0])
def test_aupro_matches_reference_on_random_maps(fpr_limit: float) -> None:
    """AUPRO is normalized by ``fpr_limit`` and matches the MVTec reference implementation.

    Regression test for https://github.com/open-edge-platform/anomalib/issues/3718.
    """
    generator = torch.Generator().manual_seed(0)
    labels = torch.zeros(4, 32, 32, dtype=torch.long)
    labels[0, 4:12, 4:12] = 1
    labels[1, 20:30, 2:6] = 1
    labels[2, 10:14, 10:30] = 1
    labels[2, 25:28, 25:28] = 1
    preds = torch.rand(4, 1, 32, 32, generator=generator)
    preds[:, 0][labels > 0] += 0.3

    aupro = AUPRO(fpr_limit=fpr_limit)
    aupro.update(preds, labels)
    computed_aupro = aupro.compute()

    fpr, _ = aupro._compute()  # noqa: SLF001
    assert torch.isclose(fpr[-1], torch.tensor(fpr_limit))
    assert 0.0 <= computed_aupro <= 1.0

    ref_aupro, _ = calculate_au_pro(
        [label.numpy() for label in labels],
        [pred.squeeze(0).numpy() for pred in preds],
        integration_limit=fpr_limit,
        num_thresholds=2000,
    )
    assert torch.isclose(computed_aupro, torch.tensor(ref_aupro, dtype=torch.float), atol=0.01)
