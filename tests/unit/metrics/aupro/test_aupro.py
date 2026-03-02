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
