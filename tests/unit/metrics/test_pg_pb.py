# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test PGn and PBn metrics."""

import pytest
import torch

from anomalib.metrics.pg_pb import _PBn as PBn
from anomalib.metrics.pg_pb import _PGn as PGn


def test_pg_basic() -> None:
    """Test PGn metric with simple binary classification."""
    metric = PGn(fnr=0.2)
    preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    metric.update(preds, labels)
    result = metric.compute()
    assert result == torch.tensor(1.0)
    assert metric.name == "PG20"


def test_pb_basic() -> None:
    """Test PBn metric with simple binary classification."""
    metric = PBn(fpr=0.2)
    preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    metric.update(preds, labels)
    result = metric.compute()
    assert result == torch.tensor(1.0)
    assert metric.name == "PB20"


def test_pg_invalid_fnr() -> None:
    """Test PGn metric raises ValueError for invalid fnr."""
    with pytest.raises(ValueError, match="False negative rate must be in the range between 0 and 1"):
        PGn(fnr=-0.1)
    with pytest.raises(ValueError, match="False negative rate must be in the range between 0 and 1"):
        PGn(fnr=1.1)


def test_pb_invalid_fpr() -> None:
    """Test PBn metric raises ValueError for invalid fpr."""
    with pytest.raises(ValueError, match="False positive rate must be in the range between 0 and 1"):
        PBn(fpr=-0.1)
    with pytest.raises(ValueError, match="False positive rate must be in the range between 0 and 1"):
        PBn(fpr=1.1)


def test_pg_no_negatives() -> None:
    """Test PGn metric raises ValueError if no negative samples."""
    metric = PGn(fnr=0.1)
    preds = torch.tensor([0.5, 0.7])
    labels = torch.tensor([1, 1])
    metric.update(preds, labels)
    with pytest.raises(ValueError, match="No negative samples found"):
        metric.compute()


def test_pb_no_positives() -> None:
    """Test PBn metric raises ValueError if no positive samples."""
    metric = PBn(fpr=0.1)
    preds = torch.tensor([0.2, 0.3])
    labels = torch.tensor([0, 0])
    metric.update(preds, labels)
    with pytest.raises(ValueError, match="No positive samples found"):
        metric.compute()
