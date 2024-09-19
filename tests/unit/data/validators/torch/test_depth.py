"""Test Torch Depth Validators."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image, Mask

from anomalib.data.validators.torch.depth import DepthBatchValidator, DepthValidator


class TestDepthValidator:
    """Test DepthValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = DepthValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid depth image."""
        image = torch.rand(3, 224, 224)
        validated_image = self.validator.validate_image(image)
        assert isinstance(validated_image, Image)
        assert validated_image.shape == (3, 224, 224)
        assert validated_image.dtype == torch.float32

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of a depth image with invalid type."""
        with pytest.raises(TypeError, match="Image must be a torch.Tensor"):
            self.validator.validate_image(np.random.default_rng().random((3, 224, 224)))

    def test_validate_image_invalid_dimensions(self) -> None:
        """Test validation of a depth image with invalid dimensions."""
        with pytest.raises(ValueError, match="Image must have shape"):
            self.validator.validate_image(torch.rand(224, 224))

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of a depth image with invalid number of channels."""
        with pytest.raises(ValueError, match="Image must have 3 channels"):
            self.validator.validate_image(torch.rand(1, 224, 224))

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of a valid ground truth label."""
        label = torch.tensor(1)
        validated_label = self.validator.validate_gt_label(label)
        assert isinstance(validated_label, torch.Tensor)
        assert validated_label.dtype == torch.bool
        assert validated_label.item() is True

    def test_validate_gt_label_none(self) -> None:
        """Test validation of a None ground truth label."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of a ground truth label with invalid type."""
        with pytest.raises(TypeError, match="Ground truth label must be an integer or a torch.Tensor"):
            self.validator.validate_gt_label("1")

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of a valid ground truth mask."""
        mask = torch.randint(0, 2, (1, 224, 224))
        validated_mask = self.validator.validate_gt_mask(mask)
        assert isinstance(validated_mask, Mask)
        assert validated_mask.shape == (224, 224)
        assert validated_mask.dtype == torch.bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of a None ground truth mask."""
        assert self.validator.validate_gt_mask(None) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of a ground truth mask with invalid type."""
        with pytest.raises(TypeError, match="Mask must be a torch.Tensor"):
            self.validator.validate_gt_mask(np.random.default_rng().integers(0, 2, (224, 224)))

    def test_validate_gt_mask_invalid_shape(self) -> None:
        """Test validation of a ground truth mask with invalid shape."""
        with pytest.raises(ValueError, match="Mask must have 1 channel, got 2."):
            self.validator.validate_gt_mask(torch.randint(0, 2, (2, 224, 224)))

    def test_validate_anomaly_map_valid(self) -> None:
        """Test validation of a valid anomaly map."""
        anomaly_map = torch.rand(1, 224, 224)
        validated_map = self.validator.validate_anomaly_map(anomaly_map)
        assert isinstance(validated_map, Mask)
        assert validated_map.shape == (224, 224)
        assert validated_map.dtype == torch.float32

    def test_validate_anomaly_map_none(self) -> None:
        """Test validation of a None anomaly map."""
        assert self.validator.validate_anomaly_map(None) is None

    def test_validate_anomaly_map_invalid_type(self) -> None:
        """Test validation of an anomaly map with invalid type."""
        with pytest.raises(TypeError, match="Anomaly map must be a torch.Tensor"):
            self.validator.validate_anomaly_map(np.random.default_rng().random((224, 224)))

    def test_validate_anomaly_map_invalid_shape(self) -> None:
        """Test validation of an anomaly map with invalid shape."""
        with pytest.raises(ValueError, match="Anomaly map with 3 dimensions must have 1 channel, got 2."):
            self.validator.validate_anomaly_map(torch.rand(2, 224, 224))

    def test_validate_pred_score_valid(self) -> None:
        """Test validation of a valid prediction score."""
        score = torch.tensor(0.8)
        validated_score = self.validator.validate_pred_score(score)
        assert isinstance(validated_score, torch.Tensor)
        assert validated_score.dtype == torch.float32
        assert validated_score.item() == pytest.approx(0.8)

    def test_validate_pred_score_none(self) -> None:
        """Test validation of a None prediction score."""
        assert self.validator.validate_pred_score(None) is None

    def test_validate_pred_score_invalid_shape(self) -> None:
        """Test validation of a prediction score with invalid shape."""
        with pytest.raises(ValueError, match="Predicted score must be a scalar"):
            self.validator.validate_pred_score(torch.tensor([0.8, 0.9]))


class TestDepthBatchValidator:  # noqa: PLR0904
    """Test DepthBatchValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = DepthBatchValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid depth image batch."""
        image_batch = torch.rand(32, 3, 224, 224)
        validated_batch = self.validator.validate_image(image_batch)
        assert isinstance(validated_batch, Image)
        assert validated_batch.shape == (32, 3, 224, 224)
        assert validated_batch.dtype == torch.float32

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of a depth image batch with invalid type."""
        with pytest.raises(TypeError, match="Image must be a torch.Tensor"):
            self.validator.validate_image(np.random.default_rng().random((32, 3, 224, 224)))

    def test_validate_image_invalid_dimensions(self) -> None:
        """Test validation of a depth image batch with invalid dimensions."""
        with pytest.raises(ValueError, match="Image must have shape"):
            self.validator.validate_image(torch.rand(32, 224, 224))

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of a depth image batch with invalid number of channels."""
        with pytest.raises(ValueError, match="Image must have 3 channels"):
            self.validator.validate_image(torch.rand(32, 1, 224, 224))

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of valid ground truth labels."""
        labels = torch.tensor([0, 1, 1, 0])
        validated_labels = self.validator.validate_gt_label(labels, batch_size=4)
        assert isinstance(validated_labels, torch.Tensor)
        assert validated_labels.dtype == torch.bool
        assert torch.equal(validated_labels, torch.tensor([False, True, True, False]))

    def test_validate_gt_label_none(self) -> None:
        """Test validation of None ground truth labels."""
        assert self.validator.validate_gt_label(None, batch_size=4) is None

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of ground truth labels with invalid type."""
        with pytest.raises(ValueError, match="too many dimensions 'str'"):
            self.validator.validate_gt_label(["0", "1"], batch_size=2)

    def test_validate_gt_label_invalid_dimensions(self) -> None:
        """Test validation of ground truth labels with invalid dimensions."""
        with pytest.raises(ValueError, match="Ground truth label must be a 1-dimensional vector"):
            self.validator.validate_gt_label(torch.tensor([[0, 1], [1, 0]]), batch_size=2)

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of valid ground truth masks."""
        masks = torch.randint(0, 2, (4, 224, 224))
        validated_masks = self.validator.validate_gt_mask(masks, batch_size=4)
        assert isinstance(validated_masks, Mask)
        assert validated_masks.shape == (4, 224, 224)
        assert validated_masks.dtype == torch.bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of None ground truth masks."""
        assert self.validator.validate_gt_mask(None, batch_size=4) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of ground truth masks with invalid type."""
        with pytest.raises(TypeError, match="Ground truth mask must be a torch.Tensor"):
            self.validator.validate_gt_mask([torch.zeros(224, 224)], batch_size=1)

    def test_validate_gt_mask_invalid_dimensions(self) -> None:
        """Test validation of ground truth masks with invalid dimensions."""
        with pytest.raises(ValueError, match="Ground truth mask must have 1 channel, got 2."):
            self.validator.validate_gt_mask(torch.zeros(4, 2, 224, 224), batch_size=4)

    def test_validate_anomaly_map_valid(self) -> None:
        """Test validation of a valid anomaly map batch."""
        anomaly_map = torch.rand(4, 224, 224)
        validated_map = self.validator.validate_anomaly_map(anomaly_map, batch_size=4)
        assert isinstance(validated_map, Mask)
        assert validated_map.shape == (4, 224, 224)
        assert validated_map.dtype == torch.float32

    def test_validate_anomaly_map_none(self) -> None:
        """Test validation of a None anomaly map batch."""
        assert self.validator.validate_anomaly_map(None, batch_size=4) is None

    def test_validate_anomaly_map_invalid_shape(self) -> None:
        """Test validation of an anomaly map batch with invalid shape."""
        with pytest.raises(ValueError, match="Anomaly map must have 1 channel, got 2."):
            self.validator.validate_anomaly_map(torch.rand(4, 2, 224, 224), batch_size=4)

    def test_validate_pred_score_valid(self) -> None:
        """Test validation of valid prediction scores."""
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        validated_scores = self.validator.validate_pred_score(scores, anomaly_map=None)
        assert torch.equal(validated_scores, scores)

    def test_validate_pred_score_none_with_anomaly_map(self) -> None:
        """Test validation of None prediction scores with anomaly map."""
        anomaly_map = torch.rand(4, 224, 224)
        computed_scores = self.validator.validate_pred_score(None, anomaly_map)
        assert computed_scores.shape == (4,)

    def test_validate_pred_label_valid(self) -> None:
        """Test validation of valid prediction labels."""
        labels = torch.tensor([[1], [0], [1], [1]])
        validated_labels = self.validator.validate_pred_label(labels)
        assert torch.equal(validated_labels, torch.tensor([[True], [False], [True], [True]]))

    def test_validate_pred_label_none(self) -> None:
        """Test validation of None prediction labels."""
        assert self.validator.validate_pred_label(None) is None

    def test_validate_pred_label_invalid_type(self) -> None:
        """Test validation of prediction labels with invalid type."""
        with pytest.raises(TypeError, match="Predicted label must be a torch.Tensor"):
            self.validator.validate_pred_label([1, 0, 1, 1])

    def test_validate_pred_label_invalid_shape(self) -> None:
        """Test validation of prediction labels with invalid shape."""
        with pytest.raises(ValueError, match="Predicted label must be 1-dimensional or 2-dimensional"):
            self.validator.validate_pred_label(torch.tensor([[[1]], [[0]], [[1]], [[1]]]))
