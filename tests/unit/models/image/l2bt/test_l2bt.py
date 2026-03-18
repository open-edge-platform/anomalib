import torch
import pytest

from anomalib.models.image.l2bt.torch_model import L2BTModel


class DummyTeacher(torch.nn.Module):
    """Mock teacher to avoid heavy backbone (e.g., DINO)."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 384
        self.patch_size = 16

    def forward(self, x):
        b = x.shape[0]
        n_patches = (x.shape[-1] // self.patch_size) ** 2

        # fake transformer outputs
        middle = torch.randn(b, n_patches, self.embed_dim)
        last = torch.randn(b, n_patches, self.embed_dim)
        return middle, last


@pytest.fixture
def model():
    model = L2BTModel()
    model.teacher = DummyTeacher()  # override heavy teacher
    return model


def test_training_forward(model):
    model.train()
    images = torch.randn(2, 3, 224, 224)

    output = model(images)

    assert "loss" in output
    assert "loss_middle" in output
    assert "loss_last" in output


def test_inference_forward(model):
    model.eval()
    images = torch.randn(2, 3, 224, 224)

    output = model(images)

    assert hasattr(output, "pred_score")
    assert hasattr(output, "anomaly_map")


def test_anomaly_map_shape(model):
    model.eval()
    images = torch.randn(1, 3, 224, 224)

    output = model(images)

    assert output.anomaly_map.shape[-2:] == (224, 224)


def test_invalid_input_shape(model):
    model.eval()
    images = torch.randn(3, 224, 224)  # missing batch dim

    with pytest.raises(ValueError):
        model(images)