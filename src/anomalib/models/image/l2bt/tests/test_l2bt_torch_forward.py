import torch
from anomalib.models.image.l2bt.torch_model import L2BTModel

def test_l2btmodel_forward_shapes():
    m = L2BTModel(
        checkpoint_folder="./checkpoints/checkpoints_visa",
        class_name="capsules",
        label="final_model",
        epochs_no=50,
        batch_size=4,
        layers=(7, 11),
    )
    x = torch.randn(2, 3, 224, 224)  # B=2
    out = m(x)
    assert out.pred_score.shape == (2,)
    assert out.anomaly_map.shape == (2, 1, 224, 224)
