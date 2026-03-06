import torch
from anomalib.models.image.l2bt.lightning_model import L2BT

def test_predict_step_shapes():
    model = L2BT(
        checkpoint_folder="./checkpoints/checkpoints_visa",
        class_name="capsules",
        label="final_model",
        epochs_no=50,
        batch_size=4,
        layers=(7, 11),
    )

    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "image_path": ["a.jpg", "b.jpg"],
        "mask_path": ["a.png", "b.png"],
    }

    out = model.predict_step(batch, 0)

    assert out.pred_score.shape == (2,)
    assert out.anomaly_map.shape == (2, 1, 224, 224)
