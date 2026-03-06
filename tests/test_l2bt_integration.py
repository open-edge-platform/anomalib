from __future__ import annotations

from pathlib import Path
import traceback
import torch

from anomalib.models.image.l2bt.torch_model import L2BTModel
from anomalib.models.image.l2bt.lightning_model import L2BT


# ====== CONFIG DA ADATTARE SOLO QUI ======
CHECKPOINT_FOLDER = Path("/home/sharon/L2BT/checkpoints/checkpoints_visa")
CLASS_NAME = "capsules"
LABEL = "final_model"
EPOCHS_NO = 50
BATCH_SIZE = 4

# repo originale: img_size tipico 1036
IMAGE_SIZE = 1036

# se vuoi un test leggero per il wrapper anomalib, usa anche 224
WRAPPER_IMAGE_SIZE = 224
# ========================================


def print_ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def print_fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def expected_checkpoint_paths() -> tuple[Path, Path]:
    base = CHECKPOINT_FOLDER / CLASS_NAME
    forward_path = base / f"forward_net_{LABEL}_{CLASS_NAME}_{EPOCHS_NO}ep_{BATCH_SIZE}bs.pth"
    backward_path = base / f"backward_net_{LABEL}_{CLASS_NAME}_{EPOCHS_NO}ep_{BATCH_SIZE}bs.pth"
    return forward_path, backward_path


def test_checkpoint_files_exist() -> None:
    forward_path, backward_path = expected_checkpoint_paths()

    assert forward_path.exists(), f"Checkpoint forward non trovato: {forward_path}"
    assert backward_path.exists(), f"Checkpoint backward non trovato: {backward_path}"

    print_ok("Checkpoint .pth trovati")


def test_model_instantiation() -> L2BTModel:
    model = L2BTModel(
        checkpoint_folder=str(CHECKPOINT_FOLDER),
        class_name=CLASS_NAME,
        label=LABEL,
        epochs_no=EPOCHS_NO,
        batch_size=BATCH_SIZE,
    ).eval()

    print_ok("L2BTModel istanziato correttamente")
    return model


def test_raw_model_forward(model: L2BTModel) -> None:
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        out = model(x)

    assert hasattr(out, "anomaly_map"), "Output senza anomaly_map"
    assert hasattr(out, "pred_score"), "Output senza pred_score"

    assert out.anomaly_map.ndim == 4, f"anomaly_map ndim atteso 4, trovato {out.anomaly_map.ndim}"
    assert out.pred_score.ndim == 1, f"pred_score ndim atteso 1, trovato {out.pred_score.ndim}"

    assert out.anomaly_map.shape[0] == 1, "Batch anomaly_map errato"
    assert out.anomaly_map.shape[1] == 1, "Canale anomaly_map atteso = 1"
    assert out.anomaly_map.shape[2] == IMAGE_SIZE, f"H attesa {IMAGE_SIZE}, trovata {out.anomaly_map.shape[2]}"
    assert out.anomaly_map.shape[3] == IMAGE_SIZE, f"W attesa {IMAGE_SIZE}, trovata {out.anomaly_map.shape[3]}"
    assert out.pred_score.shape[0] == 1, "Batch pred_score errato"

    assert torch.isfinite(out.anomaly_map).all(), "anomaly_map contiene NaN/Inf"
    assert torch.isfinite(out.pred_score).all(), "pred_score contiene NaN/Inf"

    print_ok(
        f"Forward L2BTModel ok | anomaly_map={tuple(out.anomaly_map.shape)} "
        f"| pred_score={tuple(out.pred_score.shape)}"
    )


def test_lightning_wrapper_predict_step() -> None:
    module = L2BT(
        checkpoint_folder=str(CHECKPOINT_FOLDER),
        class_name=CLASS_NAME,
        label=LABEL,
        epochs_no=EPOCHS_NO,
        batch_size=BATCH_SIZE,
    ).eval()

    batch = {"image": torch.randn(1, 3, WRAPPER_IMAGE_SIZE, WRAPPER_IMAGE_SIZE)}

    with torch.no_grad():
        out = module.predict_step(batch, batch_idx=0)

    assert hasattr(out, "anomaly_map"), "Predict step senza anomaly_map"
    assert hasattr(out, "pred_score"), "Predict step senza pred_score"

    assert out.anomaly_map.ndim == 4, f"anomaly_map ndim atteso 4, trovato {out.anomaly_map.ndim}"
    assert out.pred_score.ndim == 1, f"pred_score ndim atteso 1, trovato {out.pred_score.ndim}"

    assert out.anomaly_map.shape[0] == 1, "Batch anomaly_map errato nel wrapper"
    assert out.anomaly_map.shape[1] == 1, "Canale anomaly_map atteso = 1 nel wrapper"
    assert out.pred_score.shape[0] == 1, "Batch pred_score errato nel wrapper"

    assert torch.isfinite(out.anomaly_map).all(), "Wrapper anomaly_map contiene NaN/Inf"
    assert torch.isfinite(out.pred_score).all(), "Wrapper pred_score contiene NaN/Inf"

    print_ok(
        f"Predict step wrapper ok | anomaly_map={tuple(out.anomaly_map.shape)} "
        f"| pred_score={tuple(out.pred_score.shape)}"
    )


def run_all_tests() -> int:
    tests_failed = 0

    try:
        test_checkpoint_files_exist()
    except Exception as e:
        tests_failed += 1
        print_fail(f"Checkpoint test fallito: {e}")
        traceback.print_exc()

    model = None
    try:
        model = test_model_instantiation()
    except Exception as e:
        tests_failed += 1
        print_fail(f"Istanziazione L2BTModel fallita: {e}")
        traceback.print_exc()

    if model is not None:
        try:
            test_raw_model_forward(model)
        except Exception as e:
            tests_failed += 1
            print_fail(f"Forward L2BTModel fallita: {e}")
            traceback.print_exc()

    try:
        test_lightning_wrapper_predict_step()
    except Exception as e:
        tests_failed += 1
        print_fail(f"Predict step wrapper fallito: {e}")
        traceback.print_exc()

    print("\n==============================")
    if tests_failed == 0:
        print("[SUCCESS] Tutti i test sono passati")
    else:
        print(f"[SUMMARY] Test falliti: {tests_failed}")
    print("==============================")

    return tests_failed


if __name__ == "__main__":
    raise SystemExit(run_all_tests())