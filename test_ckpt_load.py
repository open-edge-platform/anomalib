# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Audit weights-only checkpoint loading for public Anomalib models."""

from pathlib import Path

from lightning.pytorch import Trainer
from torch.serialization import get_unsafe_globals_in_checkpoint
from tqdm import tqdm

from anomalib.engine import Engine
from anomalib.engine.plugins import AnomalibCheckpointIO
from anomalib.models import __all__ as model_names
from anomalib.models import get_model
from anomalib.models.components.base import AnomalibModule

SUCCESS_FILE = Path("successful_models.txt")
UNSUCCESS_FILE = Path("unsuccessful_models.txt")


def image_checkpoint_path(model_name: str) -> Path:
    """Return the expected trained checkpoint path for a model."""
    return Path("results") / model_name / "MVTecAD" / "bottle" / "latest" / "weights" / "lightning" / "model.ckpt"


def any_checkpoint(model_name: str) -> Path | None:
    """Prefer latest MVTecAD bottle ckpt; otherwise any .ckpt under results/<name>."""
    latest = image_checkpoint_path(model_name)
    if latest.exists():
        return latest
    root = Path("results") / model_name
    if not root.exists():
        return None
    matches = sorted(p for p in root.rglob("*.ckpt") if p.is_file())
    # Prefer non-fresh trained ckpts when present.
    trained = [p for p in matches if p.name != "fresh_safe_globals.ckpt"]
    if trained:
        return trained[-1]
    return matches[-1] if matches else None


def save_fresh_checkpoint(model_name: str, model: AnomalibModule) -> Path:
    """Save a minimal Lightning checkpoint without running a full fit."""
    trainer = Trainer(max_epochs=1, logger=False, barebones=True)
    trainer.strategy.connect(model)
    ckpt = Path("results") / model_name / "fresh_safe_globals.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt)
    return ckpt


def verify_weights_only_load(model: AnomalibModule, ckpt: Path) -> None:
    """Load via CheckpointIO + load_from_checkpoint under weights_only=True."""
    extras = list(type(model).checkpoint_safe_globals())
    io = AnomalibCheckpointIO(extra_safe_globals=extras)
    loaded = io.load_checkpoint(ckpt, weights_only=True)
    assert "state_dict" in loaded or "hyper_parameters" in loaded
    type(model).load_from_checkpoint(ckpt, weights_only=True)


def write_success_file(names: list[str]) -> None:
    """Write successfully verified model names."""
    SUCCESS_FILE.write_text("\n".join(names) + "\n", encoding="utf-8")


def write_unsuccess_file(failed: dict[str, str], notes: dict[str, str]) -> None:
    """Write hard failures and existing-ckpt issues with reasons."""
    lines: list[str] = []
    if failed:
        lines.append("# Hard failures (weights_only path did not succeed)")
        for name, reason in failed.items():
            # Keep each entry as a single line for easy grepping.
            compact = " ".join(reason.split())
            lines.append(f"{name}: {compact}")
    if notes:
        if lines:
            lines.append("")
        lines.append("# Existing checkpoint issues (fresh allowlist path succeeded)")
        for name, reason in notes.items():
            compact = " ".join(reason.split())
            lines.append(f"{name}: {compact}")
    if not lines:
        lines.append("# No unsuccessful models")
    UNSUCCESS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    models = list(model_names)
    successful: list[str] = []
    failed: dict[str, str] = {}
    notes: dict[str, str] = {}

    for model_name in tqdm(models):
        try:
            model = get_model(model_name)
            engine = Engine(max_epochs=1, logger=False)
            engine._setup_trainer(model)  # noqa: SLF001
            extras = engine.trainer.strategy.checkpoint_io.extra_safe_globals

            ckpt = any_checkpoint(model_name)
            source = "existing"
            if ckpt is None:
                ckpt = save_fresh_checkpoint(model_name, model)
                source = "fresh"

            unsafe = sorted(get_unsafe_globals_in_checkpoint(ckpt))
            print(f"\n{model_name} [{source}] extras={extras} unsafe={unsafe}")
            try:
                verify_weights_only_load(model, ckpt)
            except BaseException as load_err:
                # Trained ckpts may contain numpy LR-scheduler state or stale enums.
                # Fall back to a fresh save to validate the model-extra allowlist path.
                if source == "existing":
                    print(f"{model_name}: existing failed ({type(load_err).__name__}); retrying fresh")
                    notes[model_name] = f"existing ckpt failed: {type(load_err).__name__}: {load_err}"
                    ckpt = save_fresh_checkpoint(model_name, model)
                    unsafe = sorted(get_unsafe_globals_in_checkpoint(ckpt))
                    print(f"{model_name} [fresh] unsafe={unsafe}")
                    verify_weights_only_load(model, ckpt)
                else:
                    raise

            print(f"{model_name}: OK")
            successful.append(model_name)
        except Exception as e:  # noqa: BLE001, PERF203 - audit every model independently
            msg = f"{type(e).__name__}: {e}"
            print(f"{model_name}: FAIL {msg}")
            failed[model_name] = msg

    write_success_file(successful)
    write_unsuccess_file(failed, notes)

    print("\n=== SUMMARY ===")
    print(f"OK ({len(successful)}/{len(models)}): {successful}")
    print(f"Wrote {SUCCESS_FILE} and {UNSUCCESS_FILE}")
    if notes:
        print("NOTES (existing ckpt issues; fresh allowlist path OK):")
        for name, note in notes.items():
            print(f"  {name}: {note[:300]}")
    print(f"FAIL ({len(failed)}):")
    for name, err in failed.items():
        print(f"  {name}: {err[:400]}")
