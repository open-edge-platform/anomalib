# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model learning rate scheduler intervals and progressions."""

from types import SimpleNamespace

import pytest
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, MultiStepLR, StepLR

from anomalib.models import Dinomaly, EfficientAd, InpFormer, Uflow, UniNet
from anomalib.models.image.dinomaly import lightning_model as dinomaly_module
from anomalib.models.image.dinomaly.components.optimizer import WarmCosineScheduler
from anomalib.models.image.inp_former import lightning_model as inp_former_module
from anomalib.models.image.uflow import lightning_model as uflow_module
from anomalib.models.image.uninet import lightning_model as uninet_module


class _StubDinomalyModel(nn.Module):
    """Minimal module exposing bottleneck and decoder parameters without downloading weights."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.bottleneck = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)


class _StubInpFormerModel(nn.Module):
    """Minimal module exposing INP-Former trainable components without downloading weights."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.bottleneck = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)
        self.aggregation = nn.Linear(1, 1)
        self.prototype_token = nn.Linear(1, 1)


class _StubUniNetModel(nn.Module):
    """Minimal module exposing UniNet parameter groups without downloading weights."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.student = nn.Linear(1, 1)
        self.bottleneck = nn.Linear(1, 1)
        self.dfs = nn.Linear(1, 1)
        self.teachers = SimpleNamespace(target_teacher=nn.Linear(1, 1))


def _make_trainer(max_epochs: int = -1, max_steps: int = -1, num_batches: int = 10) -> SimpleNamespace:
    """Create a lightweight trainer stub providing epoch, step, and dataloader bounds."""
    return SimpleNamespace(
        max_epochs=max_epochs,
        max_steps=max_steps,
        datamodule=SimpleNamespace(train_dataloader=lambda: [None] * num_batches),
    )


def _advance_scheduler(optimizer: Optimizer, scheduler: LRScheduler, num_steps: int) -> None:
    """Advance an optimizer/scheduler pair by ``num_steps`` iterations."""
    for _ in range(num_steps):
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize(
    ("max_epochs", "max_steps", "num_batches", "expected_step_size"),
    [
        (-1, 100, 10, 95),
        (10, -1, 20, 190),
    ],
)
def test_efficient_ad_lr_scheduler(
    max_epochs: int,
    max_steps: int,
    num_batches: int,
    expected_step_size: int,
) -> None:
    """EfficientAd uses step interval and decays by 0.1 at 95% of total steps."""
    model = EfficientAd(
        lr=1e-4,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model.trainer = _make_trainer(max_epochs=max_epochs, max_steps=max_steps, num_batches=num_batches)

    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    lr_cfg = config["lr_scheduler"]
    scheduler = lr_cfg["scheduler"]

    assert lr_cfg["interval"] == "step"
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == expected_step_size

    _advance_scheduler(optimizer, scheduler, expected_step_size - 1)
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4)

    _advance_scheduler(optimizer, scheduler, 1)
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("model_cls", "module", "attr", "stub_cls", "base_lr", "final_lr"),
    [
        (Dinomaly, dinomaly_module, "DinomalyModel", _StubDinomalyModel, 2e-3, 2e-4),
        (InpFormer, inp_former_module, "InpFormerModel", _StubInpFormerModel, 1e-3, 1e-4),
    ],
)
@pytest.mark.parametrize(
    ("max_epochs", "max_steps", "num_batches"),
    [
        (-1, 200, 10),
        (10, -1, 20),
    ],
)
def test_warm_cosine_lr_schedulers(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[Dinomaly | InpFormer],
    module: object,
    attr: str,
    stub_cls: type[nn.Module],
    base_lr: float,
    final_lr: float,
    max_epochs: int,
    max_steps: int,
    num_batches: int,
) -> None:
    """Dinomaly and INP-Former configure WarmCosineScheduler with step interval and warmup+decay progression."""
    monkeypatch.setattr(module, attr, stub_cls)
    model = model_cls(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    model.trainer = _make_trainer(max_epochs=max_epochs, max_steps=max_steps, num_batches=num_batches)

    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    lr_cfg = config["lr_scheduler"]
    scheduler = lr_cfg["scheduler"]

    assert lr_cfg["interval"] == "step"
    assert isinstance(scheduler, WarmCosineScheduler)

    # Initial LR starts at 0 and warms up to base_lr over 100 steps, then decays to final_lr.
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)
    _advance_scheduler(optimizer, scheduler, 100)
    assert scheduler.get_last_lr()[0] == pytest.approx(base_lr)
    _advance_scheduler(optimizer, scheduler, 100)
    assert scheduler.get_last_lr()[0] == pytest.approx(final_lr)


def test_uflow_lr_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uflow configures LinearLR with step interval and linear decay toward 0.4 * lr."""
    monkeypatch.setattr(uflow_module, "UflowModel", lambda **_kwargs: nn.Linear(1, 1))
    model = Uflow(post_processor=False, evaluator=False, visualizer=False)

    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    lr_cfg = config["lr_scheduler"]
    scheduler = lr_cfg["scheduler"]

    assert lr_cfg["interval"] == "step"
    assert isinstance(scheduler, LinearLR)
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-3)

    _advance_scheduler(optimizer, scheduler, 25000)
    assert scheduler.get_last_lr()[0] == pytest.approx(4e-4)


def test_uninet_lr_scheduler_step_and_epoch_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """UniNet selects step or epoch interval and clamps milestones to at least 1."""
    monkeypatch.setattr(uninet_module, "UniNetModel", _StubUniNetModel)
    model = UniNet(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)

    # Step-based training
    model.trainer = _make_trainer(max_epochs=-1, max_steps=100)
    step_cfg = model.configure_optimizers()
    step_sched = step_cfg["lr_scheduler"]["scheduler"]
    assert step_cfg["lr_scheduler"]["interval"] == "step"
    assert isinstance(step_sched, MultiStepLR)
    assert list(step_sched.milestones.keys()) == [80]

    # Multi-epoch training
    model.trainer = _make_trainer(max_epochs=10, max_steps=-1)
    epoch_cfg = model.configure_optimizers()
    epoch_sched = epoch_cfg["lr_scheduler"]["scheduler"]
    assert epoch_cfg["lr_scheduler"]["interval"] == "epoch"
    assert isinstance(epoch_sched, MultiStepLR)
    assert list(epoch_sched.milestones.keys()) == [8]

    # Single-epoch regression: milestone must be clamped to >= 1 so initial LR is not decayed at initialization.
    model.trainer = _make_trainer(max_epochs=1, max_steps=-1)
    single_cfg = model.configure_optimizers()
    single_opt = single_cfg["optimizer"]
    single_sched = single_cfg["lr_scheduler"]["scheduler"]
    assert single_cfg["lr_scheduler"]["interval"] == "epoch"
    assert list(single_sched.milestones.keys()) == [1]
    assert single_sched.get_last_lr()[0] == pytest.approx(5e-3)

    _advance_scheduler(single_opt, single_sched, 1)
    assert single_sched.get_last_lr()[0] == pytest.approx(1e-3)
