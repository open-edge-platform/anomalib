# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for VerdictAD's equations, decisions, and Anomalib integration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from jsonargparse import ArgumentParser
from lightning import Trainer
from PIL import Image
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO, Dinomaly, Patchcore, VerdictAD, get_model
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from anomalib.models.image.verdict_ad.decider import BOXES, Q_REF, ClefFlash, decision_records
from anomalib.models.image.verdict_ad.refinement import refine_predictions
from anomalib.models.image.verdict_ad.torch_model import VerdictADModel


class TinyEncoder(nn.Module):
    """Replace only pretrained feature extraction; exercise the actual detector."""

    patch_size = 14

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.layer = kwargs["layers"][0]

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return deterministic patch tokens."""
        patches = F.avg_pool2d(images, self.patch_size, self.patch_size)
        return {self.layer: patches.flatten(2).transpose(1, 2)}


class TinyDinomalyEncoder(nn.Module):
    """Return small tokens while retaining Dinomaly's actual decoder and loss."""

    patch_size = 14
    num_register_tokens = 0

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.layers = kwargs["layers"]
        self.weight = nn.Parameter(torch.ones(24), requires_grad=False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Provide a class token and four patch tokens for every encoder layer."""
        patches = F.avg_pool2d(images, 14, 14).flatten(2).transpose(1, 2).repeat(1, 1, 8)
        tokens = torch.cat([patches.mean(1, keepdim=True), patches], dim=1) * self.weight
        return dict.fromkeys(self.layers, tokens)


@pytest.fixture(autouse=True)
def offline_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent pretrained downloads while retaining AnomalyDINO scoring."""
    monkeypatch.setattr("anomalib.models.image.anomaly_dino.torch_model.TimmFeatureExtractor", TinyEncoder)


@pytest.mark.parametrize("channels", [False, True])
def test_refinement_equations(channels: bool) -> None:
    """Average logits globally, but average powered probabilities spatially."""
    maps = torch.ones(2, 2, 3)
    maps[0, 0, 0] = 0
    if channels:
        maps = maps.unsqueeze(1)
    predictions = InferenceBatch(pred_score=torch.tensor([2.0, 4.0]), anomaly_map=maps)
    global_logits = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    probabilities = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    boxes = [(0.0, 0.0, 2 / 3, 1.0), (1 / 3, 0.0, 1.0, 1.0)]
    result = refine_predictions(predictions, global_logits, probabilities.logit(), boxes, alpha=1.0, gamma=2.0)
    assert torch.allclose(result.pred_score, torch.tensor([1.0, 2.0]))
    expected = torch.tensor([[[0.0, 0.3125, 0.5625], [0.0625, 0.3125, 0.5625]], [[0.25] * 3] * 2])
    assert torch.allclose(result.anomaly_map, expected.unsqueeze(1) if channels else expected)
    assert torch.equal(predictions.anomaly_map, maps)
    assert (result.anomaly_map <= maps).all()


def test_disabled_refinement_and_extreme_logits() -> None:
    """Zero exponents preserve the baseline; large finite logits remain stable."""
    predictions = InferenceBatch(pred_score=torch.ones(2), anomaly_map=torch.ones(2, 1, 7, 11))
    global_logits = torch.tensor([[-1000.0, -1000.0], [1000.0, 1000.0]])
    windows = global_logits[:, :1].expand(2, 32)
    identity = refine_predictions(predictions, global_logits, windows, BOXES * 2, alpha=0, gamma=0)
    assert torch.equal(identity.pred_score, predictions.pred_score)
    assert torch.equal(identity.anomaly_map, predictions.anomaly_map)
    result = refine_predictions(predictions, global_logits, windows, BOXES * 2)
    assert torch.equal(result.pred_score, torch.tensor([0.0, 1.0]))
    assert result.anomaly_map[0].count_nonzero() == 0
    assert torch.equal(result.anomaly_map[1], predictions.anomaly_map[1])


def test_refinement_preserves_small_probabilities() -> None:
    """Apply fractional powers before rounding small sigmoid probabilities to zero."""
    predictions = InferenceBatch(pred_score=torch.ones(1), anomaly_map=torch.ones(1, 1, 7, 11))
    result = refine_predictions(
        predictions,
        torch.full((1, 2), -100.0),
        torch.full((1, 32), -100.0),
        BOXES * 2,
    )
    expected_score = torch.tensor(-50.0).exp().expand_as(result.pred_score)
    expected_map = torch.tensor(-25.0).exp().expand_as(result.anomaly_map)
    torch.testing.assert_close(result.pred_score, expected_score, atol=0, rtol=1e-6)
    torch.testing.assert_close(result.anomaly_map, expected_map, atol=0, rtol=1e-6)


@pytest.mark.parametrize("failure", ["negative", "nan", "shape", "uncovered", "box", "exponent"])
def test_invalid_refinement_inputs(failure: str) -> None:
    """Reject invalid evidence instead of silently corrupting anomaly scores."""
    scores, maps = torch.ones(1), torch.ones(1, 1, 8, 8)
    global_logits, windows = torch.zeros(1, 2), torch.zeros(1, 1)
    boxes = [(0.0, 0.0, 1.0, 1.0)]
    alpha = 0.5
    if failure == "negative":
        scores *= -1
    elif failure == "nan":
        windows[:] = torch.nan
    elif failure == "shape":
        global_logits = torch.zeros(2, 1)
    elif failure == "uncovered":
        boxes = [(0.0, 0.0, 0.5, 1.0)]
    elif failure == "box":
        boxes = [(0.0, 0.0, 2.0, 1.0)]
    else:
        alpha = -1.0
    with pytest.raises(ValueError, match=r"nonnegative|finite|shapes|cover|coordinates"):
        refine_predictions(
            InferenceBatch(pred_score=scores, anomaly_map=maps),
            global_logits,
            windows,
            boxes,
            alpha=alpha,
        )


@pytest.mark.parametrize("shots", [1, 2, 4])
def test_decision_views(shots: int) -> None:
    """Check global resolution, reference tiling, crops, and red boxes."""
    references = [Image.new("RGB", (20, 20), color=(index * 60, 0, 0)) for index in range(shots)]
    records = decision_records(Image.new("RGB", (30, 20), color="white"), references)
    assert len(records) == 34
    assert records[0][1][0].size == (896, 896)
    assert len(records[0][1]) == 1
    grid = records[1][1][1]
    for index, coordinate in enumerate([(0, 0), (448, 0), (0, 448), (448, 448)]):
        assert grid.getpixel(coordinate) == ((index % shots) * 60, 0, 0)
    assert records[2][1][0].size == (448, 448)
    assert records[18][1][0].getpixel((0, 0)) == (255, 0, 0)
    assert records[18][1][0].getpixel((200, 200)) == (255, 255, 255)
    with pytest.raises(ValueError, match="one to four"):
        decision_records(references[0], [])


@pytest.mark.parametrize(
    ("include_global", "include_local", "sizes_expected"),
    [
        (True, True, [3] * 11 + [1]),
        (True, False, [2]),
        (False, True, [3] * 10 + [2]),
        (False, False, []),
    ],
)
def test_decider_option_order_and_batching(
    monkeypatch: pytest.MonkeyPatch,
    include_global: bool,
    include_local: bool,
    sizes_expected: list[int],
) -> None:
    """Interpret option IDs, not assumed logit positions, including the final partial batch."""
    decider = ClefFlash(batch_size=3)
    sizes = []
    encoded = SimpleNamespace(questions=[SimpleNamespace(option_ids=["true", "false"])])
    monkeypatch.setattr(decider, "_processor", SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(
        decider,
        "_release",
        SimpleNamespace(
            encode_record=lambda *_args, **_kwargs: encoded,
            collate_records=lambda encs, *_args: encs,
        ),
    )

    def model(batch: list) -> list:
        sizes.append(len(batch))
        return [[torch.tensor([3.0, 1.0])] for _ in batch]

    monkeypatch.setattr(decider, "_model", model)
    monkeypatch.setattr(decider, "_load", lambda _device: None)
    if not include_local:
        monkeypatch.setattr(
            "anomalib.models.image.verdict_ad.decider._crop",
            Mock(side_effect=AssertionError("Disabled local refinement must not build crops")),
        )
    image = Image.new("RGB", (14, 14))
    logits = decider.predict(
        image,
        [image],
        torch.device("cpu"),
        include_global=include_global,
        include_local=include_local,
    )
    assert torch.equal(logits, torch.full((sum(sizes_expected),), 2.0))
    assert sizes == sizes_expected


@pytest.mark.parametrize("name", ["VerdictAD", "verdict_ad", "verdict_a_d"])
def test_model_registration(name: str) -> None:
    """Resolve both user-facing names and automatic snake casing."""
    model = get_model(name, alpha=0, gamma=0)
    assert isinstance(model, VerdictAD)
    assert isinstance(model.trainer_arguments, dict)
    assert model.learning_type == LearningType.ONE_CLASS


def test_reference_selection_fitting_and_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep exactly the requested references across batches, fitting and reload."""
    model = VerdictADModel(n_shots=2, alpha=1, gamma=1)
    images = torch.rand(4, 3, 28, 28)
    model(images[:1])
    model(images[1:])
    assert torch.equal(model.reference_images, images[:2])
    model.fit()
    assert len(model.detector.memory_bank) == 8
    model.eval()
    calls = Mock(return_value=torch.zeros(34))
    monkeypatch.setattr(model.decider, "predict", calls)
    result = model(images[2:])
    assert result.pred_score.shape == (2,)
    assert result.anomaly_map.shape == (2, 1, 28, 28)
    assert calls.call_count == 2
    assert len(calls.call_args.args[1]) == 2
    restored = VerdictADModel(n_shots=2, alpha=1, gamma=1).eval()
    restored.load_state_dict(model.state_dict())
    monkeypatch.setattr(restored.decider, "predict", calls)
    restored_result = restored(images[2:])
    assert torch.equal(restored.reference_images, images[:2])
    assert torch.equal(restored_result.pred_score, result.pred_score)
    assert torch.equal(restored_result.anomaly_map, result.anomaly_map)


def test_training_validation_and_disabled_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Lightning steps with actual AnomalyDINO and no optional dependencies."""
    model = VerdictAD(n_shots=1, alpha=0, gamma=0, pre_processor=False, visualizer=False)
    backend = Mock(side_effect=AssertionError("Disabled refinement must not load Clef-Flash"))
    monkeypatch.setattr(model.model.decider, "predict", backend)
    batch = ImageBatch(image=torch.rand(2, 3, 28, 28), gt_label=torch.zeros(2, dtype=torch.bool))
    loss = model.training_step(batch)
    assert loss.requires_grad
    model.on_train_epoch_end()
    model.eval()
    predictions = model.validation_step(batch)
    assert predictions.pred_score.shape == (2,)
    assert predictions.anomaly_map.shape == (2, 28, 28)
    assert not backend.called
    batch.gt_label = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="normal"):
        model.training_step(batch)


@pytest.mark.parametrize("detector", ["anomaly_dino", "patchcore"])
def test_base_detector_training_and_refinement(detector: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both detectors fit the selected references and refine their own raw outputs."""
    configured_detector = (
        AnomalyDINO(num_neighbours=2)
        if detector == "anomaly_dino"
        else Patchcore(backbone="resnet18", pre_trained=False, coreset_sampling_ratio=0.5, num_neighbors=1)
    )
    model = VerdictAD(
        n_shots=1,
        detector=configured_detector,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    assert model.model.detector is configured_detector.model
    images = torch.rand(2, 3, 56, 56)
    model.training_step(ImageBatch(image=images[:1], gt_label=[0]))
    model.on_train_epoch_end()
    model.eval()
    calls = Mock(return_value=torch.zeros(34))
    monkeypatch.setattr(model.model.decider, "predict", calls)
    result = model.validation_step(ImageBatch(image=images, gt_label=[0, 1]))
    assert calls.call_count == 2
    assert torch.equal(model.model.reference_images, images[:1])
    assert result.pred_score.shape == (2,)
    assert result.anomaly_map.shape == (2, 56, 56)
    assert torch.isfinite(result.pred_score).all()
    assert torch.isfinite(result.anomaly_map).all()
    model.model.alpha = model.model.gamma = 0
    baseline = model.model(images)
    torch.testing.assert_close(result.pred_score, baseline.pred_score * 0.5**0.5)
    torch.testing.assert_close(result.anomaly_map, baseline.anomaly_map.squeeze(1) * 0.5**0.25)


def test_nested_detector_configuration() -> None:
    """CLI configuration instantiates the detector with its own constructor settings."""
    parser = ArgumentParser()
    parser.add_class_arguments(VerdictAD, "model")
    config = parser.parse_object({
        "model": {
            "detector": {
                "class_path": "anomalib.models.Patchcore",
                "init_args": {"backbone": "resnet18", "pre_trained": False, "coreset_sampling_ratio": 0.05},
            },
            "alpha": 0,
            "gamma": 0,
        },
    })
    model = parser.instantiate_classes(config).model
    assert isinstance(model.model.detector, PatchcoreModel)
    assert model.model.detector.backbone == "resnet18"
    assert model.hparams["detector"]["init_args"]["coreset_sampling_ratio"] == 0.05
    assert model.hparams_initial["detector"] == model.hparams["detector"]


@pytest.mark.parametrize("detector", [None, {"class_path": "anomalib.models.Patchcore"}])
@pytest.mark.parametrize(
    "options",
    [
        {"n_shots": 0},
        {"n_shots": True},
        {"alpha": -1},
        {"gamma": float("nan")},
        {"decision_batch_size": 0},
        {"decision_batch_size": True},
        {"decision_prompts": {"unknown": "question"}},
        {"decision_prompts": {"free": " "}},
    ],
)
def test_invalid_wrapper_options_do_not_load_detector(
    detector: dict | None,
    options: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate wrapper settings before any default or configured weight loading."""
    initialize = Mock(side_effect=AssertionError("Detector initialization must not run"))
    monkeypatch.setattr(AnomalyDINO, "__init__", initialize)
    monkeypatch.setattr(Patchcore, "__init__", initialize)
    with pytest.raises(ValueError, match=r"n_shots|alpha|gamma|batch_size|decision_prompts"):
        VerdictAD(detector=detector, **options)
    initialize.assert_not_called()


def test_patchcore_checkpoint_restores_detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lightning checkpoints retain the selected detector and its memory bank."""
    model = VerdictAD(
        n_shots=1,
        detector=Patchcore(backbone="resnet18", pre_trained=False, coreset_sampling_ratio=0.5),
        alpha=0,
        gamma=0,
        pre_processor=False,
        evaluator=False,
        visualizer=False,
    )
    data = DataLoader([ImageBatch(image=torch.rand(1, 3, 56, 56), gt_label=[0])], batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=data)
    checkpoint = tmp_path / "patchcore.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = VerdictAD.load_from_checkpoint(checkpoint, weights_only=True)
    assert type(restored.model.detector) is type(model.model.detector)
    assert restored.hparams["detector"] == model.hparams["detector"]
    assert restored.hparams_initial["detector"] == restored.hparams["detector"]
    assert restored._sampling_ratio == 0.5  # noqa: SLF001
    assert torch.equal(restored.model.reference_images, model.model.reference_images)
    assert torch.equal(restored.model.detector.memory_bank, model.model.detector.memory_bank)
    model.eval()
    restored.eval()
    images = torch.rand(1, 3, 56, 56)
    torch.testing.assert_close(model.model(images).pred_score, restored.model(images).pred_score)
    monkeypatch.setattr(restored.model.decider, "predict", Mock(side_effect=AssertionError("Refinement disabled")))
    restored.model(images)
    # Earlier PR checkpoints can be loaded by providing their matching detector explicitly.
    legacy = torch.load(checkpoint, weights_only=True)
    legacy["hyper_parameters"]["detector"] = "patchcore"
    legacy["hyper_parameters"]["patchcore_backbone"] = "resnet18"
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save(legacy, legacy_path)
    migrated = VerdictAD.load_from_checkpoint(
        legacy_path,
        weights_only=True,
        detector=Patchcore(backbone="resnet18", pre_trained=False, coreset_sampling_ratio=0.5),
    ).eval()
    torch.testing.assert_close(model.model(images).pred_score, migrated.model(images).pred_score)


@pytest.mark.parametrize("detector", ["unknown", nn.Identity()])
def test_invalid_detector_configuration(detector: object) -> None:
    """Reject unsupported detector instances before loading weights."""
    with pytest.raises(ValueError, match="detector"):
        VerdictADModel(detector=detector)
    with pytest.raises(ValueError, match="detector"):
        VerdictAD(detector=detector)


def test_invalid_detector_checkpoint_configuration() -> None:
    """Only the three supported first-party detector classes can be reconstructed."""
    with pytest.raises(ValueError, match="detector configuration"):
        VerdictAD(detector={"class_path": "os.system", "init_args": {}})


def test_patchcore_empty_coreset_is_rejected() -> None:
    """Report insufficient retained features and leave fit retryable."""
    model = VerdictADModel(
        n_shots=1,
        detector=PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False),
    )
    model(torch.rand(1, 3, 56, 56))
    with pytest.raises(ValueError, match="coreset would be empty"):
        model.fit(sampling_ratio=1e-9)
    assert not model.detector.embedding_store
    model.fit(sampling_ratio=0.5)
    assert model.detector.memory_bank.numel() > 0


def test_singleton_patchcore_scores_are_batch_invariant() -> None:
    """A singleton bank retains informative scores in every query batch size."""
    model = VerdictADModel(
        n_shots=1,
        detector=PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False),
        alpha=0,
        gamma=0,
    )
    model(torch.rand(1, 3, 28, 28))
    model.fit()
    assert len(model.detector.memory_bank) == 1
    restored = VerdictADModel(
        n_shots=1,
        detector=PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False),
        alpha=0,
        gamma=0,
    )
    restored.load_state_dict(model.state_dict())
    restored.eval()
    query = torch.rand(3, 3, 28, 28)
    single = restored(query[:1]).pred_score
    assert single.item() > 0
    torch.testing.assert_close(single, restored(query).pred_score[:1])


def test_prompt_overrides_reach_decision_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom questions reach the release while omitted questions retain their defaults."""
    image = Image.new("RGB", (28, 28))
    prompts = {"free": "Does this image contain an unacceptable scratch?", "crop": "Is this crop defective?"}
    records = decision_records(image, [image], prompts=prompts)
    assert records[0][2] == prompts["free"]
    assert records[1][2] == Q_REF
    assert all(record[2] == prompts["crop"] for record in records[2:18])
    decider = ClefFlash(prompts=prompts, batch_size=2)
    seen = []

    def encode(_tokenizer: SimpleNamespace, record: dict, **_kwargs) -> SimpleNamespace:
        seen.append(record["questions"]["defect"])
        return SimpleNamespace(questions=[SimpleNamespace(option_ids=["true", "false"])])

    monkeypatch.setattr(
        decider,
        "_release",
        SimpleNamespace(encode_record=encode, collate_records=lambda batch, *_: batch),
    )
    monkeypatch.setattr(decider, "_processor", SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(decider, "_load", lambda _device: None)
    monkeypatch.setattr(decider, "_model", lambda batch: [[torch.tensor([1.0, 0.0])] for _ in batch])
    decider.predict(image, [image], torch.device("cpu"), include_local=False)
    assert seen == [{"type": "noul", "instructions": prompts["free"]}, {"type": "noul", "instructions": Q_REF}]


@pytest.mark.parametrize("prompts", [{"unknown": "question"}, {"free": ""}, {"box": "  "}, {"crop": 1}])
def test_invalid_prompts_are_rejected(prompts: dict) -> None:
    """Reject misspelled keys and unusable questions before loading decision weights."""
    with pytest.raises(ValueError, match="decision_prompts"):
        ClefFlash(prompts=prompts)


def test_dinomaly_training_refinement_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train the real reconstruction path, refine its outputs, and restore custom prompts."""
    monkeypatch.setattr("anomalib.models.image.dinomaly.torch_model.TimmFeatureExtractor", TinyDinomalyEncoder)
    monkeypatch.setattr(
        "anomalib.models.image.dinomaly.torch_model.DINO_ARCHITECTURES",
        {
            "small": {"embed_dim": 24, "num_heads": 3, "target_layers": list(range(2, 10))},
        },
    )
    model = VerdictAD(
        detector=Dinomaly(encoder_name="vit_small_patch14_dinov2", decoder_depth=10, bottleneck_dropout=0.1),
        n_shots=1,
        decision_prompts={"free": "Is there an unacceptable defect?"},
        pre_processor=False,
        evaluator=False,
        visualizer=False,
    )
    assert "max_epochs" not in model.trainer_arguments
    assert "max_steps" not in model.trainer_arguments
    before = [parameter.detach().clone() for parameter in model.model.detector.decoder.parameters()]
    batches = [ImageBatch(image=torch.rand(1, 3, 28, 28), gt_label=[0]) for _ in range(2)]
    validation_images = torch.rand(2, 3, 28, 28)
    validation = DataLoader([ImageBatch(image=validation_images.clone(), gt_label=[0, 1])], batch_size=None)
    monkeypatch.setattr(
        ClefFlash,
        "predict",
        lambda _self, _image, _refs, device, **_kwargs: torch.zeros(34, device=device),
    )
    engine = Engine(
        accelerator="cpu",
        max_steps=2,
        max_epochs=3,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=1,
        default_root_dir=tmp_path,
    )
    engine.fit(model, train_dataloaders=DataLoader(batches, batch_size=None), val_dataloaders=validation)
    trainer = engine.trainer
    assert trainer.max_steps == 2
    assert trainer.global_step == 2
    assert any(
        not torch.equal(old, current)
        for old, current in zip(before, model.model.detector.decoder.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.model.detector.encoder.parameters())
    assert torch.equal(model.model.reference_images, batches[0].image)
    model.eval()
    calibrated = model.model(validation_images)
    torch.testing.assert_close(model.post_processor.image_max, calibrated.pred_score.max())
    checkpoint = tmp_path / "dinomaly.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = VerdictAD.load_from_checkpoint(checkpoint, weights_only=True)
    assert restored.model.decider.prompts == model.model.decider.prompts
    assert len(restored.model.detector.decoder) == 10
    restored.eval()
    query = torch.rand(2, 3, 28, 28)
    decisions = Mock(return_value=torch.zeros(34))
    monkeypatch.setattr(restored.model.decider, "predict", decisions)
    refined = restored.model(query)
    restored.model.alpha = restored.model.gamma = 0
    raw = restored.model(query)
    torch.testing.assert_close(refined.pred_score, raw.pred_score * 0.5**0.5)
    torch.testing.assert_close(refined.anomaly_map, raw.anomaly_map * 0.5**0.25)
    assert decisions.call_count == 2
    predictions = trainer.predict(restored, dataloaders=validation)
    assert predictions[0].pred_score.shape == (2,)


@pytest.mark.parametrize(("alpha", "gamma", "count"), [(0.5, 0.0, 2), (0.0, 0.25, 32)])
def test_single_stage_refinement(monkeypatch: pytest.MonkeyPatch, alpha: float, gamma: float, count: int) -> None:
    """Refine BF16 queries with only the enabled decisions."""
    model = VerdictADModel(n_shots=1, alpha=alpha, gamma=gamma)
    images = torch.rand(2, 3, 28, 28, dtype=torch.bfloat16)
    model(images[:1])
    model.fit()
    model.eval()
    backend = Mock(return_value=torch.zeros(count))
    monkeypatch.setattr(model.decider, "predict", backend)
    result = model(images)
    assert backend.call_count == 2
    assert backend.call_args.kwargs == {"include_global": bool(alpha), "include_local": bool(gamma)}
    model.alpha = model.gamma = 0
    baseline = model(images)
    torch.testing.assert_close(result.pred_score, baseline.pred_score * 0.5**alpha)
    torch.testing.assert_close(result.anomaly_map, baseline.anomaly_map * 0.5**gamma)


def test_refine_fp16_outputs_before_rounding() -> None:
    """A small gate can yield a representable FP16 prediction after scaling."""
    score = torch.tensor([65504], dtype=torch.float16)
    raw = InferenceBatch(pred_score=score, anomaly_map=score.reshape(1, 1, 1))
    result = refine_predictions(raw, torch.full((1, 2), -40.0), torch.full((1, 1), -80.0), [(0, 0, 1, 1)])
    expected = (score.double() * torch.tensor(-20.0).exp()).half()
    torch.testing.assert_close(result.pred_score, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.anomaly_map, expected.reshape(1, 1, 1), rtol=0, atol=0)


def test_refine_fp64_logits() -> None:
    """Wide finite logits cancel globally and retain small local probabilities."""
    score = torch.ones(1, dtype=torch.float64)
    raw = InferenceBatch(pred_score=score, anomaly_map=score.reshape(1, 1, 1))
    result = refine_predictions(
        raw,
        torch.tensor([[-1e100, 1e100]], dtype=torch.float64),
        torch.tensor([[-500.0]], dtype=torch.float64),
        [(0, 0, 1, 1)],
    )
    torch.testing.assert_close(result.pred_score, score * 0.5**0.5)
    torch.testing.assert_close(result.anomaly_map, (-125 * raw.anomaly_map).exp(), rtol=1e-12, atol=0)


def test_external_weights_follow_parent_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Move external parameters and buffers without adding them to checkpoints."""
    model = VerdictADModel(n_shots=1)
    loader = Mock(side_effect=AssertionError("Device moves must not load Clef-Flash"))
    monkeypatch.setattr(model.decider, "_load", loader)
    model.cpu()
    state_keys = set(model.state_dict())
    external = nn.Sequential(nn.Linear(2, 2))
    external.register_buffer("cache", torch.ones(2))
    monkeypatch.setattr(model.decider, "_model", external)
    apply = Mock(wraps=external._apply)  # noqa: SLF001
    monkeypatch.setattr(external, "_apply", apply)
    assert model.cpu() is model
    apply.assert_called_once()
    model.to(dtype=torch.float64)
    assert all(value.dtype == torch.float64 for value in external.state_dict().values())
    model.to("meta")
    assert all(value.device.type == "meta" for value in external.state_dict().values())
    model.to_empty(device="cpu")
    assert all(value.device.type == "cpu" for value in external.state_dict().values())
    assert set(model.state_dict()) == state_keys
    loader.assert_not_called()


def test_reference_and_input_errors() -> None:
    """Fail clearly on missing references, invalid RGB input, and missing releases."""
    model = VerdictADModel(n_shots=2)
    with pytest.raises(ValueError, match="collected 0"):
        model.fit()
    with pytest.raises(RuntimeError, match="before inference"):
        model.eval()(torch.zeros(1, 3, 28, 28))
    with pytest.raises(ValueError, match="normalization"):
        model(torch.full((1, 3, 28, 28), -1.0))
    with pytest.raises(ValueError, match="decision_model_path"):
        ClefFlash().predict(Image.new("RGB", (28, 28)), [Image.new("RGB", (28, 28))], torch.device("cpu"))


def test_preprocessor_preserves_rgb_range() -> None:
    """The shared RGB input must remain valid for both detector and decider."""
    image = torch.zeros(1, 3, 14, 14)
    image[:, :, 7:, 7:] = 1
    output = VerdictAD.configure_pre_processor((28, 28))(image)
    assert output.shape == (1, 3, 28, 28)
    assert output.min() >= 0
    assert output.max() <= 1


def test_default_resize_roundoff_is_accepted() -> None:
    """A valid white image survives default resizing, training, and inference."""
    model = VerdictAD(n_shots=1, alpha=0, gamma=0, visualizer=False)
    batch = ImageBatch(image=torch.ones(1, 3, 1024, 1024), gt_label=[0])
    model.pre_processor.on_train_batch_start(None, model, batch, 0)
    resized = batch.image.clone()
    model.training_step(batch)
    assert model.model.reference_images.min() >= 0
    assert model.model.reference_images.max() <= 1
    assert torch.equal(batch.image, resized)
    model.on_train_epoch_end()
    model.eval()
    assert torch.isfinite(model.validation_step(batch).pred_score).all()


@pytest.mark.parametrize("value", [-1e-3, 1.001, float("nan"), float("inf")])
def test_roundoff_tolerance_rejects_invalid_input(value: float) -> None:
    """Clamping must not silently accept invalid ranges or nonfinite pixels."""
    with pytest.raises(ValueError, match="normalization"):
        VerdictADModel(n_shots=1)(torch.full((1, 3, 28, 28), value))


@pytest.mark.parametrize("method", ["to_torch", "to_onnx", "to_openvino"])
def test_export_rejected(method: str, tmp_path: Path) -> None:
    """Reject unsupported exports before producing an incomplete artifact."""
    model = VerdictAD()
    with pytest.raises(NotImplementedError, match="Lightning checkpoint"):
        getattr(model, method)(export_root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_lightning_calibration_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibrate on refined predictions and restore references through a real checkpoint."""
    monkeypatch.setattr(
        ClefFlash,
        "predict",
        lambda _self, _image, _refs, device, **_kwargs: torch.zeros(34, device=device),
    )
    model = VerdictAD(n_shots=1, alpha=1, gamma=1, pre_processor=False, evaluator=False, visualizer=False)
    train = DataLoader([ImageBatch(image=torch.rand(1, 3, 28, 28), gt_label=[0])], batch_size=None)
    validation_images = torch.rand(2, 3, 28, 28)
    validation = DataLoader([ImageBatch(image=validation_images.clone(), gt_label=[0, 1])], batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=validation)
    model.eval()
    refined = model.model(validation_images)
    assert torch.allclose(model.post_processor.image_max, refined.pred_score.max())
    assert torch.allclose(model.post_processor.pixel_max, refined.anomaly_map.max())
    checkpoint = tmp_path / "verdict.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = VerdictAD.load_from_checkpoint(checkpoint, weights_only=True)
    restored.eval()
    assert torch.equal(restored.model.reference_images, model.model.reference_images)
    assert torch.equal(restored.post_processor.image_max, model.post_processor.image_max)
    assert torch.equal(restored.model(validation_images).pred_score, refined.pred_score)
    predictions = trainer.predict(restored, dataloaders=validation)
    expected = restored.post_processor(refined)
    assert torch.allclose(predictions[0].pred_score, expected.pred_score)
