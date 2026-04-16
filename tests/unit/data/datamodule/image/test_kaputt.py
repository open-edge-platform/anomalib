# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Kaputt Datamodule."""

import warnings
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Kaputt
from anomalib.data.datasets.image.kaputt import ImageMode, ImageType, make_kaputt_dataset
from anomalib.data.utils import LabelName, Split, ValSplitMode
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


# too-many-public-methods is ignored till v2.6.0. This class will be simplified in v2.6.0.
class TestKaputt(_TestAnomalibImageDatamodule):  # noqa: PLR0904
    """Kaputt Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Kaputt:
        """Create and return a Kaputt datamodule."""
        datamodule_ = Kaputt(
            root=dataset_path / "kaputt",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
            val_split_mode=ValSplitMode.FROM_DIR,
        )
        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/kaputt.yaml"

    # ------------------------------------------------------------------
    # Legacy API tests (backward-compat: string image_type, bool flags)
    # Deprecated — will be removed in v2.6.0
    # ------------------------------------------------------------------

    @staticmethod
    def test_use_reference_adds_rows(dataset_path: Path) -> None:
        """Test that use_reference=True adds reference samples to the dataset."""
        root = dataset_path / "kaputt"
        samples_without = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=False)
        samples_with = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=True)

        assert len(samples_with) > len(samples_without)
        # All reference rows should be normal
        added = samples_with[~samples_with["image_path"].isin(samples_without["image_path"])]
        assert (added["label"] == "normal").all()
        assert (added["label_index"] == int(LabelName.NORMAL)).all()
        # Reference paths must point into reference-image/
        assert added["image_path"].str.contains("reference-image").all()

    @staticmethod
    def test_image_type_crop_switches_subdirectory(dataset_path: Path) -> None:
        """Test that image_type='crop' reads from query-crop/ instead of query-image/."""
        root = dataset_path / "kaputt"
        samples_image = make_kaputt_dataset(root, split=Split.TRAIN, image_type="image")
        samples_crop = make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop")

        assert len(samples_image) == len(samples_crop)
        assert samples_image["image_path"].str.contains("query-image").all()
        assert samples_crop["image_path"].str.contains("query-crop").all()

    @staticmethod
    def test_abnormal_samples_have_mask_path(dataset_path: Path) -> None:
        """Test that abnormal samples have non-empty mask_path and normals do not."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.VAL)

        abnormal = samples[samples["label_index"] == int(LabelName.ABNORMAL)]
        normal = samples[samples["label_index"] == int(LabelName.NORMAL)]

        assert len(abnormal) > 0, "Expected abnormal samples in val split"
        assert len(normal) > 0, "Expected normal samples in val split"
        assert (abnormal["mask_path"] != "").all(), "All abnormal samples should have a mask_path"
        assert (normal["mask_path"] == "").all(), "No normal sample should have a mask_path"

    @staticmethod
    def test_use_reference_crop_paths(dataset_path: Path) -> None:
        """Test that use_reference=True with image_type='crop' uses reference-crop/."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop", use_reference=True)

        ref_rows = samples[samples["image_path"].str.contains("reference-")]
        assert len(ref_rows) > 0
        assert ref_rows["image_path"].str.contains("reference-crop").all()

    # ------------------------------------------------------------------
    # ImageMode enum tests
    # ------------------------------------------------------------------

    @staticmethod
    def test_image_mode_query_only(dataset_path: Path) -> None:
        """Test that QUERY_ONLY returns only query samples with no reference rows."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.TRAIN, image_mode=ImageMode.QUERY_ONLY)

        assert len(samples) > 0
        assert not samples["image_path"].str.contains("reference-").any()
        assert samples["image_path"].str.contains("query-").all()

    @staticmethod
    def test_image_mode_query_and_reference(dataset_path: Path) -> None:
        """Test that QUERY_AND_REFERENCE includes both query and reference rows."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.TRAIN, image_mode=ImageMode.QUERY_AND_REFERENCE)

        query_rows = samples[samples["image_path"].str.contains("query-")]
        ref_rows = samples[samples["image_path"].str.contains("reference-")]

        assert len(query_rows) > 0, "Expected query samples"
        assert len(ref_rows) > 0, "Expected reference samples"
        assert len(samples) == len(query_rows) + len(ref_rows)

    @staticmethod
    def test_image_mode_reference_only(dataset_path: Path) -> None:
        """Test that REFERENCE_ONLY returns only reference samples, all normal."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.VAL, image_mode=ImageMode.REFERENCE_ONLY)

        assert len(samples) > 0
        assert samples["image_path"].str.contains("reference-").all()
        assert not samples["image_path"].str.contains("query-").any()
        assert (samples["label"] == "normal").all()
        assert (samples["label_index"] == int(LabelName.NORMAL)).all()

    @staticmethod
    def test_image_mode_equivalence_with_legacy(dataset_path: Path) -> None:
        """Test that new ImageMode values produce identical results to legacy bool flags."""
        root = dataset_path / "kaputt"

        legacy_without = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=False)
        new_query_only = make_kaputt_dataset(root, split=Split.TRAIN, image_mode=ImageMode.QUERY_ONLY)
        assert list(legacy_without["image_path"]) == list(new_query_only["image_path"])

        legacy_with = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=True)
        new_query_and_ref = make_kaputt_dataset(root, split=Split.TRAIN, image_mode=ImageMode.QUERY_AND_REFERENCE)
        assert list(legacy_with["image_path"]) == list(new_query_and_ref["image_path"])

    @staticmethod
    def test_image_mode_reference_only_with_crop(dataset_path: Path) -> None:
        """Test REFERENCE_ONLY combined with ImageType.CROP uses reference-crop/."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(
            root,
            split=Split.TRAIN,
            image_type=ImageType.CROP,
            image_mode=ImageMode.REFERENCE_ONLY,
        )

        assert len(samples) > 0
        assert samples["image_path"].str.contains("reference-crop").all()

    # ------------------------------------------------------------------
    # ImageType enum tests
    # ------------------------------------------------------------------

    @staticmethod
    def test_image_type_enum_matches_string(dataset_path: Path) -> None:
        """Test that ImageType enum produces identical results to legacy string."""
        root = dataset_path / "kaputt"

        from_string = make_kaputt_dataset(root, split=Split.TRAIN, image_type="image")
        from_enum = make_kaputt_dataset(root, split=Split.TRAIN, image_type=ImageType.IMAGE)
        assert list(from_string["image_path"]) == list(from_enum["image_path"])

        from_string_crop = make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop")
        from_enum_crop = make_kaputt_dataset(root, split=Split.TRAIN, image_type=ImageType.CROP)
        assert list(from_string_crop["image_path"]) == list(from_enum_crop["image_path"])

    # ------------------------------------------------------------------
    # Category filtering tests
    # ------------------------------------------------------------------

    @staticmethod
    def test_category_filters_by_item_material(dataset_path: Path) -> None:
        """Test that category filters query samples by item_material."""
        root = dataset_path / "kaputt"

        all_samples = make_kaputt_dataset(root, split=Split.VAL)
        cardboard = make_kaputt_dataset(root, split=Split.VAL, category="cardboard")
        plastic = make_kaputt_dataset(root, split=Split.VAL, category="plastic")

        assert len(all_samples) > len(cardboard)
        assert len(all_samples) > len(plastic)
        assert len(cardboard) > 0
        assert len(plastic) > 0
        assert (cardboard["item_material"] == "cardboard").all()
        assert (plastic["item_material"] == "plastic").all()

    @staticmethod
    def test_category_cardboard_is_normal(dataset_path: Path) -> None:
        """Test that category='cardboard' yields only normal samples in dummy data."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.VAL, category="cardboard")

        assert len(samples) > 0
        assert (samples["label"] == "normal").all()

    @staticmethod
    def test_category_plastic_is_abnormal(dataset_path: Path) -> None:
        """Test that category='plastic' yields only abnormal samples in dummy data."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.VAL, category="plastic")

        assert len(samples) > 0
        assert (samples["label"] == "abnormal").all()

    @staticmethod
    def test_category_none_returns_all(dataset_path: Path) -> None:
        """Test that category=None returns same as no category argument."""
        root = dataset_path / "kaputt"

        default = make_kaputt_dataset(root, split=Split.VAL)
        explicit_none = make_kaputt_dataset(root, split=Split.VAL, category=None)
        assert list(default["image_path"]) == list(explicit_none["image_path"])

    @staticmethod
    def test_category_nonexistent_returns_empty(dataset_path: Path) -> None:
        """Test that a category not in the data raises RuntimeError (no images found)."""
        root = dataset_path / "kaputt"
        with pytest.raises(RuntimeError, match="Found 0 images"):
            make_kaputt_dataset(root, split=Split.TRAIN, category="nonexistent_material")

    @staticmethod
    def test_category_with_reference_mode(dataset_path: Path) -> None:
        """Test that category filtering applies to query rows but reference rows are still included."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(
            root,
            split=Split.VAL,
            category="cardboard",
            image_mode=ImageMode.QUERY_AND_REFERENCE,
        )

        query_rows = samples[samples["image_path"].str.contains("query-")]
        ref_rows = samples[samples["image_path"].str.contains("reference-")]

        assert len(query_rows) > 0
        assert len(ref_rows) > 0
        assert (query_rows["item_material"] == "cardboard").all()

    # ------------------------------------------------------------------
    # Deprecation warning tests
    # ------------------------------------------------------------------

    @staticmethod
    def test_use_reference_emits_deprecation_warning(dataset_path: Path) -> None:
        """Test that passing use_reference emits a DeprecationWarning."""
        root = dataset_path / "kaputt"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_kaputt_dataset(root, split=Split.TRAIN, use_reference=True)

        deprecation_msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_msgs) >= 1
        assert "use_reference" in str(deprecation_msgs[0].message)

    @staticmethod
    def test_reference_only_emits_deprecation_warning(dataset_path: Path) -> None:
        """Test that passing reference_only emits a DeprecationWarning."""
        root = dataset_path / "kaputt"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_kaputt_dataset(root, split=Split.TRAIN, reference_only=True)

        deprecation_msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_msgs) >= 1
        assert "reference_only" in str(deprecation_msgs[0].message)

    @staticmethod
    def test_string_image_type_emits_deprecation_warning(dataset_path: Path) -> None:
        """Test that passing image_type as a string emits a DeprecationWarning."""
        root = dataset_path / "kaputt"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop")

        deprecation_msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_msgs) >= 1
        assert "image_type" in str(deprecation_msgs[0].message)

    @staticmethod
    def test_enum_image_type_no_warning(dataset_path: Path) -> None:
        """Test that passing ImageType enum does not emit a DeprecationWarning."""
        root = dataset_path / "kaputt"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_kaputt_dataset(root, split=Split.TRAIN, image_type=ImageType.IMAGE)

        deprecation_msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_msgs) == 0
