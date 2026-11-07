# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for str-based Enum compatibility with pandas >= 3.0.

pandas 3.0 introduced ``StringDtype`` as the default for string data,
which changed how ``str``-based ``Enum`` members are compared in
``DataFrame`` / ``Series`` operations. These tests verify that
:class:`DirType` and :class:`Split` comparisons work correctly
across both pandas 2.x and 3.x.

See: https://github.com/open-edge-platform/anomalib/issues/3303
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pandas import DataFrame

from anomalib.data.utils import DirType, LabelName, Split


class TestStrEnumDataFrameCompat:
    """Verify str-based Enum comparisons in DataFrame operations."""

    @staticmethod
    def _make_samples() -> DataFrame:
        """Create a minimal samples DataFrame matching make_folder_dataset output."""
        return DataFrame(
            {
                "image_path": ["a.png", "b.png", "c.png", "d.png"],
                "label": [DirType.NORMAL, DirType.NORMAL, DirType.ABNORMAL, DirType.NORMAL_TEST],
            },
        )

    def test_dirtype_equality_with_value(self) -> None:
        """DirType.value comparison must match the correct rows."""
        samples = self._make_samples()
        mask = samples.label == DirType.NORMAL.value
        assert mask.sum() == 2, f"Expected 2 NORMAL matches, got {mask.sum()}"

    def test_dirtype_abnormal_equality(self) -> None:
        """DirType.ABNORMAL.value comparison must match abnormal rows."""
        samples = self._make_samples()
        mask = samples.label == DirType.ABNORMAL.value
        assert mask.sum() == 1

    def test_dirtype_normal_test_equality(self) -> None:
        """DirType.NORMAL_TEST.value comparison must match normal_test rows."""
        samples = self._make_samples()
        mask = samples.label == DirType.NORMAL_TEST.value
        assert mask.sum() == 1

    def test_loc_label_index_assignment(self) -> None:
        """Label index assignment via .loc must populate all rows (no NaN)."""
        samples = self._make_samples()
        samples.loc[
            (samples.label == DirType.NORMAL.value) | (samples.label == DirType.NORMAL_TEST.value),
            "label_index",
        ] = LabelName.NORMAL
        samples.loc[(samples.label == DirType.ABNORMAL.value), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype("Int64")

        assert samples["label_index"].notna().all(), "Some label_index values are NaN"
        assert (samples["label_index"] == LabelName.NORMAL).sum() == 3
        assert (samples["label_index"] == LabelName.ABNORMAL).sum() == 1

    def test_split_assignment_and_filtering(self) -> None:
        """Split assignment and subsequent filtering must work correctly."""
        samples = self._make_samples()
        samples.loc[(samples.label == DirType.NORMAL.value), "split"] = Split.TRAIN
        samples.loc[
            (samples.label == DirType.ABNORMAL.value) | (samples.label == DirType.NORMAL_TEST.value),
            "split",
        ] = Split.TEST

        assert samples["split"].notna().all()

        # Filter by split value
        train = samples[samples.split == Split.TRAIN.value]
        test = samples[samples.split == Split.TEST.value]
        assert len(train) == 2
        assert len(test) == 2

    def test_split_enum_param_filtering(self) -> None:
        """Filtering with Split enum parameter must work via .value fallback."""
        samples = self._make_samples()
        samples["split"] = [Split.TRAIN, Split.TRAIN, Split.TEST, Split.TEST]

        # Simulate the fixed pattern: isinstance check + .value
        split = Split.TRAIN
        split_value = split.value if isinstance(split, Split) else split
        result = samples[samples.split == split_value]
        assert len(result) == 2

    @staticmethod
    def test_row_filtering_excludes_mask_rows() -> None:
        """Row filtering must correctly exclude non-data rows like MASK."""
        samples = DataFrame(
            {
                "image_path": ["a.png", "b.png", "c.png", "mask.png"],
                "label": [DirType.NORMAL, DirType.ABNORMAL, DirType.NORMAL_TEST, DirType.MASK],
            },
        )
        filtered = samples.loc[
            (samples.label == DirType.NORMAL.value)
            | (samples.label == DirType.ABNORMAL.value)
            | (samples.label == DirType.NORMAL_TEST.value)
        ]
        assert len(filtered) == 3
        assert DirType.MASK.value not in filtered.label.to_numpy()

    def test_dict_map_still_works(self) -> None:
        """Dict-based .map() with DirType keys must remain functional."""
        samples = self._make_samples()
        label_to_index = {
            DirType.ABNORMAL: LabelName.ABNORMAL,
            DirType.NORMAL: LabelName.NORMAL,
            DirType.NORMAL_TEST: LabelName.NORMAL,
        }
        mapped = samples["label"].map(label_to_index)
        assert mapped.notna().all(), "Dict .map() with enum keys returned NaN values"


class TestMakeFolderDatasetCompat:
    """Integration test: make_folder_dataset with real temp directory."""

    @staticmethod
    def test_make_folder_dataset_creates_valid_dataframe() -> None:
        """make_folder_dataset must produce correct label_index and split columns."""
        from anomalib.data.datasets.image.folder import make_folder_dataset

        with TemporaryDirectory() as tmpdir:
            normal = Path(tmpdir) / "good"
            abnormal = Path(tmpdir) / "bad"
            normal.mkdir()
            abnormal.mkdir()
            for i in range(3):
                (normal / f"n{i:03d}.png").touch()
            for i in range(2):
                (abnormal / f"a{i:03d}.png").touch()

            samples = make_folder_dataset(normal_dir=normal, abnormal_dir=abnormal)

            assert "label_index" in samples.columns
            assert "split" in samples.columns
            assert samples["label_index"].notna().all()
            assert samples["split"].notna().all()
            assert len(samples) == 5

    @staticmethod
    def test_make_folder_dataset_split_filter() -> None:
        """make_folder_dataset must correctly filter by split parameter."""
        from anomalib.data.datasets.image.folder import make_folder_dataset

        with TemporaryDirectory() as tmpdir:
            normal = Path(tmpdir) / "good"
            abnormal = Path(tmpdir) / "bad"
            normal.mkdir()
            abnormal.mkdir()
            for i in range(3):
                (normal / f"n{i:03d}.png").touch()
            for i in range(2):
                (abnormal / f"a{i:03d}.png").touch()

            train = make_folder_dataset(normal_dir=normal, abnormal_dir=abnormal, split=Split.TRAIN)
            test = make_folder_dataset(normal_dir=normal, abnormal_dir=abnormal, split=Split.TEST)

            assert len(train) == 3
            assert len(test) == 2
