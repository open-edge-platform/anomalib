# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AnomalibDataFrame wrapper."""

import polars as pl
import pytest
from polars.exceptions import OutOfBoundsError

from anomalib.data.utils.dataframe import AnomalibDataFrame


@pytest.fixture()
def sample_df() -> AnomalibDataFrame:
    """Return a small AnomalibDataFrame with pre-set attrs."""
    return AnomalibDataFrame(
        {"image_path": ["a.png", "b.png", "c.png"], "label": ["good", "bad", "good"]},
        attrs={"task": "segmentation", "split": "train"},
    )


# ---------- construction -----------------------------------------------------


class TestConstruction:
    """Tests for ``AnomalibDataFrame.__init__``."""

    @staticmethod
    def test_from_dict() -> None:
        """Construct from a plain dict."""
        df = AnomalibDataFrame({"col": [1, 2, 3]})
        assert len(df) == 3
        assert df.attrs == {}

    @staticmethod
    def test_from_polars_dataframe() -> None:
        """Construct from an existing ``pl.DataFrame``."""
        raw = pl.DataFrame({"x": [10, 20]})
        df = AnomalibDataFrame(raw, attrs={"key": "value"})
        assert len(df) == 2
        assert df.attrs == {"key": "value"}

    @staticmethod
    def test_from_anomalib_dataframe() -> None:
        """Construct from another ``AnomalibDataFrame`` — attrs are copied."""
        original = AnomalibDataFrame({"a": [1]}, attrs={"task": "classification"})
        copy = AnomalibDataFrame(original)
        assert copy.attrs == {"task": "classification"}
        # Mutation of the copy must not affect the original.
        copy.attrs["task"] = "detection"
        assert original.attrs["task"] == "classification"

    @staticmethod
    def test_from_none() -> None:
        """Construct an empty frame from ``None``."""
        df = AnomalibDataFrame(None)
        assert len(df) == 0
        assert df.attrs == {}

    @staticmethod
    def test_from_list_of_dicts() -> None:
        """Construct from a list of row dicts."""
        df = AnomalibDataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert len(df) == 2
        assert "a" in df
        assert "b" in df

    @staticmethod
    def test_attrs_default_to_empty_dict() -> None:
        """When no ``attrs`` are passed, an empty dict is used."""
        df = AnomalibDataFrame({"x": [1]})
        assert df.attrs == {}
        assert isinstance(df.attrs, dict)

    @staticmethod
    def test_schema_forwarded() -> None:
        """The ``schema`` kwarg is forwarded to ``pl.DataFrame``."""
        df = AnomalibDataFrame({"x": [1, 2]}, schema={"x": pl.Float64})
        assert df.df.schema["x"] == pl.Float64


# ---------- dunder protocols --------------------------------------------------


class TestDunderProtocols:
    """Tests for special methods (``__len__``, ``__repr__``, etc.)."""

    @staticmethod
    def test_len(sample_df: AnomalibDataFrame) -> None:
        """``len()`` returns the number of rows."""
        assert len(sample_df) == 3

    @staticmethod
    def test_bool_nonempty(sample_df: AnomalibDataFrame) -> None:
        """Non-empty frame is truthy."""
        assert bool(sample_df) is True

    @staticmethod
    def test_bool_empty() -> None:
        """Empty frame is falsy."""
        assert bool(AnomalibDataFrame(None)) is False

    @staticmethod
    def test_contains(sample_df: AnomalibDataFrame) -> None:
        """``in`` operator checks column names."""
        assert "image_path" in sample_df
        assert "nonexistent" not in sample_df

    @staticmethod
    def test_iter(sample_df: AnomalibDataFrame) -> None:
        """Iterating yields ``pl.Series`` columns."""
        cols = list(sample_df)
        assert all(isinstance(c, pl.Series) for c in cols)
        assert len(cols) == 2

    @staticmethod
    def test_repr(sample_df: AnomalibDataFrame) -> None:
        """``repr`` includes attrs info."""
        r = repr(sample_df)
        assert "AnomalibDataFrame" in r
        assert "segmentation" in r

    @staticmethod
    def test_str(sample_df: AnomalibDataFrame) -> None:
        """``str`` includes attrs info."""
        s = str(sample_df)
        assert "AnomalibDataFrame" in s

    @staticmethod
    def test_eq_with_anomalib_dataframe() -> None:
        """Element-wise equality between two ``AnomalibDataFrame`` instances."""
        df1 = AnomalibDataFrame({"x": [1, 2, 3]})
        df2 = AnomalibDataFrame({"x": [1, 0, 3]})
        result = df1 == df2
        assert isinstance(result, pl.DataFrame)
        assert result["x"].to_list() == [True, False, True]

    @staticmethod
    def test_eq_with_scalar() -> None:
        """Element-wise equality with a scalar."""
        df = AnomalibDataFrame({"x": [1, 2, 1]})
        result = df == 1
        assert isinstance(result, pl.DataFrame)
        assert result["x"].to_list() == [True, False, True]

    @staticmethod
    def test_hash_is_none() -> None:
        """``AnomalibDataFrame`` must be unhashable."""
        df = AnomalibDataFrame({"x": [1]})
        with pytest.raises(TypeError):
            hash(df)


# ---------- __getitem__ -------------------------------------------------------


class TestGetItem:
    """Tests for ``__getitem__`` indexing behaviour."""

    @staticmethod
    def test_single_column_returns_series(sample_df: AnomalibDataFrame) -> None:
        """Selecting one column returns a ``pl.Series``."""
        result = sample_df["image_path"]
        assert isinstance(result, pl.Series)

    @staticmethod
    def test_multi_column_returns_anomalib_df(sample_df: AnomalibDataFrame) -> None:
        """Selecting multiple columns returns an ``AnomalibDataFrame`` with attrs."""
        result = sample_df[["image_path", "label"]]
        assert isinstance(result, AnomalibDataFrame)
        assert result.attrs == sample_df.attrs

    @staticmethod
    def test_row_slice_returns_anomalib_df(sample_df: AnomalibDataFrame) -> None:
        """Slicing rows returns an ``AnomalibDataFrame`` with attrs."""
        result = sample_df[:2]
        assert isinstance(result, AnomalibDataFrame)
        assert len(result) == 2
        assert result.attrs == sample_df.attrs


# ---------- attrs propagation through delegated methods -----------------------


class TestAttrsPropagation:
    """Ensure ``attrs`` survive common Polars operations."""

    @staticmethod
    def test_filter_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``filter`` returns an ``AnomalibDataFrame`` with the same attrs."""
        filtered = sample_df.filter(pl.col("label") == "good")
        assert isinstance(filtered, AnomalibDataFrame)
        assert len(filtered) == 2
        assert filtered.attrs == {"task": "segmentation", "split": "train"}

    @staticmethod
    def test_select_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``select`` returns an ``AnomalibDataFrame`` with the same attrs."""
        selected = sample_df.select("image_path")
        assert isinstance(selected, AnomalibDataFrame)
        assert selected.columns == ["image_path"]
        assert selected.attrs == sample_df.attrs

    @staticmethod
    def test_sort_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``sort`` returns an ``AnomalibDataFrame`` with the same attrs."""
        sorted_df = sample_df.sort("label")
        assert isinstance(sorted_df, AnomalibDataFrame)
        assert sorted_df.attrs == sample_df.attrs

    @staticmethod
    def test_with_columns_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``with_columns`` returns an ``AnomalibDataFrame`` with the same attrs."""
        extended = sample_df.with_columns(pl.lit(0).alias("score"))
        assert isinstance(extended, AnomalibDataFrame)
        assert "score" in extended
        assert extended.attrs == sample_df.attrs

    @staticmethod
    def test_rename_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``rename`` returns an ``AnomalibDataFrame`` with the same attrs."""
        renamed = sample_df.rename({"label": "target"})
        assert isinstance(renamed, AnomalibDataFrame)
        assert "target" in renamed
        assert renamed.attrs == sample_df.attrs

    @staticmethod
    def test_drop_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``drop`` returns an ``AnomalibDataFrame`` with the same attrs."""
        dropped = sample_df.drop("label")
        assert isinstance(dropped, AnomalibDataFrame)
        assert "label" not in dropped
        assert dropped.attrs == sample_df.attrs

    @staticmethod
    def test_head_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``head`` returns an ``AnomalibDataFrame`` with the same attrs."""
        h = sample_df.head(1)
        assert isinstance(h, AnomalibDataFrame)
        assert len(h) == 1
        assert h.attrs == sample_df.attrs

    @staticmethod
    def test_tail_preserves_attrs(sample_df: AnomalibDataFrame) -> None:
        """``tail`` returns an ``AnomalibDataFrame`` with the same attrs."""
        t = sample_df.tail(1)
        assert isinstance(t, AnomalibDataFrame)
        assert len(t) == 1
        assert t.attrs == sample_df.attrs

    @staticmethod
    def test_chained_operations_preserve_attrs(sample_df: AnomalibDataFrame) -> None:
        """Chaining multiple operations still preserves attrs."""
        result = sample_df.filter(pl.col("label") == "good").sort("image_path").head(1)
        assert isinstance(result, AnomalibDataFrame)
        assert result.attrs == sample_df.attrs

    @staticmethod
    def test_attrs_are_copied_not_shared(sample_df: AnomalibDataFrame) -> None:
        """Each derived frame gets its own attrs copy (mutation isolation)."""
        derived = sample_df.filter(pl.col("label") == "good")
        derived.attrs["extra"] = True
        assert "extra" not in sample_df.attrs


# ---------- delegated non-DataFrame returning methods -------------------------


class TestDelegatedScalarMethods:
    """Methods that return non-DataFrame values should pass through unchanged."""

    @staticmethod
    def test_columns(sample_df: AnomalibDataFrame) -> None:
        """``columns`` property is delegated."""
        assert sample_df.columns == ["image_path", "label"]

    @staticmethod
    def test_shape(sample_df: AnomalibDataFrame) -> None:
        """``shape`` property is delegated."""
        assert sample_df.shape == (3, 2)

    @staticmethod
    def test_schema(sample_df: AnomalibDataFrame) -> None:
        """``schema`` property is delegated."""
        schema = sample_df.schema
        assert "image_path" in schema
        assert "label" in schema

    @staticmethod
    def test_height(sample_df: AnomalibDataFrame) -> None:
        """``height`` property is delegated."""
        assert sample_df.height == 3

    @staticmethod
    def test_width(sample_df: AnomalibDataFrame) -> None:
        """``width`` property is delegated."""
        assert sample_df.width == 2


# ---------- concat ------------------------------------------------------------


class TestConcat:
    """Tests for ``AnomalibDataFrame.concat``."""

    @staticmethod
    def test_concat_two_anomalib_frames() -> None:
        """Concatenating two AnomalibDataFrames merges attrs and data."""
        df1 = AnomalibDataFrame({"x": [1]}, attrs={"task": "classification"})
        df2 = AnomalibDataFrame({"x": [2]}, attrs={"split": "test"})
        result = AnomalibDataFrame.concat([df1, df2])
        assert isinstance(result, AnomalibDataFrame)
        assert len(result) == 2
        assert result.attrs == {"task": "classification", "split": "test"}

    @staticmethod
    def test_concat_attrs_later_wins() -> None:
        """When attrs keys conflict, later frames win."""
        df1 = AnomalibDataFrame({"x": [1]}, attrs={"task": "classification"})
        df2 = AnomalibDataFrame({"x": [2]}, attrs={"task": "segmentation"})
        result = AnomalibDataFrame.concat([df1, df2])
        assert result.attrs["task"] == "segmentation"

    @staticmethod
    def test_concat_with_plain_polars() -> None:
        """Concatenating with plain ``pl.DataFrame`` works; they contribute no attrs."""
        adf = AnomalibDataFrame({"x": [1]}, attrs={"task": "detection"})
        raw = pl.DataFrame({"x": [2]})
        result = AnomalibDataFrame.concat([adf, raw])
        assert isinstance(result, AnomalibDataFrame)
        assert len(result) == 2
        assert result.attrs == {"task": "detection"}

    @staticmethod
    def test_concat_empty_attrs() -> None:
        """Concatenating frames with no attrs yields empty attrs."""
        df1 = AnomalibDataFrame({"x": [1]})
        df2 = AnomalibDataFrame({"x": [2]})
        result = AnomalibDataFrame.concat([df1, df2])
        assert result.attrs == {}

    @staticmethod
    def test_concat_horizontal() -> None:
        """Horizontal concatenation is supported via ``how`` kwarg."""
        df1 = AnomalibDataFrame({"a": [1, 2]}, attrs={"key": "val"})
        df2 = AnomalibDataFrame({"b": [3, 4]})
        result = AnomalibDataFrame.concat([df1, df2], how="horizontal")
        assert result.columns == ["a", "b"]
        assert result.attrs == {"key": "val"}


# ---------- row_as_dict -------------------------------------------------------


class TestRowAsDict:
    """Tests for ``row_as_dict``."""

    @staticmethod
    def test_returns_dict(sample_df: AnomalibDataFrame) -> None:
        """``row_as_dict`` returns a plain dict for the requested index."""
        row = sample_df.row_as_dict(0)
        assert isinstance(row, dict)
        assert row == {"image_path": "a.png", "label": "good"}

    @staticmethod
    def test_last_row(sample_df: AnomalibDataFrame) -> None:
        """Can access the last row by negative-free positive index."""
        row = sample_df.row_as_dict(2)
        assert row == {"image_path": "c.png", "label": "good"}

    @staticmethod
    def test_out_of_bounds(sample_df: AnomalibDataFrame) -> None:
        """Out-of-range index raises."""
        with pytest.raises(OutOfBoundsError):
            sample_df.row_as_dict(10)


# ---------- df property -------------------------------------------------------


class TestDfProperty:
    """Tests for the ``.df`` property."""

    @staticmethod
    def test_df_returns_polars_dataframe(sample_df: AnomalibDataFrame) -> None:
        """``.df`` exposes the underlying ``pl.DataFrame``."""
        assert isinstance(sample_df.df, pl.DataFrame)
        assert not isinstance(sample_df.df, AnomalibDataFrame)
