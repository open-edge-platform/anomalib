# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalibDataFrame — thin wrapper around ``pl.DataFrame`` with metadata support.

Polars DataFrames do not natively support an ``attrs`` dictionary (as pandas
does).  This wrapper stores arbitrary metadata alongside the data so that the
existing ``samples.attrs["task"]`` pattern keeps working without changes to
every downstream consumer.

Usage::

    >>> import polars as pl
    >>> from anomalib.data.utils.dataframe import AnomalibDataFrame
    >>> df = AnomalibDataFrame({"image_path": ["a.png"], "split": ["train"]})
    >>> df.attrs["task"] = "segmentation"
    >>> df.attrs["task"]
    'segmentation'

All regular Polars DataFrame operations (``filter``, ``sort``, ``select``,
``with_columns``, …) are available through transparent delegation and always
return a new ``AnomalibDataFrame`` that preserves the metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class AnomalibDataFrame:
    """Lightweight wrapper around :class:`polars.DataFrame` with an ``attrs`` dict.

    The wrapper delegates every attribute / method look-up to the inner
    ``pl.DataFrame`` via ``__getattr__``.  Methods that return a new
    ``pl.DataFrame`` are automatically re-wrapped so that ``attrs`` is
    preserved through chained operations.

    Parameters
    ----------
    data : pl.DataFrame | dict | list | None
        Data to construct the frame from.  Accepts anything that the
        ``pl.DataFrame`` constructor accepts, or an existing ``pl.DataFrame``
        instance.
    schema : dict | Sequence | None
        Column name → dtype mapping (forwarded to ``pl.DataFrame``).
    orient : str | None
        Row / column orientation (forwarded to ``pl.DataFrame``).
    attrs : dict | None
        Initial metadata dictionary.  Defaults to an empty ``dict``.
    """

    __hash__ = None  # type: ignore[assignment]  # Unhashable (mutable container).

    # ---- construction --------------------------------------------------------

    def __init__(
        self,
        data: pl.DataFrame | dict | list[dict] | list[tuple] | None = None,
        *,
        schema: dict[str, pl.DataType] | Sequence[str] | None = None,
        orient: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(data, AnomalibDataFrame):
            object.__setattr__(self, "_df", data.df)
            object.__setattr__(self, "attrs", attrs if attrs is not None else data.attrs.copy())
        elif isinstance(data, pl.DataFrame):
            object.__setattr__(self, "_df", data)
            object.__setattr__(self, "attrs", attrs if attrs is not None else {})
        elif data is not None:
            kwargs: dict[str, Any] = {}
            if schema is not None:
                kwargs["schema"] = schema
            if orient is not None:
                kwargs["orient"] = orient
            object.__setattr__(self, "_df", pl.DataFrame(data, **kwargs))
            object.__setattr__(self, "attrs", attrs if attrs is not None else {})
        else:
            object.__setattr__(self, "_df", pl.DataFrame())
            object.__setattr__(self, "attrs", attrs if attrs is not None else {})

    # ---- access to the inner DataFrame ---------------------------------------

    @property
    def df(self) -> pl.DataFrame:
        """Return the underlying :class:`polars.DataFrame`."""
        return self._df

    # ---- dunder protocols ----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._df.height

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return f"AnomalibDataFrame(attrs={self.attrs})\n{self._df!r}"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"AnomalibDataFrame(attrs={self.attrs})\n{self._df!s}"

    def __bool__(self) -> bool:
        """Return ``True`` when the frame is non-empty."""
        return len(self._df) > 0

    def __iter__(self) -> Iterator[pl.Series]:
        """Iterate over columns as :class:`polars.Series`."""
        return iter(self._df)

    def __contains__(self, item: str) -> bool:
        """Check whether *item* is a column name."""
        return item in self._df.columns

    def __eq__(self, other: object) -> pl.DataFrame:
        """Element-wise equality; returns a :class:`polars.DataFrame` of booleans."""
        if isinstance(other, AnomalibDataFrame):
            return self._df.__eq__(other.df)
        return self._df.__eq__(other)

    # ---- item access ---------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        """Index into the underlying frame.

        Supports the same indexing as ``pl.DataFrame.__getitem__``.  If the
        result is a ``pl.DataFrame`` it is re-wrapped to preserve attrs.
        """
        result = self._df.__getitem__(key)
        if isinstance(result, pl.DataFrame):
            return AnomalibDataFrame(result, attrs=self.attrs.copy())
        return result

    # ---- attribute delegation ------------------------------------------------

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate attribute access to the inner ``pl.DataFrame``.

        Callable attributes (methods) are wrapped so that if they return a
        ``pl.DataFrame`` the result is automatically re-wrapped.
        """
        attr = getattr(self._df, name)
        if callable(attr):

            def _wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                result = attr(*args, **kwargs)
                if isinstance(result, pl.DataFrame):
                    return AnomalibDataFrame(result, attrs=self.attrs.copy())
                return result

            return _wrapper
        return attr

    # ---- concatenation -------------------------------------------------------

    @staticmethod
    def concat(
        frames: Sequence[AnomalibDataFrame | pl.DataFrame],
        *,
        how: str = "vertical",
        **kwargs: Any,  # noqa: ANN401
    ) -> AnomalibDataFrame:
        """Concatenate multiple (Anomalib)DataFrames, preserving ``attrs``.

        Parameters
        ----------
        frames : sequence of AnomalibDataFrame | pl.DataFrame
            Frames to concatenate.
        how : str
            Concatenation strategy (forwarded to ``pl.concat``).
        **kwargs
            Extra keyword arguments forwarded to ``pl.concat``.

        Returns:
        -------
        AnomalibDataFrame
            Concatenated frame whose ``attrs`` is the merged union of all
            input ``attrs`` dictionaries (later frames win on key conflicts).
        """
        raw: list[pl.DataFrame] = []
        merged_attrs: dict[str, Any] = {}
        for frame in frames:
            if isinstance(frame, AnomalibDataFrame):
                raw.append(frame.df)
                merged_attrs.update(frame.attrs)
            else:
                raw.append(frame)
        return AnomalibDataFrame(pl.concat(raw, how=how, **kwargs), attrs=merged_attrs)

    # ---- convenience helpers used across anomalib ----------------------------

    def row_as_dict(self, index: int) -> dict[str, Any]:
        """Return row *index* as a plain ``dict``.

        This replaces the pandas ``df.iloc[index].attr`` pattern.
        """
        return self._df.row(index, named=True)
