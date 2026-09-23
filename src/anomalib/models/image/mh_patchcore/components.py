# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Statistical components for MH-PatchCore."""

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from anomalib.models.components import DynamicBufferMixin


class StreamingPCA(DynamicBufferMixin):
    """Fit incremental PCA and persist its inference projection in Torch.

    Args:
        variance_ratio (float): Fraction of explained variance to retain.
            Defaults to ``0.99``.

    Raises:
        ValueError: If ``variance_ratio`` is not in ``(0, 1]``.
    """

    def __init__(self, variance_ratio: float = 0.99) -> None:
        super().__init__()
        if isinstance(variance_ratio, bool) or not 0.0 < variance_ratio <= 1.0:
            msg = "variance_ratio must be in (0, 1]."
            raise ValueError(msg)

        self.variance_ratio = float(variance_ratio)
        self.register_buffer("components", torch.empty(0, dtype=torch.float32))
        self.register_buffer("mean", torch.empty(0, dtype=torch.float64))
        self.register_buffer("projected_mean", torch.empty(0, dtype=torch.float32))
        self.register_buffer("num_components", torch.tensor(0, dtype=torch.int64))
        self.components: torch.Tensor
        self.mean: torch.Tensor
        self.projected_mean: torch.Tensor
        self.num_components: torch.Tensor

        self._estimator: IncrementalPCA | None = None
        self._carry: torch.Tensor | None = None
        self._pending_batch: torch.Tensor | None = None
        self._feature_dimension: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the persistent PCA projection is available."""
        return bool(self.num_components.item())

    def update(self, embeddings: torch.Tensor) -> None:
        """Add an embedding batch to the incremental PCA stream.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Raises:
            RuntimeError: If PCA was already finalized.
            ValueError: If the embeddings are empty, non-finite, not two-dimensional,
                or have an inconsistent feature dimension.
        """
        if self.is_fitted:
            msg = "StreamingPCA cannot be updated after finalization."
            raise RuntimeError(msg)

        batch = self._prepare_batch(embeddings)
        if self._carry is not None:
            batch = torch.cat((self._carry, batch), dim=0)
            self._carry = None

        if len(batch) < batch.shape[1]:
            self._carry = batch
            return

        if self._pending_batch is not None:
            self._partial_fit(self._pending_batch)
        self._pending_batch = batch

    def finalize(self) -> None:
        """Finalize fitting and convert the estimator to persistent Torch buffers.

        Raises:
            RuntimeError: If PCA is already finalized, received no samples, or
                produced non-finite fitted state.
        """
        if self.is_fitted:
            msg = "StreamingPCA is already finalized."
            raise RuntimeError(msg)
        if self._pending_batch is None and self._carry is None:
            msg = "StreamingPCA received no embedding batches."
            raise RuntimeError(msg)

        final_batch = self._pending_batch
        if final_batch is None:
            final_batch = self._carry
        elif self._carry is not None:
            final_batch = torch.cat((final_batch, self._carry), dim=0)
        if final_batch is None:  # pragma: no cover - guarded above
            msg = "StreamingPCA received no embedding batches."
            raise RuntimeError(msg)
        self._partial_fit(final_batch)

        estimator = self._estimator
        if estimator is None:  # pragma: no cover - internal invariant
            msg = "StreamingPCA estimator is unavailable during finalization."
            raise RuntimeError(msg)
        explained_variance_ratio = estimator.explained_variance_ratio_
        if not np.isfinite(explained_variance_ratio).all():
            msg = "StreamingPCA produced non-finite explained variance."
            raise RuntimeError(msg)

        cumulative_variance = np.cumsum(explained_variance_ratio)
        matching_components = np.flatnonzero(cumulative_variance >= self.variance_ratio)
        retained_components = int(matching_components[0] + 1) if matching_components.size else len(cumulative_variance)
        fitted_components = estimator.components_[:retained_components]
        fitted_mean = estimator.mean_
        projected_mean = fitted_mean @ fitted_components.T
        if (
            not np.isfinite(fitted_components).all()
            or not np.isfinite(fitted_mean).all()
            or not np.isfinite(projected_mean).all()
        ):
            msg = "StreamingPCA produced non-finite fitted state."
            raise RuntimeError(msg)

        self.components = torch.from_numpy(fitted_components.astype(np.float32, copy=True))
        self.mean = torch.from_numpy(fitted_mean.astype(np.float64, copy=True))
        self.projected_mean = torch.from_numpy(projected_mean.astype(np.float32, copy=True))
        self.num_components = torch.tensor(retained_components, dtype=torch.int64)
        self._estimator = None
        self._carry = None
        self._pending_batch = None

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings using the finalized Torch PCA state.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Returns:
            torch.Tensor: Projected float32 embeddings with shape ``[N, K]``.

        Raises:
            RuntimeError: If PCA has not been finalized.
            ValueError: If the embeddings are empty, non-finite, not two-dimensional,
                or have an incompatible feature dimension.
        """
        if not self.is_fitted:
            msg = "StreamingPCA must be finalized before transform()."
            raise RuntimeError(msg)
        if embeddings.ndim != 2 or len(embeddings) == 0:
            msg = "embeddings must be a non-empty two-dimensional tensor."
            raise ValueError(msg)
        if embeddings.shape[1] != self.components.shape[1]:
            msg = (
                f"embeddings have feature dimension {embeddings.shape[1]}, "
                f"but fitted PCA expects {self.components.shape[1]}."
            )
            raise ValueError(msg)
        if not torch.isfinite(embeddings).all():
            msg = "embeddings must contain only finite values."
            raise ValueError(msg)

        embeddings = embeddings.to(dtype=self.components.dtype)
        return embeddings @ self.components.T - self.projected_mean

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings using the finalized Torch PCA state.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Returns:
            torch.Tensor: Projected float32 embeddings with shape ``[N, K]``.
        """
        return self.transform(embeddings)

    def _prepare_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or len(embeddings) == 0:
            msg = "embeddings must be a non-empty two-dimensional tensor."
            raise ValueError(msg)
        if not torch.isfinite(embeddings).all():
            msg = "embeddings must contain only finite values."
            raise ValueError(msg)

        feature_dimension = int(embeddings.shape[1])
        if self._feature_dimension is None:
            self._feature_dimension = feature_dimension
        elif feature_dimension != self._feature_dimension:
            msg = (
                f"embeddings have feature dimension {feature_dimension}, "
                f"but previous batches have dimension {self._feature_dimension}."
            )
            raise ValueError(msg)
        return embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()

    def _partial_fit(self, batch: torch.Tensor) -> None:
        if self._estimator is None:
            self._estimator = IncrementalPCA(n_components=None)
        self._estimator.partial_fit(batch.numpy())
