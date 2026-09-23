# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Statistical components for MH-PatchCore."""

import math

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from anomalib.models.components import DynamicBufferMixin

_EIGENVALUE_FLOOR_RATIO = 1e-8
_MIN_EIGENVALUE_SCALE = 1e-12
_CHOLESKY_JITTERS = (
    0.0,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
)


def _regularize_covariance(covariance: torch.Tensor, shrinkage: float) -> torch.Tensor:
    covariance = covariance.detach().to(device="cpu", dtype=torch.float64)
    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    mean_eigenvalue = float(torch.trace(covariance).item()) / dimension
    identity = torch.eye(dimension, dtype=covariance.dtype)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * mean_eigenvalue * identity

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = _EIGENVALUE_FLOOR_RATIO * max(abs(mean_eigenvalue), _MIN_EIGENVALUE_SCALE)
    eigenvalues = eigenvalues.clamp_min(floor)
    covariance = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def _stable_cholesky(covariance: torch.Tensor, sample_count: int) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    for attempt, jitter in enumerate(_CHOLESKY_JITTERS):
        candidate = covariance if attempt == 0 else covariance + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        if int(info.item()) == 0 and bool(torch.isfinite(factor).all()):
            return factor

    msg = (
        "Failed to compute covariance Cholesky factor: "
        f"stage=covariance_finalization, shape={tuple(covariance.shape)}, "
        f"sample_count={sample_count}, attempted_jitter={_CHOLESKY_JITTERS[-1]:.1e}."
    )
    raise RuntimeError(msg)


def _compute_whitening_matrix(cholesky_factor: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(cholesky_factor.shape[0], dtype=cholesky_factor.dtype)
    inverse_lower = torch.linalg.solve_triangular(cholesky_factor, identity, upper=False)
    return inverse_lower.T.contiguous()


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


class CovarianceWhitening(DynamicBufferMixin):
    """Fit streaming covariance and persist its whitening transform.

    Args:
        shrinkage (float): Fixed covariance shrinkage coefficient in ``[0, 1]``.
            Defaults to ``0.07``.

    Raises:
        ValueError: If ``shrinkage`` is not a finite number in ``[0, 1]``.
    """

    def __init__(self, shrinkage: float = 0.07) -> None:
        super().__init__()
        if (
            isinstance(shrinkage, bool)
            or not isinstance(shrinkage, int | float)
            or not math.isfinite(shrinkage)
            or not 0.0 <= shrinkage <= 1.0
        ):
            msg = "shrinkage must be a finite number in [0, 1]."
            raise ValueError(msg)

        self.shrinkage = float(shrinkage)
        self.register_buffer("mean", torch.empty(0, dtype=torch.float64))
        self.register_buffer("whitening_matrix", torch.empty(0, dtype=torch.float64))
        self.mean: torch.Tensor
        self.whitening_matrix: torch.Tensor

        self._sample_count = 0
        self._running_mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None
        self._feature_dimension: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the persistent whitening transform is available."""
        return bool(self.whitening_matrix.numel())

    def update(self, embeddings: torch.Tensor) -> None:
        """Add an embedding batch to the covariance stream.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Raises:
            RuntimeError: If covariance fitting was already finalized.
            ValueError: If the embeddings are empty, non-finite, not two-dimensional,
                or have an inconsistent feature dimension.
        """
        if self.is_fitted:
            msg = "CovarianceWhitening cannot be updated after finalization."
            raise RuntimeError(msg)

        batch = self._prepare_batch(embeddings)
        batch_count = len(batch)
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_m2 = centered.T @ centered

        if self._running_mean is None:
            self._sample_count = batch_count
            self._running_mean = batch_mean
            self._m2 = batch_m2
            return

        if self._m2 is None:  # pragma: no cover - internal invariant
            msg = "CovarianceWhitening second moment is unavailable during update."
            raise RuntimeError(msg)
        delta = batch_mean - self._running_mean
        combined_count = self._sample_count + batch_count
        self._running_mean += delta * (batch_count / combined_count)
        self._m2 += batch_m2 + torch.outer(delta, delta) * (self._sample_count * batch_count / combined_count)
        self._sample_count = combined_count

    def covariance(self) -> torch.Tensor:
        """Return the current unbiased sample covariance.

        Returns:
            torch.Tensor: Float64 covariance matrix with shape ``[D, D]``.

        Raises:
            RuntimeError: If fewer than two samples are available or fitting was finalized.
        """
        if self._sample_count < 2 or self._m2 is None:
            msg = "CovarianceWhitening requires at least two samples to compute covariance."
            raise RuntimeError(msg)
        return self._m2 / (self._sample_count - 1)

    def finalize(self) -> None:
        """Finalize covariance fitting and persist the whitening transform.

        Raises:
            RuntimeError: If fitting is already finalized, fewer than two samples
                were received, or the finalized state is non-finite.
        """
        if self.is_fitted:
            msg = "CovarianceWhitening is already finalized."
            raise RuntimeError(msg)
        if self._sample_count == 0:
            msg = "CovarianceWhitening received no embedding batches."
            raise RuntimeError(msg)
        if self._sample_count == 1:
            msg = "CovarianceWhitening requires at least two samples to finalize."
            raise RuntimeError(msg)
        if self._running_mean is None:  # pragma: no cover - internal invariant
            msg = "CovarianceWhitening mean is unavailable during finalization."
            raise RuntimeError(msg)

        sample_count = self._sample_count
        covariance = self.covariance()
        regularized_covariance = _regularize_covariance(covariance, self.shrinkage)
        cholesky_factor = _stable_cholesky(regularized_covariance, sample_count)
        whitening_matrix = _compute_whitening_matrix(cholesky_factor)
        if not torch.isfinite(self._running_mean).all() or not torch.isfinite(whitening_matrix).all():
            msg = "CovarianceWhitening produced non-finite fitted state."
            raise RuntimeError(msg)

        self.mean = self._running_mean.clone()
        self.whitening_matrix = whitening_matrix
        self._sample_count = 0
        self._running_mean = None
        self._m2 = None
        self._feature_dimension = None

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Whiten embeddings using the finalized covariance state.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Returns:
            torch.Tensor: Whitened float64 embeddings with shape ``[N, D]``.

        Raises:
            RuntimeError: If covariance fitting has not been finalized or the
                transformation produces non-finite values.
            ValueError: If the embeddings are empty, non-finite, not two-dimensional,
                or have an incompatible feature dimension.
        """
        if not self.is_fitted:
            msg = "CovarianceWhitening must be finalized before transform()."
            raise RuntimeError(msg)
        if embeddings.ndim != 2 or len(embeddings) == 0 or embeddings.shape[1] == 0:
            msg = "embeddings must be a non-empty two-dimensional tensor."
            raise ValueError(msg)
        if embeddings.shape[1] != self.mean.shape[0]:
            msg = (
                f"embeddings have feature dimension {embeddings.shape[1]}, "
                f"but fitted covariance expects {self.mean.shape[0]}."
            )
            raise ValueError(msg)
        if not torch.isfinite(embeddings).all():
            msg = "embeddings must contain only finite values."
            raise ValueError(msg)

        embeddings = embeddings.to(device=self.mean.device, dtype=torch.float64)
        whitened = (embeddings - self.mean) @ self.whitening_matrix
        if not torch.isfinite(whitened).all():
            msg = "CovarianceWhitening produced non-finite transformed embeddings."
            raise RuntimeError(msg)
        return whitened

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Whiten embeddings using the finalized covariance state.

        Args:
            embeddings (torch.Tensor): Embeddings with shape ``[N, D]``.

        Returns:
            torch.Tensor: Whitened float64 embeddings with shape ``[N, D]``.
        """
        return self.transform(embeddings)

    def _prepare_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or len(embeddings) == 0 or embeddings.shape[1] == 0:
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
        return embeddings.detach().to(device="cpu", dtype=torch.float64).contiguous()
