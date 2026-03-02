# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Area Under Per-Region Overlap (AUPRO) metric.

This module provides the ``AUPRO`` class which computes the area under the
per-region overlap curve for evaluating anomaly segmentation performance.

The AUPRO score measures how well predicted anomaly maps overlap with ground truth
anomaly regions. It is computed by:

1. Performing connected component analysis on ground truth masks
2. Computing per-region ROC curves for each component
3. Averaging the curves and computing area under the curve up to a FPR limit

Example:
    >>> from anomalib.metrics import AUPRO
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.randint(0, 2, (1, 10, 5))  # Binary segmentation masks
    >>> scores = torch.rand_like(labels)  # Anomaly segmentation maps
    >>> # Initialize and compute AUPRO
    >>> metric = AUPRO(fpr_limit=0.3)
    >>> aupro_score = metric(scores, labels)
    >>> aupro_score
    tensor(0.4321)

The metric can also be updated incrementally with batches:

    >>> for batch_scores, batch_labels in dataloader:
    ...     metric.update(batch_scores, batch_labels)
    >>> final_score = metric.compute()

Args:
    dist_sync_on_step (bool): Synchronize metric state across processes at each
        ``forward()`` before returning the value at the step.
        Defaults to ``False``.
    process_group (Any | None): Specify the process group on which
        synchronization is called. Defaults to ``None`` (entire world).
    dist_sync_fn (Callable | None): Callback that performs the allgather
        operation on the metric state. When ``None``, DDP will be used.
        Defaults to ``None``.
    fpr_limit (float): Limit for the false positive rate.
        Defaults to ``0.3``.
    num_thresholds (int | None): Number of thresholds to use for computing the
        ROC curve. When ``None``, uses thresholds from torchmetrics.
        Defaults to ``None``.

Note:
    The AUPRO score ranges from 0 to 1, with 1 indicating perfect overlap between
    predictions and ground truth regions.
"""

from collections.abc import Callable
from typing import Any

import torch
from matplotlib.figure import Figure
from torchmetrics import Metric
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.metrics.pro import connected_components_cpu, connected_components_gpu
from anomalib.utils import deprecate

from .base import AnomalibMetric
from .utils import plot_metric_curve


@deprecate(
    args={"num_thresholds": None},
    since="2.1.0",
    remove="3.0.0",
    reason="New AUPRO computation does not require number of thresholds",
)
class _AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric.

    Args:
        dist_sync_on_step (bool): Synchronize metric state across processes at
            each ``forward()`` before returning the value at the step.
            Defaults to ``False``.
        process_group (Any | None): Specify the process group on which
            synchronization is called. Defaults to ``None`` (entire world).
        dist_sync_fn (Callable | None): Callback that performs the allgather
            operation on the metric state. When ``None``, DDP will be used.
            Defaults to ``None``.
        fpr_limit (float): Limit for the false positive rate.
            Defaults to ``0.3``.
        num_thresholds (int | None): Present for backward compatibility with the
            old implementation, but ignored in this fast version.

    Examples:
        >>> import torch
        >>> from anomalib.metrics import AUPRO
        >>> # Create sample data
        >>> labels = torch.randint(0, 2, (1, 10, 5), dtype=torch.float32)
        >>> preds = torch.rand_like(labels)
        >>> # Initialize and compute
        >>> aupro = AUPRO(fpr_limit=0.3)
        >>> aupro(preds, labels)
        tensor(0.4321)

        Increasing the ``fpr_limit`` will increase the AUPRO value:

        >>> aupro = AUPRO(fpr_limit=0.7)
        >>> aupro(preds, labels)
        tensor(0.5271)
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[torch.Tensor]
    target: list[torch.Tensor]
    num_thresholds: int | None

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Any | None = None,  # noqa: ANN401
        dist_sync_fn: Callable | None = None,
        fpr_limit: float = 0.3,
        num_thresholds: int | None = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

        # Kept for API compatibility; ignored by the fast implementation.
        self.num_thresholds = num_thresholds

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def perform_cca(self) -> torch.Tensor:
        """Perform the Connected Component Analysis on the self.target tensor.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with
                requirements imposed by kornia for connected component analysis.

        Returns:
            Tensor: Components labeled from 0 to N.
        """
        target = dim_zero_cat(self.target)  # (B, ..., H, W)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            msg = (
                "kornia.contrib.connected_components expects input to lie in the "
                f"interval [0, 1], but found [{target.min()}, {target.max()}]."
            )
            raise ValueError(
                msg,
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        return connected_components_gpu(target) if target.is_cuda else connected_components_cpu(target)

    @staticmethod
    def _make_global_region_labels(cca: torch.Tensor) -> torch.Tensor:
        """Offset connected component labels across batch to make them unique.

        Args:
            cca (torch.Tensor): (B, H, W) integer labels, starting at 0 for each image.

        Returns:
            torch.Tensor: (B, H, W) labels where:
                - 0 is still background
                - positive labels are unique across the batch
        """
        # We iterate over batch dimension only; typically small.
        batch_size = cca.size(0)
        cca_off = cca.clone()
        current_offset = 0

        for b in range(batch_size):
            img_labels = cca_off[b]
            unique = img_labels.unique()
            unique_fg = unique[unique != 0]
            num_regions = int(unique_fg.numel())
            if num_regions == 0:
                continue

            # shift all foreground labels in this image by current_offset
            fg_mask = img_labels > 0
            img_labels[fg_mask] = img_labels[fg_mask] + current_offset

            cca_off[b] = img_labels
            current_offset += num_regions

        return cca_off

    @deprecate(
        args={"target": None},
        since="2.1.0",
        remove="3.0.0",
        reason="Compute PRO computes overlap with connected components not target.",
    )
    def compute_pro(
        self,
        cca: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the PRO curve (FPR vs. averaged per-region TPR/overlap).

        This implementation is inspired by the MvTec implementation found at
        https://www.mvtec.com/company/research/datasets/mvtec-ad

        Args:
            cca (torch.Tensor):
                Connected-component labels of shape (B, H, W). Must contain integers
                ≥ 0, where 0 denotes background and >0 denote region IDs.
            preds (torch.Tensor):
                Prediction scores of shape (B, H, W). Higher values indicate more
                anomalous.
            target (torch.Tensor | None):
                Unused; accepted only for API compatibility.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                (fpr, pro), both 1-D tensors sorted by increasing FPR and clipped
                at ``self.fpr_limit``.
        """
        del target
        device = preds.device

        # flatten (already on correct device)
        labels = cca.reshape(-1).long()
        preds_flat = preds.reshape(-1).float()

        # background = FPR contribution
        background = labels == 0
        fp_change = background.float()
        num_bg = fp_change.sum()

        if num_bg == 0:
            f = float(self.fpr_limit)
            return (
                torch.tensor([0.0, f], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        max_label = int(labels.max())
        if max_label == 0:
            f = float(self.fpr_limit)
            return (
                torch.tensor([0.0, f], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        region_sizes = torch.bincount(labels, minlength=max_label + 1).float()
        num_regions = (region_sizes[1:] > 0).sum()

        if num_regions == 0:
            f = float(self.fpr_limit)
            return (
                torch.tensor([0.0, f], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        fg_mask = labels > 0

        pro_change = torch.zeros_like(preds_flat)
        pro_change[fg_mask] = 1.0 / region_sizes[labels[fg_mask]]

        # global sort
        idx = torch.argsort(preds_flat, descending=True)
        fp_sorted = fp_change[idx]
        pro_sorted = pro_change[idx]
        preds_sorted = preds_flat[idx]

        # cumulative sums
        fpr = torch.cumsum(fp_sorted, 0) / num_bg
        pro = torch.cumsum(pro_sorted, 0) / num_regions
        fpr.clamp_(max=1.0)
        pro.clamp_(max=1.0)

        # remove duplicate thresholds
        keep = torch.ones_like(preds_sorted, dtype=torch.bool)
        keep[:-1] = preds_sorted[:-1] != preds_sorted[1:]
        fpr = fpr[keep]
        pro = pro[keep]

        # prepend zero
        fpr = torch.cat([torch.tensor([0.0], device=device), fpr])
        pro = torch.cat([torch.tensor([0.0], device=device), pro])

        # FPR limit clipping
        f_lim = float(self.fpr_limit)
        mask = fpr <= f_lim

        if mask.any():
            i = mask.nonzero(as_tuple=True)[0][-1].item()

            if fpr[i] < f_lim and i + 1 < fpr.numel():
                f1, f2 = fpr[i], fpr[i + 1]
                p1, p2 = pro[i], pro[i + 1]
                p_lim = p1 + (p2 - p1) * (f_lim - f1) / (f2 - f1)

                fpr = torch.cat([fpr[: i + 1], torch.tensor([f_lim], device=device)])
                pro = torch.cat([pro[: i + 1], torch.tensor([p_lim], device=device)])
            else:
                fpr = fpr[: i + 1]
                pro = pro[: i + 1]
        else:
            fpr = torch.tensor([0.0, f_lim], device=device)
            pro = torch.tensor([0.0, 0.0], device=device)

        return fpr, pro

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the PRO curve (FPR vs PRO) for all stored predictions."""
        cca = self.perform_cca()  # (B, H, W)
        preds = dim_zero_cat(self.preds)  # (B, 1, H, W) or (B, H, W)
        if preds.dim() > 3 and preds.size(1) == 1:
            preds = preds.squeeze(1)
        return self.compute_pro(cca=cca, preds=preds)

    def compute(self) -> torch.Tensor:
        """First compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, tpr = self._compute()

        aupro = auc(fpr, tpr, reorder=True)
        return aupro / fpr[-1]  # normalize the area

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the PRO curve and the AUPRO.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure
                title to be used for logging
        """
        fpr, tpr = self._compute()
        aupro = self.compute()

        xlim = (0.0, float(self.fpr_limit.detach().cpu().item()))
        ylim = (0.0, 1.0)
        xlabel = "Global FPR"
        ylabel = "Averaged Per-Region TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_metric_curve(fpr, tpr, aupro, xlim, ylim, xlabel, ylabel, loc, title, metric_name="AUPRO")

        return fig, title

    @staticmethod
    def interp1d(old_x: torch.Tensor, old_y: torch.Tensor, new_x: torch.Tensor) -> torch.Tensor:
        """Interpolate a 1D signal linearly to new sampling points.

        Args:
            old_x (torch.Tensor): original 1-D x values (same size as y)
            old_y (torch.Tensor): original 1-D y values (same size as x)
            new_x (torch.Tensor): x-values where y should be interpolated at

        Returns:
            Tensor: y-values at corresponding new_x values.
        """
        # Compute slope
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # Prepare idx for linear interpolation
        idx = torch.searchsorted(old_x, new_x)

        # searchsorted looks for the index where the values must be inserted
        # to preserve order, but we actually want the preceeding index.
        idx -= 1
        # we clamp the index, because the number of intervals = old_x.size(0) -1,
        # and the left neighbour should hence be at most number of intervals -1,
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # perform actual linear interpolation
        return old_y[idx] + slope[idx] * (new_x - old_x[idx])


class AUPRO(AnomalibMetric, _AUPRO):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUPRO metric."""
