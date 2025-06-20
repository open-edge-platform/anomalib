# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class used as mechanism to merge ensemble predictions from each tile into complete whole-image representation."""

import torch
from torch import Tensor

from anomalib.data import ImageBatch

from .ensemble_tiling import EnsembleTiler
from .prediction_data import EnsemblePredictions


class PredictionMergingMechanism:
    """Class used for merging the data predicted by each separate model of tiled ensemble.

    Tiles are stacked in one tensor and untiled using Ensemble Tiler.
    Boxes from tiles are either stacked or generated anew from anomaly map.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image.
    Scores are averaged across all tiles.

    Args:
        ensemble_predictions (EnsemblePredictions): Object containing predictions on tile level.
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    Example:
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> data = EnsemblePredictions()
        >>> merger = PredictionMergingMechanism(data, tiler)
        >>>
        >>> # we can then start merging procedure for each batch
        >>> merger.merge_tile_predictions(0)
    """

    def __init__(self, ensemble_predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0,
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

        self.tiler = tiler

    def merge_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """Merge tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to merge.

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        first_tiles = getattr(batch_data[0, 0], tile_key)
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        single_channel = False
        if len(first_tiles.shape) == 3:
            single_channel = True
            # in some cases, we don't have channels but just B, H, W
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]
        else:
            # some tiles also have channels
            num_channels = first_tiles.shape[1]
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                int(num_channels),
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]

        # create new empty tensor for merged tiles
        merge_buffer = torch.zeros(size=merged_size, device=device)

        # insert tile into merged tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            merge_buffer[tile_i, tile_j, ...] = getattr(tile_data, tile_key)

        if single_channel:
            # add channel as tiler needs it
            merge_buffer = merge_buffer.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        merged_output = self.tiler.untile(merge_buffer)

        if single_channel:
            # remove previously added channels
            merged_output = merged_output.squeeze(1)

        return merged_output

    def merge_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """Join scores and their corresponding label predictions from all tiles for each image.

        Label merging is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        # create accumulator with same shape as original
        labels = torch.zeros(batch_data[0, 0].pred_label.shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[0, 0].pred_score.shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data.pred_label
            curr_scores = curr_tile_data.pred_score

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_tiles

        return {"pred_label": labels, "pred_score": scores}

    def merge_tile_predictions(self, batch_index: int) -> ImageBatch:
        """Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of merged predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        merged_predictions = {
            "image_path": current_batch_data[0, 0].image_path,
            "gt_label": current_batch_data[0, 0].gt_label,
        }
        if hasattr(current_batch_data[0, 0], "mask_path"):
            merged_predictions["mask_path"] = current_batch_data[0, 0].mask_path

        tiled_data = ["image", "gt_mask"]
        if hasattr(current_batch_data[0, 0], "anomaly_map") and current_batch_data[0, 0].anomaly_map is not None:
            tiled_data += ["anomaly_map", "pred_mask"]

        # merge all tiled data
        for t_key in tiled_data:
            if hasattr(current_batch_data[0, 0], t_key):
                merged_predictions[t_key] = self.merge_tiles(current_batch_data, t_key)

        # label and score merging
        merged_scores_and_labels = self.merge_labels_and_scores(current_batch_data)
        merged_predictions["pred_label"] = merged_scores_and_labels["pred_label"]
        merged_predictions["pred_score"] = merged_scores_and_labels["pred_score"]

        return ImageBatch(**merged_predictions)
