"""
Class used as mechanism to join/combine ensemble predictions from each tile into complete image-level representation
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
from tools.tiled_ensemble.predictions.prediction_data import EnsemblePredictions
from torch import Tensor


class EnsemblePredictionJoiner:
    """
    Class used for joining/combining the data predicted by each separate model of tiled ensemble.

    Tiles are stacked in one tensor and untiled using Ensemble Tiler.
    Boxes from tiles are stacked resulting in one tensor of box coordinates and scores per image.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image.
    Scores are averaged across all tiles.

    Args:
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.prediction_data import BasicEnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = EnsemblePredictionJoiner(tiler)
        >>> data = BasicEnsemblePredictions()
        >>>
        >>> # joiner needs to be setup with ensemble predictions storage object
        >>> joiner.setup(data)
        >>>
        >>> # we can then start joining procedure for each batch
        >>> joiner.join_tile_predictions(0)
    """

    def __init__(self, tiler: EnsembleTiler) -> None:
        self.tiler = tiler

        self.ensemble_predictions: EnsemblePredictions = None
        self.num_batches = 0

    def setup(self, ensemble_predictions: EnsemblePredictions) -> None:
        """
        Prepare the joiner for given prediction data.

        Args:
            ensemble_predictions (EnsemblePredictions): Dictionary containing batched predictions for each tile.

        """
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

    def join_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to join

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        first_tiles = batch_data[(0, 0)][tile_key]
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        if tile_key == "mask":
            # in case of ground truth masks, we don't have channels
            joined_size = (
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            )
        else:
            # all tiles beside masks also have channels
            num_channels = first_tiles.shape[1]
            joined_size = (
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                num_channels,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            )

        # create new empty tensor for joined tiles
        joined_masks = torch.zeros(size=joined_size, device=device)

        # insert tile into joined tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            joined_masks[tile_i, tile_j, ...] = tile_data[tile_key]

        if tile_key == "mask":
            # add channel as tiler needs it
            joined_masks = joined_masks.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        joined_output = self.tiler.untile(joined_masks)

        if tile_key == "mask":
            # remove previously added channels
            joined_output = joined_output.squeeze(1)

        return joined_output

    def join_boxes(self, batch_data: dict) -> dict:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.

        Joining is done by stacking boxes from all tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict: Dictionary with joined boxes, box scores and box labels.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        batch_size = len(batch_data[(0, 0)]["pred_boxes"])

        # create array of placeholder arrays, that will contain all boxes for each image
        boxes: list[list[Tensor]] = [[] for _ in range(batch_size)]
        scores: list[list[Tensor]] = [[] for _ in range(batch_size)]
        labels: list[list[Tensor]] = [[] for _ in range(batch_size)]

        # go over all tiles and add box data tensor to belonging array
        for (tile_i, tile_j), curr_tile_pred in batch_data.items():
            for i in range(batch_size):
                # boxes have form [x_1, y_1, x_2, y_2]
                curr_boxes = curr_tile_pred["pred_boxes"][i]

                # tile position offset
                offset_w = self.tiler.tile_size_w * tile_j
                offset_h = self.tiler.tile_size_h * tile_i

                # offset in x-axis
                curr_boxes[:, 0] += offset_w
                curr_boxes[:, 2] += offset_w

                # offset in y-axis
                curr_boxes[:, 1] += offset_h
                curr_boxes[:, 3] += offset_h

                boxes[i].append(curr_boxes)
                scores[i].append(curr_tile_pred["box_scores"][i])
                labels[i].append(curr_tile_pred["box_labels"][i])

        # arrays with box data for each batch
        joined_boxes: dict[str, list[Tensor]] = {"pred_boxes": [], "box_scores": [], "box_labels": []}
        for i in range(batch_size):
            # n in this case represents number of predicted boxes
            # stack boxes into form [n, 4] (vertical stack)
            joined_boxes["pred_boxes"].append(torch.vstack(boxes[i]))
            # stack scores and labels into form [n] (horizontal stack)
            joined_boxes["box_scores"].append(torch.hstack(scores[i]))
            joined_boxes["box_labels"].append(torch.hstack(labels[i]))

        return joined_boxes

    def join_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """
        Join scores and their corresponding label predictions from all tiles for each image.

        Label joining is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        # create accumulator with same shape as original
        labels = torch.zeros(batch_data[(0, 0)]["pred_labels"].shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[(0, 0)]["pred_scores"].shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data["pred_labels"]
            curr_scores = curr_tile_data["pred_scores"]

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_tiles

        joined = {"pred_labels": labels, "pred_scores": scores}

        return joined

    def join_tile_predictions(self, batch_index: int) -> dict[str, Tensor | list]:
        """
        Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of joined predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        tiled_keys = ["image", "mask", "anomaly_maps", "pred_masks"]
        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        joined_predictions = {
            "image_path": current_batch_data[(0, 0)]["image_path"],
            "label": current_batch_data[(0, 0)]["label"],
        }
        if "mask_path" in current_batch_data[(0, 0)].keys():
            joined_predictions["mask_path"] = current_batch_data[(0, 0)]["mask_path"]
        if "boxes" in current_batch_data[(0, 0)].keys():
            joined_predictions["boxes"] = current_batch_data[(0, 0)]["boxes"]

            # join all box data from all tiles
            joined_box_data = self.join_boxes(current_batch_data)
            joined_predictions["pred_boxes"] = joined_box_data["pred_boxes"]
            joined_predictions["box_scores"] = joined_box_data["box_scores"]
            joined_predictions["box_labels"] = joined_box_data["box_labels"]

        # join all tiled data
        for t_key in tiled_keys:
            if t_key in current_batch_data[(0, 0)].keys():
                joined_predictions[t_key] = self.join_tiles(current_batch_data, t_key)

        # label and score joining
        joined_scores_and_labels = self.join_labels_and_scores(current_batch_data)
        joined_predictions["pred_labels"] = joined_scores_and_labels["pred_labels"]
        joined_predictions["pred_scores"] = joined_scores_and_labels["pred_scores"]

        return joined_predictions
