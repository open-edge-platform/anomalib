# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Classes used to store ensemble predictions."""

from anomalib.data import ImageBatch


class EnsemblePredictions:
    """Basic implementation of EnsemblePredictionData that keeps all predictions in main memory."""

    def __init__(self) -> None:
        super().__init__()
        self.all_data: dict[tuple[int, int], list] = {}

    def add_tile_prediction(self, tile_index: tuple[int, int], tile_prediction: list[ImageBatch]) -> None:
        """Add tile prediction data at provided index to class dictionary in main memory.

        Args:
            tile_index (tuple[int, int]): Index of tile that we are adding in form (row, column).
            tile_prediction (list[dict[str, Tensor | list]]):
                List of batches containing all predicted data for current tile position.

        """
        self.num_batches = len(tile_prediction)

        self.all_data[tile_index] = tile_prediction

    def get_batch_tiles(self, batch_index: int) -> dict[tuple[int, int], ImageBatch]:
        """Get all tiles of current batch from class dictionary.

        Called by merging mechanism.

        Args:
            batch_index (int): Index of current batch of tiles to be returned.

        Returns:
            dict[tuple[int, int], dict]: Dictionary mapping tile index to predicted data, for provided batch index.
        """
        batch_data = {}

        for index, batches in self.all_data.items():
            batch_data[index] = batches[batch_index]

        return batch_data
