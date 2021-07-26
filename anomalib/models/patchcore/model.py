"""
Towards Total Recall in Industrial Anomaly Detection
https://arxiv.org/abs/2106.08265
"""

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.ndimage import gaussian_filter
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torch import Tensor

from anomalib.core.callbacks.model_loader import LoadModelCallback
from anomalib.core.model.dynamic_module import DynamicBufferModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.datasets.utils import Denormalize
from anomalib.models.base.model import BaseAnomalySegmentationLightning
from anomalib.models.padim.model import concat_layer_embedding
from anomalib.models.patchcore.sampling_methods.kcenter_greedy import kCenterGreedy
from anomalib.utils.visualizer import Visualizer


class Callbacks:
    """PADIM-specific callbacks"""

    def __init__(self, config: DictConfig):
        self.config = config

    def get_callbacks(self) -> Sequence:
        """Get PADIM model callbacks."""
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.config.project.path, "weights"),
            filename="model",
        )
        callbacks = [checkpoint]

        # TODO: Check if we load the model properly: https://jira.devtools.intel.com/browse/IAAALD-13
        if "weight_file" in self.config.model.keys():
            model_loader = LoadModelCallback(
                weights_path=os.path.join(self.config.project.path, "weights", self.config.model.weight_file)
            )
            callbacks.append(model_loader)

        return callbacks

    def __call__(self):
        return self.get_callbacks()


class AnomalyMapGenerator(BaseAnomalyMapGenerator):
    """
    Generate Anomaly Heatmap
    """

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        alpha: float = 0.4,
        gamma: int = 0,
        sigma: int = 4,
    ):
        super().__init__(input_size=input_size, alpha=alpha, gamma=gamma, sigma=sigma)

    def compute_anomaly_map(self, score_patches: np.ndarray) -> np.ndarray:
        """
        Pixel Level Anomaly Heatmap

        Args:
            score_patches (np.ndarray): [description]
        """
        anomaly_map = score_patches[:, 0].reshape((28, 28))
        anomaly_map = cv2.resize(anomaly_map, self.input_size)
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.sigma)

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: np.ndarray) -> np.ndarray:
        """
        Compute Image-Level Anomaly Score

        Args:
            patch_scores (np.ndarray): [description]
        """
        confidence = patch_scores[np.argmax(patch_scores[:, 0])]
        weights = 1 - (np.max(np.exp(confidence)) / np.sum(np.exp(confidence)))
        score = weights * max(patch_scores[:, 0])
        return score

    def __call__(self, patch_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        anomaly_map = self.compute_anomaly_map(patch_scores)
        anomaly_score = self.compute_anomaly_score(patch_scores)

        return anomaly_map, anomaly_score


class PatchcoreModel(DynamicBufferModule):
    """
    Padim Module
    """

    def __init__(self, backbone: str, layers: List[str], input_size: Union[ListConfig, Tuple]):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.input_size = input_size

        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.nn_search = NearestNeighbors(n_neighbors=9)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        # TODO: Define memory_bank here: https://jira.devtools.intel.com/browse/IAAALD-13
        self.register_buffer("memory_bank", torch.Tensor())

    def forward(self, input_tensor: Tensor) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get features from a CNN.
        Generate embedding based on the feautures.
        Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.training:
            output = embedding
        else:
            patch_scores, _ = self.nn_search.kneighbors(embedding)

            anomaly_map, anomaly_score = self.anomaly_map_generator(patch_scores)
            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> np.ndarray:
        """
        Generate embedding from hierarchical feature map

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector

        """

        layer_embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embeddings = concat_layer_embedding(layer_embeddings, features[layer])

        embedding = self.reshape_embedding(layer_embeddings).cpu().numpy()
        return embedding

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """
        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    @staticmethod
    def subsample_embedding(embedding: np.ndarray, sampling_ratio: float) -> np.ndarray:
        """
        Subsample embedding based on coreset sampling

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio

        Returns:
            np.ndarray: Subsampled embedding whose dimensionality is reduced.
        """

        # Random projection
        random_projector = SparseRandomProjection(n_components="auto", eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        random_projector.fit(embedding)

        # Coreset Subsampling
        selector = kCenterGreedy(embedding, 0, 0)
        selected_idx = selector.select_batch(
            model=random_projector,
            already_selected=[],
            N=int(embedding.shape[0] * sampling_ratio),
        )
        embedding_coreset = embedding[selected_idx]
        return embedding_coreset


class PatchcoreLightning(BaseAnomalySegmentationLightning):
    """
    PatchcoreLightning Module to train PatchCore algorithm
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)

        backbone, layers, input_size = hparams.model.backbone, hparams.model.layers, hparams.model.input_size
        self._model = PatchcoreModel(backbone=backbone, layers=layers, input_size=input_size)

        self.automatic_optimization = False
        self.callbacks = Callbacks(hparams)()

    def configure_optimizers(self):
        """
        Configure optimizers

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch, batch_idx):
        """
        Generate feature embedding of the batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            batch_idx (int): Batch Index

        Returns:
            Dict[str, np.ndarray]: Embedding Vector
        """
        self._model.feature_extractor.eval()
        embedding = self._model(batch["image"])

        return {"embedding": embedding}

    def training_epoch_end(self, outputs):
        """
        Concatenate batch embeddings to generate normal embedding.
        Apply coreset subsampling to the embedding set for dimensionality reduction.

        Args:
            outputs (List[Dict[str, np.ndarray]]): List of embedding vectors
        """
        embedding = np.vstack([output["embedding"] for output in outputs])
        sampling_ratio = self.hparams.model.coreset_sampling_ratio
        embedding = self._model.subsample_embedding(embedding, sampling_ratio)

        self._model.nn_search = self._model.nn_search.fit(embedding)
        self._model.memory_bank = torch.from_numpy(embedding)

    def validation_step(self, batch, batch_idx):
        """
        Load the normal embedding to use it as memory bank.
        Apply nearest neighborhood to the embedding.
        Generate the anomaly map.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            batch_idx (int): Batch Index

        Returns:
            Dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # TODO: Remove this Use Model Load Callback: https://jira.devtools.intel.com/browse/IAAALD-13
        filenames, images, labels, masks = batch["image_path"], batch["image"], batch["label"], batch["mask"]

        anomaly_map, _ = self._model(images)

        return {
            "filenames": filenames,
            "images": images,
            "true_labels": labels.cpu().numpy(),
            "true_masks": masks.squeeze().cpu().numpy(),
            "anomaly_maps": anomaly_map,
        }

    def validation_epoch_end(self, outputs):
        """
        Compute image and pixel level roc scores.

        Args:
          outputs: Batch of outputs from the validation step

        """
        self.filenames = [Path(f) for x in outputs for f in x["filenames"]]
        self.images = [x["images"] for x in outputs]

        self.true_masks = np.stack([output["true_masks"] for output in outputs])
        self.anomaly_maps = np.stack([output["anomaly_maps"] for output in outputs])

        self.true_labels = np.stack([output["true_labels"] for output in outputs])
        self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)

        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        _, self.image_f1_score = self._model.anomaly_map_generator.compute_adaptive_threshold(
            self.true_labels, self.pred_labels
        )

        self.log(name="Image-Level AUC", value=self.image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Image-Level F1", value=self.image_f1_score, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=self.pixel_roc_auc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Similar to validation, compute image and pixel level roc scores.

        Args:
            outputs: Batch of outputs from the test step

        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Compute and save anomaly scores of the test set, based on the embedding
            extracted from deep hierarchical CNN features.

        Args:
            outputs: Batch of outputs from the validation step

        """
        self.validation_epoch_end(outputs)
        threshold, _ = self._model.anomaly_map_generator.compute_adaptive_threshold(self.true_masks, self.anomaly_maps)

        for (filename, image, true_mask, anomaly_map) in zip(
            self.filenames, self.images, self.true_masks, self.anomaly_maps
        ):
            image = Denormalize()(image.squeeze())

            heat_map = self._model.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = self._model.anomaly_map_generator.compute_mask(anomaly_map=anomaly_map, threshold=threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")
            visualizer.save(Path(self.hparams.project.path) / "images" / filename.parent.name / filename.name)
            visualizer.close()
