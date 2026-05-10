# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# https://github.com/luow23/INP-Former/blob/5252579e5f401199643fbd16e030175856386f12/models/uad.py

"""TODO
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from anomalib.data import InferenceBatch

from .loss import GlobalCosineHmAdaptiveLoss


class InpFormerModel(nn.Module):
    """TODO
    """

    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
    ) -> None:
        super(InpFormerModel, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.loss = GlobalCosineHmAdaptiveLoss()


    def get_inp_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """INP coherence loss helps to ensure that INPs represent normal features 
        while minimizing the capture of anomalous features during testing. It minimizes 
        the distances between individual normal features and the corresponding nearest INP.

        Args:
            query (torch.Tensor): Fused encoder features (element-wise average).
            keys (torch.Tensor): Prototype visual token.

        Returns:
            torch.Tensor: INP coherence loss.
        """
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        inp_loss = self.distance.mean()
        return inp_loss

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass of the INPFormerModel model.

        During training, the model extracts features from the encoder and decoder
        and returns them for loss computation. During inference, it computes
        anomaly maps by comparing encoder and decoder features using cosine similarity,
        applies Gaussian smoothing, and returns anomaly scores and maps. TODO

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor | InferenceBatch:
                - During training: Encoder and decoder features, INP coherence loss.
                - During inference: InferenceBatch with pred_score (anomaly scores)
                  and anomaly_map (pixel-level anomaly maps).

        """
        en, de, inp_loss = self.get_encoder_decoder_inploss(self, batch)

        if self.training:
            return self.loss(encoder_features=en, decoder_features=de, inp_loss=inp_loss)

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors by averaging.

        Takes a list of feature tensors and computes their element-wise average
        to create a fused representation.

        Args:
            feat_list (list[torch.Tensor]): List of feature tensors to fuse.

        Returns:
            torch.Tensor: Averaged feature tensor.

        """
        return torch.stack(feat_list, dim=1).mean(dim=1)
    
    def get_encoder_decoder_inploss(self, x):
        """Extract and process features through encoder and decoder.

        This method processes input images through the DINOv2 encoder to extract
        features from target layers, fuses them through a bottleneck MLP, and
        reconstructs them using the decoder. Features are reshaped for spatial
        anomaly map computation. TODO

        Args:
            x (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple containing:
                - en: List of fused encoder features reshaped to spatial dimensions
                - de: List of fused decoder features reshaped to spatial dimensions
                - inp_loss: INP coherence loss to guide INP Extractor
        """
        x = self.encoder.prepare_tokens(x)
        batch_size = x.shape[0]
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self._fuse_feature(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((batch_size, 1, 1)), x)
        inp_loss = self.get_inp_loss(x, agg_prototype)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self._fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de, inp_loss
