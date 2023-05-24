"""Regions extraction module of AI-VAD model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.ops import box_area, clip_boxes_to_image
from torchvision.transforms.functional import gaussian_blur, rgb_to_grayscale

from anomalib.data.utils.boxes import boxes_to_masks, masks_to_boxes

PERSON_LABEL = 1


class RegionExtractor(nn.Module):
    """Region extractor for AI-VAD.

    Args:
        box_score_thresh (float): Confidence threshold for bounding box predictions.
    """

    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        self.persons_only = False
        self.min_bbox_area = 100
        self.max_overlap = 0.65
        self.binary_threshold = 18
        self.gaussian_kernel_size = 3

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.backbone = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh, rpn_nms_thresh=0.3)

    def forward(self, first_frame, last_frame: Tensor) -> list[dict]:
        """Forward pass through region extractor.

        Args:
            batch (Tensor): Batch of input images of shape (N, C, H, W)
        Returns:
            list[dict]: List of Mask RCNN predictions for each image in the batch.
        """
        with torch.no_grad():
            regions = self.backbone(last_frame)

        regions = self._add_foreground_boxes(regions, first_frame, last_frame, self.binary_threshold)
        regions = self.post_process_bbox_detections(regions)

        return regions

    def post_process_bbox_detections(self, batch_regions):
        filtered_regions = []
        for im_regions in batch_regions:
            if self.persons_only:
                im_regions = self._keep_only_persons(im_regions)
            im_regions = self._filter_by_area(im_regions, self.min_bbox_area)
            im_regions = self._delete_overlapping_boxes(im_regions, self.max_overlap)
            filtered_regions.append(im_regions)
        return filtered_regions

    def _keep_only_persons(self, regions):
        keep = torch.where(regions["labels"] == PERSON_LABEL)
        return self.subsample_regions(regions, keep)

    def _filter_by_area(self, regions, min_area):
        """Remove all regions with a surface area smaller than the specified value."""

        areas = box_area(regions["boxes"])
        keep = torch.where(areas > min_area)
        return self.subsample_regions(regions, keep)

    def _delete_overlapping_boxes(self, regions, threshold):
        """Delete overlapping bounding boxes, larger boxes are kept."""

        # sort boxes by area
        areas = box_area(regions["boxes"])
        indices = areas.argsort()

        keep = []
        for idx in range(len(indices)):
            overlap_coords = torch.hstack(
                [
                    torch.max(regions["boxes"][indices[idx], :2], regions["boxes"][indices[idx + 1 :], :2]),  # x1, y1
                    torch.min(regions["boxes"][indices[idx], 2:], regions["boxes"][indices[idx + 1 :], 2:]),  # x2, y2
                ]
            )
            mask = torch.all(overlap_coords[:, :2] < overlap_coords[:, 2:], dim=1)  # filter non-overlapping
            overlap = box_area(overlap_coords) * mask.int()
            overlap_ratio = overlap / areas[indices[idx]]

            if not any(overlap_ratio > threshold):
                keep.append(indices[idx])

        return self.subsample_regions(regions, torch.stack(keep))

    def _add_foreground_boxes(self, regions, first_frame, last_frame, binary_threshold):
        # apply gaussian blur to first and last frame
        first_frame = gaussian_blur(first_frame, [self.gaussian_kernel_size, self.gaussian_kernel_size])
        last_frame = gaussian_blur(last_frame, [self.gaussian_kernel_size, self.gaussian_kernel_size])

        # take the abs diff between the blurred images and convert to grayscale
        pixel_diff = torch.abs(first_frame - last_frame)
        pixel_diff = rgb_to_grayscale(pixel_diff).squeeze(1)

        # apply binary threshold to the diff
        foreground_map = (pixel_diff > binary_threshold / 255).int()

        # remove regions already detected by region extractor
        boxes_list = [im_regions["boxes"] for im_regions in regions]
        boxes_list = [
            clip_boxes_to_image(boxes + Tensor([-2, -2, 2, 2]).to(boxes.device), foreground_map.shape[-2:])
            for boxes in boxes_list
        ]
        boxes_mask = boxes_to_masks(boxes_list, foreground_map.shape[-2:]).int()
        foreground_map *= -boxes_mask + 1  # invert mask

        # find boxes from foreground map
        batch_boxes, _ = masks_to_boxes(foreground_map)

        # append foreground detections to region extractor detections
        for image_regions, boxes, pixel_mask in zip(regions, batch_boxes, foreground_map):
            if boxes.shape[0] == 0:
                continue
            image_regions["boxes"] = torch.cat([image_regions["boxes"], boxes])
            image_regions["labels"] = torch.cat(
                [image_regions["labels"], torch.zeros(boxes.shape[0], device=boxes.device)]
            )

            image_boxes_as_list = [box.unsqueeze(0) for box in boxes]  # list with one box per element
            boxes_mask = boxes_to_masks(image_boxes_as_list, pixel_mask.shape[-2:]).int()
            new_masks = pixel_mask.repeat((len(image_boxes_as_list), 1, 1)) * boxes_mask

            image_regions["masks"] = torch.cat([image_regions["masks"], new_masks.unsqueeze(1)])
            image_regions["scores"] = torch.cat(
                [image_regions["scores"], torch.ones(boxes.shape[0], device=boxes.device) * 0.5]
            )

        return regions

    @staticmethod
    def remove_boxes_from_mask(mask, boxes):
        boxes_mask = boxes_to_masks(boxes, mask.shape[-2:]).int()
        return mask * (-boxes_mask + 1)  # invert mask

    @staticmethod
    def subsample_regions(regions, indices):
        new_regions_dict = {}
        for key, value in regions.items():
            new_regions_dict[key] = value[indices]
        return new_regions_dict
