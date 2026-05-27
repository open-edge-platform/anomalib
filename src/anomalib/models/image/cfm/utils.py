# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Point Cloud and 3D operations in CFM."""

import numpy as np
import tifffile as tiff
import torch


def organized_pc_to_unorganized_pc(organized_pc: np.ndarray) -> np.ndarray:
    """Converts an organized point cloud to an unorganized one."""
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path: str) -> np.ndarray:
    """Reads a TIFF file and returns the point cloud."""
    return tiff.imread(path)


def resize_organized_pc(
    organized_pc: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    tensor_out: bool = True,
) -> torch.Tensor | np.ndarray:
    """Resizes an organized point cloud."""
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(
        torch_organized_pc,
        size=(target_height, target_width),
        mode="nearest",
    )
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()


def organized_pc_to_depth_map(organized_pc: np.ndarray) -> np.ndarray:
    """Extracts the depth map from an organized point cloud."""
    return organized_pc[:, :, 2]


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalizes a point cloud to fit within a unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc / np.max(np.sqrt(np.sum(pc**2, axis=1)))


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Euclidean squared distance between two set of points."""
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Extract points given an index tensor."""
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Samples furthest points(FPS)."""
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long).to(device)
    distance = torch.ones(b, n).to(device) * 1e10
    farthest = torch.randint(0, n, (b,), dtype=torch.long).to(device)
    batch_indices = torch.arange(b, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def interpolating_points(xyz1: torch.Tensor, xyz2: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """Interpolates features of sampled points (xyz2) on original points (xyz1)."""
    xyz1 = xyz1.permute(0, 2, 1)
    xyz2 = xyz2.permute(0, 2, 1)
    points2 = points2.permute(0, 2, 1)

    b, n, _ = xyz1.shape
    _, s, _ = xyz2.shape

    if s == 1:
        interpolated_points = points2.repeat(1, n, 1).permute(0, 2, 1)
    else:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # Take 3 nearest neighbors
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(b, n, 3, 1), dim=2)
        interpolated_points = interpolated_points.permute(0, 2, 1)

    return interpolated_points
