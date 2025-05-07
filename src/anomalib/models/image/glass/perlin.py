# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import math

import imgaug.augmenters as iaa
import numpy as np
import PIL
import PIL.Image
import torch
from torch import nn
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def generate_thr(img_shape, min=0, max=4):
    min_perlin_scale = min
    max_perlin_scale = max
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_noise_np = rand_perlin_2d_np((img_shape[1], img_shape[2]), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise_np)
    perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
    return perlin_thr


def perlin_mask(img_shape, feat_size, min, max, mask_fg, flag=0):
    mask = np.zeros((feat_size, feat_size))
    while np.max(mask) == 0:
        perlin_thr_1 = generate_thr(img_shape, min, max)
        perlin_thr_2 = generate_thr(img_shape, min, max)
        temp = torch.rand(1).numpy()[0]
        if temp > 2 / 3:
            perlin_thr = perlin_thr_1 + perlin_thr_2
            perlin_thr = np.where(perlin_thr > 0, np.ones_like(perlin_thr), np.zeros_like(perlin_thr))
        elif temp > 1 / 3:
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            perlin_thr = perlin_thr_1
        perlin_thr = torch.from_numpy(perlin_thr)
        perlin_thr_fg = perlin_thr * mask_fg
        down_ratio_y = int(img_shape[1] / feat_size)
        down_ratio_x = int(img_shape[2] / feat_size)
        mask_ = perlin_thr_fg
        mask = torch.nn.functional.max_pool2d(
            perlin_thr_fg.unsqueeze(0).unsqueeze(0),
            (down_ratio_y, down_ratio_x),
        ).float()
        mask = mask.numpy()[0, 0]
    mask_s = mask
    if flag != 0:
        mask_l = mask_.numpy()
    if flag == 0:
        return mask_s
    return mask_s, mask_l


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0),
        d[1],
        axis=1,
    )
    dot = lambda grad, shift: (
        np.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), axis=-1)
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


class PerlinNoise(nn.Module):
    def __init__(self, anomaly_source_path):
        super().__init__()
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

    def forward(self, image):
        aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
        transform_aug = self.rand_augmenter()
        aug = transform_aug(aug)

        mask_all = perlin_mask(image.shape, self.imgsize // self.downsampling, 0, 6, mask_fg, 1)
        mask_s = torch.from_numpy(mask_all[0])
        mask_l = torch.from_numpy(mask_all[1])
        mask_fg = torch.tensor([1])

        beta = np.random.normal(loc=self.mean, scale=self.std)
        beta = np.clip(beta, 0.2, 0.8)
        aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l
        return aug_image, mask_s

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug
