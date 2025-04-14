import imgaug.augmenters as iaa
import numpy as np
import torch
import math


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
        mask = torch.nn.functional.max_pool2d(perlin_thr_fg.unsqueeze(0).unsqueeze(0), (down_ratio_y, down_ratio_x)).float()
        mask = mask.numpy()[0, 0]
    mask_s = mask
    if flag != 0:
        mask_l = mask_.numpy()
    if flag == 0:
        return mask_s
    else:
        return mask_s, mask_l


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1],
                                                  axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])
