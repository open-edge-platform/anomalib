"""Attention Bottleneck for UniNet."""

# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding.

    Args:
        in_planes (int): Number of input planes.
        out_planes (int): Number of output planes.
        stride (int): Stride of the convolution.
        groups (int): Number of groups.
        dilation (int): Dilation rate.

    Returns:
        nn.Conv2d: 3x3 convolution with padding.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution.

    Args:
        in_planes (int): Number of input planes.
        out_planes (int): Number of output planes.
        stride (int): Stride of the convolution.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def fuse_bn(conv: nn.Module, bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse convolution and batch normalization layers.

    Args:
        conv (nn.Module): Convolution layer.
        bn (nn.Module): Batch normalization layer.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Fused convolution and batch normalization layers.
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class AttentionBottleneck(nn.Module):
    """Attention Bottleneck for UniNet."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        attention: bool = True,
        halve: int = 1,
    ) -> None:
        super().__init__()
        self.attention = attention
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups  # 512
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.halve = halve
        k = 7
        p = 3

        self.bn2 = norm_layer(width // halve)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.bn4 = norm_layer(width // 2)
        self.bn5 = norm_layer(width // 2)
        self.bn6 = norm_layer(width // 2)
        self.bn7 = norm_layer(width)
        self.conv3x3 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3x3_ = nn.Conv2d(width // 2, width // 2, 3, 1, 1, bias=False)
        self.conv7x7 = nn.Conv2d(inplanes // 2, width // 2, kernel_size=k, stride=stride, padding=p, bias=False)
        self.conv7x7_ = nn.Conv2d(width // 2, width // 2, k, 1, p, bias=False)

    def get_same_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get same kernel and bias of the bottleneck."""
        k1, b1 = fuse_bn(self.conv3x3, self.bn2)
        k2, b2 = fuse_bn(self.conv3x3_, self.bn6)

        return k1, b1, k2, b2

    def merge_kernel(self) -> None:
        """Merge kernel of the bottleneck."""
        k1, b1, k2, b2 = self.get_same_kernel_bias()
        self.conv7x7 = nn.Conv2d(
            self.conv3x3.in_channels,
            self.conv3x3.out_channels,
            self.conv3x3.kernel_size,
            self.conv3x3.stride,
            self.conv3x3.padding,
            self.conv3x3.dilation,
            self.conv3x3.groups,
        )
        self.conv7x7_ = nn.Conv2d(
            self.conv3x3_.in_channels,
            self.conv3x3_.out_channels,
            self.conv3x3_.kernel_size,
            self.conv3x3_.stride,
            self.conv3x3_.padding,
            self.conv3x3_.dilation,
            self.conv3x3_.groups,
        )
        self.conv7x7.weight.data = k1
        self.conv7x7.bias.data = b1
        self.conv7x7_.weight.data = k2
        self.conv7x7_.bias.data = b2

    @staticmethod
    def _process_branch(
        branch: torch.Tensor,
        conv1: nn.Module,
        bn1: nn.Module,
        conv2: nn.Module,
        bn2: nn.Module,
        relu: nn.Module,
    ) -> torch.Tensor:
        """Process a branch of the bottleneck.

        Args:
            branch (torch.Tensor): Input tensor.
            conv1 (nn.Module): First convolution layer.
            bn1 (nn.Module): First batch normalization layer.
            conv2 (nn.Module): Second convolution layer.
            bn2 (nn.Module): Second batch normalization layer.
            relu (nn.Module): ReLU activation layer.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = conv1(branch)
        out = bn1(out)
        out = relu(out)
        out = conv2(out)
        return bn2(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        if self.halve == 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

        else:
            num_channels = x.shape[1]
            x_split = torch.split(x, [num_channels // 2, num_channels // 2], dim=1)

            out1 = self._process_branch(x_split[0], self.conv3x3, self.bn2, self.conv3x3_, self.bn5, self.relu)
            out2 = self._process_branch(x_split[-1], self.conv7x7, self.bn4, self.conv7x7_, self.bn6, self.relu)

            out = torch.cat([out1, out2], dim=1)

            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class BottleneckLayer(nn.Module):
    """Batch Normalization layer for UniNet.

    Args:
        block (Type[AttentionBottleneck]): Attention bottleneck.
        layers (int): Number of layers.
        groups (int): Number of groups.
        width_per_group (int): Width per group.
        norm_layer (Callable[..., nn.Module] | None): Normalization layer.
        halve (int): Number of halved channels.
    """

    def __init__(
        self,
        block: type[AttentionBottleneck],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
        halve: int = 2,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.halve = halve
        self.bn_layer = nn.Sequential(self._make_layer(block, 512, layers, stride=2))
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif self.halve == 2:
                m.merge_kernel()

    def _make_layer(
        self,
        block: type[AttentionBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Make a layer of the bottleneck.

        Args:
            block (AttentionBottleneck): Attention bottleneck.
            planes (int): Number of planes.
            blocks (int): Number of blocks.
            stride (int): Stride of the convolution.
            dilate (bool): Whether to dilate the convolution.

        Returns:
            nn.Sequential: Sequential layer.
        """
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                norm_layer,
                halve=self.halve,
            ),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
                halve=self.halve,
            )
            for _ in range(1, blocks)
        )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)  # 16*2048*8*8
        return output.contiguous()
