# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DoRA Layers for AnomalyVFM."""

import torch
from torch import nn
from torch.nn import functional


class _DoRALinearBase(nn.Module):
    """DoRA base class.

    Args:
        layer (nn.Module): underlying module
        r (int): rank of low-rank matrices
        alpha (int): scaling factor
    """

    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        self.layer = layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0


class DoRAWrapper(_DoRALinearBase):
    """DoRA Wrapper for Linear Layers.

    Args:
        layer (nn.Module): underlying module
        r (int): rank of low-rank matrices
        alpha (int): scaling factor
    """

    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0) -> None:
        super().__init__(layer=layer, r=r, alpha=alpha)
        out_dim = self.layer.weight.shape[0]
        in_dim = self.layer.weight.shape[1]

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
            with torch.no_grad():
                init_mag = self.layer.weight.norm(dim=1)
            self.magnitude = nn.Parameter(init_mag)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.register_parameter("magnitude", None)

    @property
    def weight(self) -> torch.Tensor:
        """Returns the weights of the original layer.

        Returns:
            (torch.Tensor): original weights.
        """
        return self.layer.weight

    @property
    def bias(self) -> torch.Tensor:
        """Returns the bias of the original layer.

        Returns:
            (torch.Tensor): original bias.
        """
        return self.layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features through the original block and the DoRA block.

        Args:
            x: input features

        Returns:
            (torch.Tensor): processed features.
        """
        if self.r <= 0:
            return self.layer(x)

        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        base_w = self.layer.weight
        adapted_w = base_w + delta_w

        direction = adapted_w / (adapted_w.norm(dim=1, keepdim=True) + 1e-8)
        final_w = self.magnitude.unsqueeze(1) * direction
        return functional.linear(x, final_w, self.layer.bias)


class DoRAQKVWrapper(_DoRALinearBase):
    """DoRA Wrapper for QKV Layers.

    Args:
        layer (nn.Module): underlying module
        r (int): rank of low-rank matrices
        alpha (int): scaling factor
    """

    def __init__(self, layer: nn.Module, r: int = 4, alpha: float = 1.0) -> None:
        super().__init__(layer=layer, r=r, alpha=alpha)
        out_dim_total, in_dim = self.layer.weight.shape
        out_dim = out_dim_total // 3
        self.in_features = in_dim

        if r > 0:
            self.lora_A_q = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_q = nn.Parameter(torch.zeros(out_dim, r))

            self.lora_A_v = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B_v = nn.Parameter(torch.zeros(out_dim, r))

            with torch.no_grad():
                q_w, _, v_w = self.layer.weight.chunk(3, dim=0)
                self.mag_q = nn.Parameter(q_w.norm(dim=1))
                self.mag_v = nn.Parameter(v_w.norm(dim=1))
        else:
            self.register_parameter("lora_A_q", None)
            self.register_parameter("lora_B_q", None)
            self.register_parameter("lora_A_v", None)
            self.register_parameter("lora_B_v", None)
            self.register_parameter("mag_q", None)
            self.register_parameter("mag_v", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features throuhg the original block and the DoRA block.

        Args:
            x: input features

        Returns:
            (torch.Tensor): processed features.
        """
        if self.r <= 0:
            return self.layer(x)

        base_q, base_k, base_v = self.layer.weight.chunk(3, dim=0)
        delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling
        delta_v = (self.lora_B_v @ self.lora_A_v) * self.scaling

        adapted_q = base_q + delta_q
        adapted_v = base_v + delta_v

        dir_q = adapted_q / (adapted_q.norm(dim=1, keepdim=True) + 1e-8)
        dir_v = adapted_v / (adapted_v.norm(dim=1, keepdim=True) + 1e-8)

        final_q = self.mag_q.unsqueeze(1) * dir_q
        final_v = self.mag_v.unsqueeze(1) * dir_v
        final_w = torch.cat([final_q, base_k, final_v], dim=0)

        return functional.linear(x, final_w, self.layer.bias)


def add_peft(model: nn.Module, r: int = 64, alpha: float = 1.0) -> None:
    """Traversal through the model layers that adds DoRA blocks to the predefined ones.

    Args:
        model (nn:module): model
        r (int): rank of low-rank matrices
        alpha (int): scaling factor
    """
    for name, module in model.named_children():
        if name == "qkv":
            wrapped = DoRAQKVWrapper(module, r=r, alpha=alpha)
            setattr(model, name, wrapped)
        elif name == "proj":
            wrapped = DoRAWrapper(module, r=r, alpha=alpha)
            setattr(model, name, wrapped)
        else:
            add_peft(module, r=r, alpha=alpha)
