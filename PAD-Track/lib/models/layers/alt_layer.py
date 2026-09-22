import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConvRefiner(nn.Module):
    """
    Lightweight dynamic convolution for feature refinement.

    Input:
        U: [B, C, H, W]
        W_map: optional, kept only for interface compatibility with PATD
    Output:
        [B, C, H, W]
    """

    def __init__(self, in_channels, num_experts=2, init_scale=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts

        self.router = nn.Linear(in_channels, num_experts)

        self.experts = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False
            )
            for _ in range(num_experts)
        ])

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        for conv in self.experts:
            nn.init.constant_(conv.weight, 1.0 / 9.0)

        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, U, W_map=None):
        B, C, H, W = U.shape

        pooled = F.adaptive_avg_pool2d(U, 1).flatten(1)
        routing = F.softmax(self.router(pooled), dim=-1)

        expert_outputs = torch.stack(
            [expert(U) for expert in self.experts],
            dim=1
        )

        routing = routing.view(B, self.num_experts, 1, 1, 1)
        refined = torch.sum(routing * expert_outputs, dim=1)

        delta = refined - U
        return U + self.scale * delta


class GatedConvRefiner(nn.Module):
    """
    Lightweight gated convolution for feature refinement.

    Input:
        U: [B, C, H, W]
        W_map: optional, kept only for interface compatibility with PATD
    Output:
        [B, C, H, W]
    """

    def __init__(self, in_channels, init_scale=0.1):
        super().__init__()

        self.feature_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.gate_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True
        )

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.constant_(self.feature_conv.weight, 1.0 / 9.0)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, U, W_map=None):
        feature = self.feature_conv(U)
        gate = torch.sigmoid(self.gate_conv(U))

        delta = gate * (feature - U)
        return U + self.scale * delta


class LocalAttentionRefiner(nn.Module):
    """
    Lightweight local spatial attention with a k x k neighborhood.

    Depthwise 1x1 Q/K/V projections are used to keep the
    parameter count relatively small.

    Input:
        U: [B, C, H, W]
        W_map: optional, kept only for interface compatibility with PATD
    Output:
        [B, C, H, W]
    """

    def __init__(self, in_channels, kernel_size=3, init_scale=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_neighbors = kernel_size * kernel_size
        self.padding = kernel_size // 2

        self.q_proj = nn.Conv2d(
            in_channels, in_channels, 1,
            groups=in_channels, bias=True
        )
        self.k_proj = nn.Conv2d(
            in_channels, in_channels, 1,
            groups=in_channels, bias=True
        )
        self.v_proj = nn.Conv2d(
            in_channels, in_channels, 1,
            groups=in_channels, bias=True
        )
        self.out_proj = nn.Conv2d(
            in_channels, in_channels, 1,
            groups=in_channels, bias=True
        )

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, U, W_map=None):
        B, C, H, W = U.shape
        N = H * W

        q = self.q_proj(U)
        k = self.k_proj(U)
        v = self.v_proj(U)

        k_local = F.unfold(
            k,
            kernel_size=self.kernel_size,
            padding=self.padding
        ).view(B, C, self.num_neighbors, N)

        v_local = F.unfold(
            v,
            kernel_size=self.kernel_size,
            padding=self.padding
        ).view(B, C, self.num_neighbors, N)

        q = q.view(B, C, 1, N)

        attn = q * k_local
        attn = attn / math.sqrt(float(self.num_neighbors))
        attn = F.softmax(attn, dim=2)

        refined = torch.sum(attn * v_local, dim=2)
        refined = refined.view(B, C, H, W)
        refined = self.out_proj(refined)

        delta = refined - U
        return U + self.scale * delta


def build_refinement_operator(
    name,
    in_channels,
    num_experts=2,
    kernel_size=3,
    init_scale=0.1
):
    """
    Factory for replacement experiments.

    Example:
        refiner = build_refinement_operator(
            "dynamic_conv",
            in_channels=hidden_dim
        )
        diffused_2d = refiner(feat_2d, W_map)
    """
    name = name.lower()

    if name in ["dynamic", "dynamic_conv", "dynamic_convolution"]:
        return DynamicConvRefiner(
            in_channels=in_channels,
            num_experts=num_experts,
            init_scale=init_scale
        )

    if name in ["gated", "gated_conv", "gated_convolution"]:
        return GatedConvRefiner(
            in_channels=in_channels,
            init_scale=init_scale
        )

    if name in ["local_attn", "local_attention", "attention"]:
        return LocalAttentionRefiner(
            in_channels=in_channels,
            kernel_size=kernel_size,
            init_scale=init_scale
        )

    raise ValueError(
        f"Unknown refinement operator: {name}. "
        f"Supported: dynamic_conv, gated_conv, local_attention."
    )


if __name__ == "__main__":
    B, C, H, W = 2, 192, 16, 16
    U = torch.randn(B, C, H, W)
    W_map = torch.rand(B, 1, H, W)

    for method in ["dynamic_conv", "gated_conv", "local_attention"]:
        module = build_refinement_operator(method, C)
        out = module(U, W_map)
        params = sum(p.numel() for p in module.parameters())

        print(
            f"{method:16s} | "
            f"input={tuple(U.shape)} | "
            f"output={tuple(out.shape)} | "
            f"params={params}"
        )