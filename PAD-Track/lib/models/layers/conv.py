import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PDEAdapter(nn.Module):
    """
    卷积消融版本：
    不含 FSA、DSP-Map、GDRI，并将各向异性热扩散替换为局部卷积。

    保留：
    1. Down/Up bottleneck projection
    2. Direct residual injection

    替换：
    1. RobustAnisotropicDiffusion -> ConvolutionRefiner
    """

    def __init__(
        self,
        dim,
        search_size=256,
        patch_size=16,
        template_len=64,
        bottleneck_ratio=4,
        num_iters=1,          # 保留该参数仅用于兼容原调用
        return_residual=True,
        init_scale=1e-3,
    ):
        super().__init__()

        self.dim = dim
        self.template_len = template_len
        self.H_s = search_size // patch_size
        self.W_s = search_size // patch_size
        self.return_residual = return_residual

        hidden_dim = max(dim // bottleneck_ratio, 32)
        self.hidden_dim = hidden_dim

        # 1) 对称下投影 / 上投影
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, dim)

        # 2) 用卷积替代各向异性热扩散
        self.pde_solver = ConvolutionRefiner(
            in_channels=hidden_dim
        )

        # 3) 总体残差缩放
        self.scale = nn.Parameter(
            torch.ones(1, 1, dim) * init_scale
        )

    def _branch_forward(self, self_tokens, H, W):
        """
        单分支：
        tokens -> 2D feature -> convolution
        -> convolution residual -> direct projection
        """
        B, N, Ch = self_tokens.shape

        # 1. Token -> 2D feature
        feat_2d = (
            self_tokens.transpose(1, 2)
            .reshape(B, Ch, H, W)
            .contiguous()
        )

        # 2. 卷积特征精炼
        refined_2d = self.pde_solver(feat_2d)

        # 3. 提取卷积产生的残差
        delta_tokens = (
            (refined_2d - feat_2d)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )  # [B, N, Ch]

        # 4. 无 GDRI：直接投影并注入
        residual = self.scale * self.up_proj(
            delta_tokens
        )  # [B, N, C]

        return residual

    def forward(self, x, template_len=None):
        """
        x: [B, N, C]
        joint tokens: [template ; search]
        """
        B, N, C = x.shape
        template_len = template_len or self.template_len

        # 1. Split template/search
        z_tokens = x[:, :template_len, :]
        x_tokens = x[:, template_len:, :]

        Nt = z_tokens.shape[1]
        Ns = x_tokens.shape[1]

        h_z = w_z = int(math.sqrt(Nt))
        h_x = w_x = int(math.sqrt(Ns))

        if h_z * w_z != Nt or h_x * w_x != Ns:
            raise ValueError(
                "Token length must be a perfect square."
            )

        # 2. Bottleneck projection
        z_side = self.down_proj(z_tokens)
        x_side = self.down_proj(x_tokens)

        # 3. 独立卷积精炼
        z_residual = self._branch_forward(
            z_side, h_z, w_z
        )
        x_residual = self._branch_forward(
            x_side, h_x, w_x
        )

        # 4. Recombine residuals
        residual = torch.cat(
            [z_residual, x_residual],
            dim=1
        )

        if self.return_residual:
            # 保持与原 adapter 外部解包接口一致
            return residual, None
        else:
            return x + residual


class ConvolutionRefiner(nn.Module):
    """
    用于替代 PATD 的局部卷积精炼算子。

    为了与 PATD 的逐通道空间演化方式尽量保持公平，
    采用 depthwise 3x3 convolution：
        [B, C, H, W] -> [B, C, H, W]

    不使用：
    - DSP-Map
    - directional diffusion
    - conduction coefficient
    - GDRI
    """

    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )

        # 与原 PATD 中 smoother 的初始化保持接近，
        # 初始为局部 3x3 平均卷积。
        nn.init.constant_(
            self.conv.weight,
            1.0 / 9.0
        )

    def forward(self, U):
        """
        U: [B, C, H, W]
        """
        return self.conv(U)


if __name__ == "__main__":

    B = 2
    C = 768
    Nt = 64
    Ns = 256

    x = torch.randn(
        B,
        Nt + Ns,
        C
    )

    adapter = PDEA2(
        dim=C,
        template_len=Nt,
        bottleneck_ratio=4,
        num_iters=1,
        return_residual=True
    )

    residual, w_map = adapter(x)

    print("Input shape:   ", x.shape)
    print("Residual shape:", residual.shape)
    print("W_map:        ", w_map)

    conv_params = sum(
        p.numel()
        for p in adapter.pde_solver.parameters()
    )
    print("Conv params:  ", conv_params)