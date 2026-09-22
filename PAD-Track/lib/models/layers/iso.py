import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PDEAdapter(nn.Module):
    """
    各向同性热扩散消融版本：
    不含 FSA、DSP-Map、GDRI，并将各向异性热扩散替换为各向同性热扩散。

    保留：
    1. Down/Up bottleneck projection
    2. Direct diffusion residual injection
    3. 与原 PATD 相同的 8 邻域空间传播范围

    替换：
    1. RobustAnisotropicDiffusion -> IsotropicThermalDiffusion
    2. 删除方向相关 conduction coefficient c_k
    3. 删除 learnable directional weights omega_k
    4. 所有方向采用相同的扩散系数 kappa
    """

    def __init__(
        self,
        dim,
        search_size=256,
        patch_size=16,
        template_len=64,
        bottleneck_ratio=4,
        num_iters=1,
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

        # 2) 各向同性热扩散
        self.pde_solver = IsotropicThermalDiffusion(
            in_channels=hidden_dim,
            num_iters=num_iters
        )

        # 3) 总体残差缩放
        self.scale = nn.Parameter(
            torch.ones(1, 1, dim) * init_scale
        )

    def _branch_forward(self, self_tokens, H, W):
        """
        单分支：
        tokens -> 2D feature -> isotropic thermal diffusion
        -> diffusion residual -> direct projection
        """
        B, N, Ch = self_tokens.shape

        # 1. Token -> 2D feature field
        feat_2d = (
            self_tokens.transpose(1, 2)
            .reshape(B, Ch, H, W)
            .contiguous()
        )

        # 2. 各向同性热扩散
        diffused_2d = self.pde_solver(feat_2d)

        # 3. 提取 diffusion residual
        delta_tokens = (
            (diffused_2d - feat_2d)
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

        # 1. Split template/search tokens
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

        # 3. Independent isotropic thermal diffusion
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
            # 保持外部接口兼容
            return residual, None
        else:
            return x + residual


class IsotropicThermalDiffusion(nn.Module):
    """
    8 邻域各向同性热扩散。

    与各向异性版本的关键区别：
    ------------------------------------------------
    Anisotropic:
        c_k = exp(-(g_k / K)^2)
        flux = sum_k omega_k * c_k * g_k

    Isotropic:
        flux = kappa * sum_k g_k

    即：
    - 不根据局部梯度改变扩散强度；
    - 不使用方向相关的可学习权重；
    - 所有空间方向采用相同的扩散系数 kappa；
    - 不使用 DSP-Map。

    对角方向仍除以 sqrt(2)，仅用于补偿几何距离，
    这并不引入方向选择性。
    """

    def __init__(
        self,
        in_channels,
        num_iters=1,
        init_kappa=1.0,
        init_dt=0.1,
    ):
        super().__init__()

        self.num_iters = num_iters
        self.in_channels = in_channels

        # 单一、共享的各向同性扩散系数
        # 所有通道、所有方向使用同一系数
        self.kappa = nn.Parameter(
            torch.tensor(float(init_kappa))
        )

        # 时间步长
        self.dt = nn.Parameter(
            torch.tensor(float(init_dt))
        )

    def _get_8dir_gradients(self, U):
        """
        计算 8 邻域差分。
        对角方向使用 1/sqrt(2) 进行空间距离校正。
        """
        U_pad = F.pad(
            U,
            (1, 1, 1, 1),
            mode='replicate'
        )

        curr = U_pad[:, :, 1:-1, 1:-1]

        # Cardinal directions
        dN = U_pad[:, :, 0:-2, 1:-1] - curr
        dS = U_pad[:, :, 2:, 1:-1] - curr
        dW = U_pad[:, :, 1:-1, 0:-2] - curr
        dE = U_pad[:, :, 1:-1, 2:] - curr

        # Diagonal directions
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        dNW = (
            U_pad[:, :, 0:-2, 0:-2] - curr
        ) * inv_sqrt2

        dNE = (
            U_pad[:, :, 0:-2, 2:] - curr
        ) * inv_sqrt2

        dSW = (
            U_pad[:, :, 2:, 0:-2] - curr
        ) * inv_sqrt2

        dSE = (
            U_pad[:, :, 2:, 2:] - curr
        ) * inv_sqrt2

        return [
            dN, dS, dW, dE,
            dNW, dNE, dSW, dSE
        ]

    def forward(self, U):
        """
        U: [B, C, H, W]
        """

        for _ in range(self.num_iters):

            # 1. 计算 8 个方向的局部差分
            grads = self._get_8dir_gradients(U)

            # 2. 各向同性扩散：
            #    所有方向使用完全相同的扩散系数
            flux_div = self.kappa * sum(grads)

            # 3. 显式 Euler 时间步进
            U = U + self.dt * flux_div

        return U


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

    iso_params = sum(
        p.numel()
        for p in adapter.pde_solver.parameters()
    )
    print("Isotropic diffusion params:", iso_params)