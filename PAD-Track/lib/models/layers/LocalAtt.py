import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PDEAdapter(nn.Module):
    """
    Feature-refinement ablation version.

    Removed:
    - FSA
    - DSP-Map
    - GDRI

    Replaced:
    - RobustAnisotropicDiffusion -> LocalAttentionRefiner
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

        # Down/Up bottleneck projection
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, dim)

        # Alternative feature refinement operator
        self.pde_solver = LocalAttentionRefiner(
            in_channels=hidden_dim,
            num_iters=num_iters
        )

        # Direct residual injection scale
        self.scale = nn.Parameter(
            torch.ones(1, 1, dim) * init_scale
        )

    def _branch_forward(self, self_tokens, H, W):
        """
        tokens -> 2D feature -> refinement operator
        -> refinement residual -> direct projection
        """
        B, N, Ch = self_tokens.shape

        feat_2d = (
            self_tokens.transpose(1, 2)
            .reshape(B, Ch, H, W)
            .contiguous()
        )

        refined_2d = self.pde_solver(feat_2d)

        delta_tokens = (
            (refined_2d - feat_2d)
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        # No GDRI: directly project the refinement residual
        residual = self.scale * self.up_proj(delta_tokens)

        return residual

    def forward(self, x, template_len=None):
        """
        x: [B, N, C], joint tokens [template ; search]
        """
        B, N, C = x.shape
        template_len = template_len or self.template_len

        z_tokens = x[:, :template_len, :]
        x_tokens = x[:, template_len:, :]

        Nt = z_tokens.shape[1]
        Ns = x_tokens.shape[1]

        h_z = w_z = int(math.sqrt(Nt))
        h_x = w_x = int(math.sqrt(Ns))

        if h_z * w_z != Nt or h_x * w_x != Ns:
            raise ValueError("Token length must be a perfect square.")

        z_side = self.down_proj(z_tokens)
        x_side = self.down_proj(x_tokens)

        z_residual = self._branch_forward(z_side, h_z, w_z)
        x_residual = self._branch_forward(x_side, h_x, w_x)

        residual = torch.cat(
            [z_residual, x_residual],
            dim=1
        )

        if self.return_residual:
            # Keep the original external unpacking interface:
            # residual, w_map = adapter(x)
            return residual, None
        else:
            return x + residual



class LocalAttentionRefiner(nn.Module):
    """
    Lightweight 3x3 local attention.

    For every spatial position, attention is computed only within
    its local 3x3 neighborhood. Depthwise 1x1 Q/K/V projections
    are used to keep the computational budget lightweight.
    """

    def __init__(
        self,
        in_channels,
        num_iters=1,
        kernel_size=3
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_neighbors = kernel_size * kernel_size

        # Lightweight channel-wise Q/K/V projections
        self.q_proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True
        )

        self.k_proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True
        )

        self.v_proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True
        )

        self.out_proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=in_channels,
            bias=True
        )

        # Identity-like initialization
        for layer in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj
        ]:
            nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _refine_once(self, U):
        B, C, H, W = U.shape
        N = H * W

        q = self.q_proj(U)
        k = self.k_proj(U)
        v = self.v_proj(U)

        # Local K/V neighborhoods:
        # [B, C*k*k, H*W] -> [B, C, k*k, H*W]
        k_local = F.unfold(
            k,
            kernel_size=self.kernel_size,
            padding=self.padding
        ).view(
            B,
            C,
            self.num_neighbors,
            N
        )

        v_local = F.unfold(
            v,
            kernel_size=self.kernel_size,
            padding=self.padding
        ).view(
            B,
            C,
            self.num_neighbors,
            N
        )

        # [B, C, 1, H*W]
        q = q.view(
            B,
            C,
            1,
            N
        )

        # Local content-dependent attention
        attn = q * k_local
        attn = attn / math.sqrt(
            float(self.num_neighbors)
        )
        attn = F.softmax(
            attn,
            dim=2
        )

        refined = torch.sum(
            attn * v_local,
            dim=2
        )

        refined = refined.view(
            B,
            C,
            H,
            W
        )

        refined = self.out_proj(
            refined
        )

        return refined

    def forward(self, U):
        for _ in range(self.num_iters):
            U = self._refine_once(U)
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

    refiner_params = sum(
        p.numel()
        for p in adapter.pde_solver.parameters()
    )
    print("Refiner params:", refiner_params)