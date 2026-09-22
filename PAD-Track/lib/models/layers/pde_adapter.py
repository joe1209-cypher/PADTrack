
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttentionPooling(nn.Module):
    """
    空间注意力池化：自动抑制红外背景噪声，提取 100% 纯净的前景目标 Summary。
    """

    def __init__(self, dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        """ x: [B, N, C] """
        # 计算空间权重并 Softmax 归一化
        attn_scores = self.attn_net(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]

        # 核心：按注意力权重聚合，抛弃纯背景 Token
        weighted_feat = torch.sum(x * attn_weights, dim=1)  # [B, C]
        summary = self.feat_proj(weighted_feat)  # [B, C]

        return summary, attn_weights


class PDEAdapter(nn.Module):
    """
    对称跨区域 PDE 适配器 (Symmetric Cross-Region PDE Adapter)
    专为红外目标跟踪单流网络 (One-Stream ViT) 设计的残差侧调模块。
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

        # 1) 对称下投影 (降维加速)
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, dim)

        # # 【微调生命线】：上投影零初始化，保证初期对原网络 0 干扰
        # nn.init.zeros_(self.up_proj.weight)
        # nn.init.zeros_(self.up_proj.bias)
        # nn.init.xavier_uniform_(self.up_proj.weight)
        # nn.init.zeros_(self.up_proj.bias)
        # 2) 物理引擎
        self.pde_solver = RobustAnisotropicDiffusion(
            in_channels=hidden_dim,
            num_iters=num_iters
        )

        # 3) 动态绝热墙生成器 (融合当前特征与对侧 Summary)
        self.prior_generator = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 0~1 的先验概率图 W_map
        )

        # 4) 【升级版】高级空间注意力池化提取器
        self.summary_extractor = SpatialAttentionPooling(hidden_dim)
        # self.summary_extractor=torch.mean(dim=1)
        # 5) 动态门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 6) 总体控制阀门
        self.scale = nn.Parameter(torch.ones(1, 1, dim) * 0.001
                                  )

    def _branch_forward(self, self_tokens, peer_summary, H, W):
        """ 处理单一分支，生成 PDE 精炼残差 """
        B, N, Ch = self_tokens.shape

        # 1. 2D 空间化与 Summary 广播
        feat_2d = self_tokens.transpose(1, 2).reshape(B, Ch, H, W).contiguous()
        peer_map = peer_summary[:, :, None, None].expand(-1, -1, H, W)

        # 2. 生成绝热墙 W_map 并求解 PDE
        prior_input = torch.cat([feat_2d, peer_map], dim=1)
        W_map = self.prior_generator(prior_input)

        diffused_2d = self.pde_solver(feat_2d, W_map)

        # 3. 提取纯净 c (去掉了极易导致 NaN 的 LayerNorm)
        delta_tokens = (diffused_2d - feat_2d).flatten(2).transpose(1, 2).contiguous()  # [B, N, Ch]

        # 4. 计算门控并缩放
        peer_summary_tokens = peer_summary.unsqueeze(1).expand(-1, N, -1)
        gate_input = torch.cat([self_tokens, delta_tokens, peer_summary_tokens], dim=-1)
        gate = self.gate_network(gate_input)  # [B, N, Ch]

        delta_gated = gate * delta_tokens
        residual = self.scale * self.up_proj(delta_gated)  # [B, N, C]
        # return residual
        return residual,W_map

    def forward(self, x, template_len=None):
        """
        x: [B, N, C], 拼接的联合 Token [template ; search]
        """
        B, N, C = x.shape
        template_len = template_len or self.template_len

        # 序列拆分
        z_tokens = x[:, :template_len, :]
        x_tokens = x[:, template_len:, :]

        Nt, Ns = z_tokens.shape[1], x_tokens.shape[1]
        h_z = w_z = int(math.sqrt(Nt))
        h_x = w_x = int(math.sqrt(Ns))

        # 安全检查
        if h_z * w_z != Nt or h_x * w_x != Ns:
            raise ValueError("Token length must be a perfect square.")

        # ==========================================
        # 核心交互流
        # ==========================================
        # 1. 降维
        z_side = self.down_proj(z_tokens)
        x_side = self.down_proj(x_tokens)

        # 2. 【核心升级】：提取无背景污染的前景 Summary
        z_summary, z_weights = self.summary_extractor(z_side)
        x_summary, x_weights = self.summary_extractor(x_side)
        # z_summary=torch.mean(z_side, dim=1)
        # x_summary = torch.mean(x_side,dim=1)
        # 3. 交叉引导 PDE 演化
        # z_residual = self._branch_forward(z_side, x_summary, h_z, w_z)
        # x_residual = self._branch_forward(x_side, z_summary, h_x, w_x)
        z_residual, _= self._branch_forward(z_side, x_summary, h_z, w_z)
        x_residual, x_W_map = self._branch_forward(x_side, z_summary, h_x, w_x)

        # 4. 重组输出
        residual = torch.cat([z_residual, x_residual], dim=1)

        if self.return_residual:
            return residual,x_W_map
        else:
            return x + residual


class RobustAnisotropicDiffusion(nn.Module):
    """
    高性能红外抗干扰版：8 向可微各向异性扩散 PDE 求解器。
    【最新版】：作为纯粹的物理引擎，接收外部算好的 W_map 进行演化。
    """

    def __init__(self, in_channels, num_iters=1):
        super().__init__()
        self.num_iters = num_iters
        self.in_channels = in_channels

        # 物理约束的可学习参数
        self.K = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1) * 2.0)
        self.dt = nn.Parameter(torch.ones(1) * 0.1)
        self.beta = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.05)

        # 8 个方向的可学习通量权重 (适应红外特定运动拖影)
        self.dir_weights = nn.Parameter(torch.ones(8))

        # 可学习的红外抗噪平滑核 (替代 avg_pool2d)
        self.smoother = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        nn.init.constant_(self.smoother.weight, 1.0 / 9.0)

    def _get_8dir_gradients(self, U):
        """张量切片：极速计算 8 向物理距离校正差分"""
        U_pad = F.pad(U, (1, 1, 1, 1), mode='replicate')
        curr = U_pad[:, :, 1:-1, 1:-1]

        # 正交 4 向 (距离 1)
        dN = U_pad[:, :, 0:-2, 1:-1] - curr
        dS = U_pad[:, :, 2:, 1:-1] - curr
        dW = U_pad[:, :, 1:-1, 0:-2] - curr
        dE = U_pad[:, :, 1:-1, 2:] - curr

        # # 对角 4 向 (距离 sqrt(2))
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        dNW = (U_pad[:, :, 0:-2, 0:-2] - curr) * inv_sqrt2
        dNE = (U_pad[:, :, 0:-2, 2:] - curr) * inv_sqrt2
        dSW = (U_pad[:, :, 2:, 0:-2] - curr) * inv_sqrt2
        dSE = (U_pad[:, :, 2:, 2:] - curr) * inv_sqrt2

        return [dN, dS, dW, dE, dNW, dNE, dSW, dSE]
        # return [dN,dS,dW,dE]
    def forward(self, U, W_map):
        """
        U: [B, C, H, W] 当前输入的 2D 特征张量
        W_map: [B, 1, H, W] 由外部 Adapter 喂进来的空间绝热先验图 (0~1)
        """
        # U_init = U.clone()

        for _ in range(self.num_iters):
            # 1. 抗噪平滑梯度
            U_smoothed = self.smoother(U)
            grads_smooth = self._get_8dir_gradients(U_smoothed)

            # 2. 动态绝热阈值 K_mod (W_map 趋近 1 时，K 极小，停止扩散)
            K_mod = self.K* torch.exp(-self.gamma * W_map)
            # K_mod = self.K
            # 3. 8 向指数型各向异性扩散系数 (保证深层梯度顺畅)
            c = [torch.exp(-(g / (K_mod + 1e-6)) ** 2) for g in grads_smooth]

            # 4. 散度通量计算
            grads_real = self._get_8dir_gradients(U)
            flux_div = sum(self.dir_weights[i] * c[i] * grads_real[i] for i in range(8))

            U = U + self.dt * flux_div

        return U



# class RobustAnisotropicDiffusion(nn.Module):
#     """
#     高性能红外抗干扰版：4 向可微各向异性扩散 PDE 求解器。
#     作为纯粹的物理引擎，接收外部算好的 W_map 进行演化。
#     """
#
#     def __init__(self, in_channels, num_iters=1):
#         super().__init__()
#         self.num_iters = num_iters
#         self.in_channels = in_channels
#
#         # 物理约束的可学习参数
#         self.K = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)   # 边缘保留阈值
#         self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1) * 2.0)         # 空间先验调制强度
#         self.dt = nn.Parameter(torch.ones(1) * 0.1)                     # 演化步长
#         self.beta = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.05)  # 防漂移保真项权重
#
#         # 4 个方向的可学习通量权重
#         self.dir_weights = nn.Parameter(torch.ones(4))
#
#         # 可学习的红外抗噪平滑核
#         self.smoother = nn.Conv2d(
#             in_channels, in_channels,
#             kernel_size=3, padding=1,
#             groups=in_channels, bias=False
#         )
#         nn.init.constant_(self.smoother.weight, 1.0 / 9.0)
#
#     def _get_4dir_gradients(self, U):
#         """张量切片：计算 4 向差分梯度"""
#         U_pad = F.pad(U, (1, 1, 1, 1), mode='replicate')
#         curr = U_pad[:, :, 1:-1, 1:-1]
#
#         # 4 个正交方向
#         dN = U_pad[:, :, 0:-2, 1:-1] - curr
#         dS = U_pad[:, :, 2:,   1:-1] - curr
#         dW = U_pad[:, :, 1:-1, 0:-2] - curr
#         dE = U_pad[:, :, 1:-1, 2:]   - curr
#
#         return [dN, dS, dW, dE]
#
#     def forward(self, U, W_map):
#         """
#         U: [B, C, H, W] 当前输入的 2D 特征张量
#         W_map: [B, 1, H, W] 外部 Adapter 喂进来的空间绝热先验图 (0~1)
#         """
#         U_init = U.clone()
#
#         for _ in range(self.num_iters):
#             # 1. 抗噪平滑梯度
#             U_smoothed = self.smoother(U)
#             grads_smooth = self._get_4dir_gradients(U_smoothed)
#
#             # 2. 动态绝热阈值 K_mod
#             # W_map 趋近 1 时，K 变小，扩散受抑制
#             # K_mod = self.K * torch.exp(-self.gamma * W_map)
#             K_mod = self.K
#             # 3. 4 向指数型各向异性扩散系数
#             c = [torch.exp(-(g / (K_mod + 1e-6)) ** 2) for g in grads_smooth]
#
#             # 4. 散度通量计算
#             grads_real = self._get_4dir_gradients(U)
#             flux_div = sum(self.dir_weights[i] * c[i] * grads_real[i] for i in range(4))
#
#             # 5. 时间步进
#             # fidelity_term = -self.beta * (U - U_init)
#             # U = U + self.dt * flux_div + fidelity_term
#             U = U + self.dt * flux_div

        # return U