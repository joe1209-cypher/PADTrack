import torch
import torch.nn as nn
import torch.fft


class SpectralPDEEnhancer(nn.Module):
    """
    纯图像域的频域 PDE 增强模块 (Image-to-Image)

    功能:
        输入原始红外图像 [B, C, H, W]，输出边缘与细节增强后的图像 [B, C, H, W]。
        不改变张量维度，不进行 Patch Embedding。

    适用场景:
        放置在 Backbone 之前，作为可学习的预处理层。
    """

    def __init__(self, channels=3, alpha_init=0.8):
        super(SpectralPDEEnhancer, self).__init__()

        # --- PDE 参数 (物理意义) ---
        # 1. Alpha: 分数阶微分阶数 (控制锐化对纹理的敏感度)
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * alpha_init)

        # 2. Gain: 扩散强度 (控制增强的幅度)
        # 初始化为0，确保训练开始时等同于恒等映射 (Identity)，保证稳定性
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # 3. Sigma: 正则化系数 (抑制高频噪声)
        self.sigma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.001)

        # --- 门控参数 ---
        # 这是一个标量或通道向量，用来控制最终叠加回原图的比例
        # 类似于 ResNet 的残差权重，但我们让它可学习
        self.res_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def get_frequency_grid(self, h, w, device):
        """生成频域坐标网格 |w|"""
        ky = torch.fft.fftfreq(h, device=device).view(-1, 1)
        kx = torch.fft.rfftfreq(w, device=device).view(1, -1)
        omega = torch.sqrt(ky ** 2 + kx ** 2)
        return omega.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        输入: x [B, C, H, W] (原始红外图像)
        输出: out [B, C, H, W] (增强后的红外图像)
        """
        B, C, H, W = x.shape

        # 1. RFFT2: 转到频域
        fft_x = torch.fft.rfft2(x, norm="forward")

        # 2. 分离幅度与相位
        # 严格保留相位 (Phase)，这对跟踪任务的定位精度至关重要
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # 3. 计算 PDE 算子
        omega = self.get_frequency_grid(H, W, x.device)

        # 参数约束 (防止数值爆炸)
        curr_alpha = torch.sigmoid(self.alpha) * 3.0 + 0.1  # 范围 (0.1, 3.1)
        curr_gain = torch.tanh(self.gain)                   # 范围 (-1, 1)
        curr_sigma = torch.abs(self.sigma) + 1e-6           # 范围 > 0

        # 构建滤波器 H(w)
        # 锐化项 (High-pass like)
        diff_term = torch.pow(omega + 1e-6, curr_alpha)
        # 平滑项 (Low-pass gaussian)
        smooth_term = torch.exp(-curr_sigma * (omega ** 2))

        # 完整的 PDE 演化算子
        pde_operator = curr_gain * diff_term * smooth_term

        # 4. 幅度调制
        # 仅修改幅度谱: A_new = A_old + A_old * Operator
        enhanced_amplitude = amplitude * (1.0 + pde_operator)

        # 5. 重建与逆变换
        enhanced_fft = torch.polar(enhanced_amplitude, phase)
        enhanced_residual = torch.fft.irfft2(enhanced_fft, s=(H, W), norm="forward")

        # 6. 残差连接 (Residual Connection)
        # 输出 = 原图 + 学习到的系数 * (增强后的图 - 原图)
        # 注意：实际上 enhanced_residual 已经是重建后的图像了
        # 我们这里计算纯粹的"高频增量"部分 (Delta)
        # Delta = Reconstruction - Original
        # 这样做是为了让 self.res_scale 精确控制注入的纹理量
        delta = enhanced_residual - x

        # 最终输出
        out = x + self.res_scale * delta

        return out