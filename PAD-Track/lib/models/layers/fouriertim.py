import torch
import torch.nn as nn
import torch.fft


class UniversalSpectralEvolver(nn.Module):
    """
    通用神经谱演化器 (Universal Spectral Evolver)

    适用场景:
        通用视觉任务 (分类、分割、去噪、超分、目标检测)。
        不局限于红外或小目标，能够处理纹理、结构和边缘。

    原理:
        在频域中模拟通用的 PDE 演化过程: du/dt = K(w) * u
        其中 K(w) (演化核) 是由一个轻量级神经网络根据频率分布自动学习的。
        这允许模型针对不同通道学习不同的物理特性 (有的通道做平滑，有的通道做锐化)。
    """

    def __init__(self, channels, width=None, height=None):
        super(UniversalSpectralEvolver, self).__init__()

        # 如果能预知尺寸，我们可以预计算频率网格以加速
        self.pre_computed_grid = False
        if width is not None and height is not None:
            # 确保尺寸大于0
            if width > 0 and height > 0:
                self.register_buffer('omega', self._build_grid(height, width))
                self.pre_computed_grid = True

        # --- 演化核生成器 (Kernel Generator) ---
        # 这是一个"元网络" (Meta-Network)，它输入频率坐标 w，输出演化权重 K(w)
        # 使用 1x1 Conv 实现频域的 Point-wise MLP
        # 输入: 1 (频率模长 |w|)
        # 输出: C (每个通道独立的演化系数)
        self.kernel_net = nn.Sequential(
            nn.Conv2d(1, max(4, channels // 4), kernel_size=1),  # 确保至少4个通道
            nn.GELU(),
            nn.Conv2d(max(4, channels // 4), channels, kernel_size=1)
        )

        # --- 演化时间 t (Evolution Time) ---
        # 控制演化的强度。初始化为较小值，让网络从恒等映射开始学起
        self.time_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- 空间残差融合 ---
        # 保证基本的特征传递
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)
        # 初始化为 0，实现残差结构的 Zero-init 策略
        nn.init.constant_(self.post_conv.weight, 0)
        nn.init.constant_(self.post_conv.bias, 0)

    def _build_grid(self, h, w, device='cpu'):
        """构建归一化的频率坐标网格 [0, 1]"""
        # 处理 h=0 或 w=0 的情况
        if h <= 0 or w <= 0:
            return torch.zeros(1, 1, 1, 1, device=device)

        ky = torch.fft.fftfreq(h, device=device).view(-1, 1)
        kx = torch.fft.rfftfreq(w, device=device).view(1, -1)
        omega = torch.sqrt(ky ** 2 + kx ** 2)

        # 归一化到 [0, 1] 范围，确保分母不为零
        max_val = omega.max()
        if max_val == 0:
            # 如果所有频率分量都为0（例如1x1图像），返回全零
            return torch.zeros_like(omega).unsqueeze(0).unsqueeze(0)

        # 归一化并避免除零
        omega = omega / (max_val + 1e-8)
        return omega.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W_half]

    def forward(self, x):
        B, C, H, W = x.shape

        # 处理极小尺寸的输入（如1x1）
        if H <= 1 or W <= 1:
            # 对于极小尺寸的输入，直接跳过频域处理
            return x

        # 1. FFT 变换
        # x_fft: [B, C, H, W//2 + 1] (复数)
        x_fft = torch.fft.rfft2(x, norm="forward")

        # 2. 获取频率坐标 (Grid)
        if self.pre_computed_grid and self.omega.shape[-2:] == (H, W // 2 + 1):
            omega = self.omega
        else:
            omega = self._build_grid(H, W, x.device)

        # 检查omega是否全零
        if torch.all(omega == 0):
            # 如果频率网格全零，直接返回原图
            return x

        # 3. 生成动态演化核 K(w)
        # 我们希望 K(w) 仅仅依赖于频率的"距离" (各向同性假设)，或者你可以扩展为依赖 kx, ky
        # 这里输入 [1, 1, H, W_half], 输出 [1, C, H, W_half]
        # 这相当于为频谱上的每一个点，学习了一个增益系数
        evolution_kernel = self.kernel_net(omega)

        # 4. 模拟 PDE 演化
        # 频域解: u(t) = u(0) * exp(K(w) * t)
        # activation: 我们使用 tanh 来约束 K(w) 的范围，防止数值爆炸
        # time_scale: 允许正(增强)或负(抑制)

        # Transfer Function H(w)
        # 这里的 exp(tanh * t) 可以拟合低通、高通、带通、带阻等任意滤波器
        transfer_function = torch.exp(torch.tanh(evolution_kernel) * (self.time_scale + 1.0))

        # 应用滤波器 (复数乘法: 模长缩放，相位不变)
        # 注意: 我们这里只改变模长(Amplitude)，保持相位(Phase)不变
        # 保持相位对于通用视觉任务（如语义分割、位置识别）至关重要，因为相位包含结构信息
        # 使用正确的复数乘法
        x_fft_real = x_fft.real * transfer_function
        x_fft_imag = x_fft.imag * transfer_function
        x_fft_enhanced = torch.complex(x_fft_real, x_fft_imag)

        # 5. 逆变换 IFFT
        x_enhanced = torch.fft.irfft2(x_fft_enhanced, s=(H, W), norm="forward")

        # 6. 残差连接与特征校正
        # 加上原图，并通过 1x1 卷积做通道间的信息交互
        out = x + self.post_conv(x_enhanced - x)

        return out


