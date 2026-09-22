import torch
import torch.nn as nn
import torch.fft
import math


class PISpectralExtractor(nn.Module):
    def __init__(self, in_channels=3, learnable=True, init_lambda=0.01):
        """
        Physics-Informed Spectral Extractor (PIS-Extractor)
        替代 EVP 论文中的 "High Frequency Extraction" 模块

        Args:
            in_channels: 输入图像通道数
            learnable: 是否学习扩散系数 lambda
            init_lambda: 初始扩散系数，对应热扩散的时间步长
        """
        super().__init__()

        self.learnable = learnable

        # 定义可学习的扩散系数 lambda
        # 使用 Softplus 确保 lambda 始终 > 0，符合物理意义 (时间不能倒流)
        if self.learnable:
            self.lambda_param = nn.Parameter(torch.tensor([init_lambda] * in_channels))
        else:
            self.register_buffer('lambda_param', torch.tensor([init_lambda] * in_channels))

        self.softplus = nn.Softplus()

    def get_frequency_grid(self, H, W, device):
        """
        生成频域坐标网格 (u^2 + v^2)
        对应论文中的频域中心移动 [cite: 130]
        """
        u = torch.fft.fftfreq(H, device=device)
        v = torch.fft.fftfreq(W, device=device)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')

        # 计算圆频率 k^2 = u^2 + v^2
        # 注意：这里不需要手动 fftshift，因为 fftfreq 已经按照标准 FFT 输出排列
        k_squared = u_grid ** 2 + v_grid ** 2
        return k_squared.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            x_hfc: 物理信息增强的高频分量 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. 快速傅里叶变换 (FFT)
        # rfft2 只计算一半的频率，节省计算量且结果为复数
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 2. 构建热扩散滤波器
        # 获取对应 rfft2 的频域网格
        u = torch.fft.fftfreq(H, device=x.device)
        v = torch.fft.rfftfreq(W, device=x.device)  # 注意 rfftfreq
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        k_squared = (u_grid ** 2 + v_grid ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W_half]

        # 3. 计算 High-Pass Filter: 1 - exp(-lambda * k^2)
        # 这里的 lambda 控制热扩散的程度
        # lambda 越大 -> 扩散越强 -> (1-lowpass) 保留的高频越多（包含中频）
        # lambda 越小 -> 扩散越弱 -> (1-lowpass) 只保留极高频边缘
        curr_lambda = self.softplus(self.lambda_param).view(1, C, 1, 1)

        # 物理滤波器公式
        # explicit_filter 形状: [1, C, H, W_half]
        thermal_filter = 1.0 - torch.exp(-curr_lambda * k_squared * (H * W))

        # 4. 应用滤波器
        x_fft_filtered = x_fft * thermal_filter

        # 5. 傅里叶逆变换 (IFFT)
        x_hfc = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')

        # 残差连接 (可选，EVP论文中主要是直接用HFC作为Prompt)
        # 如果需要完全保留原图语义只加边缘，可以直接返回 x_hfc
        return x_hfc