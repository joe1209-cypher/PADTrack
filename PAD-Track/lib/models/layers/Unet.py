import torch
import torch.nn as nn
import torch.nn.init as init


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # 残差块特殊初始化
        self._init_weights()

    def _init_weights(self):
        # 第一层卷积使用Kaiming初始化
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.zeros_(self.conv1.bias)

        # 第二层卷积使用较小方差初始化
        stdv = 0.1 * (self.conv2.weight.shape[1] ** -0.5)
        self.conv2.weight.data.uniform_(-stdv, stdv)
        init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class SimpleIRDenoiser(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        # 初始卷积
        self.enc_conv0 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 编码器
        self.enc_block1 = ResidualBlock(base_channels)
        self.downsample = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        )

        # 解码器
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1)
        self.dec_block1 = ResidualBlock(base_channels)

        # 输出层 - 关键：最后一层卷积特殊初始化
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        # 应用初始化
        self._init_weights()

    def _init_weights(self):
        # 初始化规则列表
        init_rules = [
            (self.enc_conv0, 'enc_conv0'),
            (self.bottleneck[0], 'bottleneck_conv1'),
            (self.bottleneck[2], 'bottleneck_conv2'),
            (self.dec_conv1, 'dec_conv1'),
        ]

        # 应用Kaiming初始化到指定层
        for module, name in init_rules:
            if hasattr(module, 'weight'):
                # 输出层之前的卷积使用He初始化
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

        # 输出层特殊初始化 - 零初始化确保训练起始阶段输出接近输入
        init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            init.zeros_(self.out_conv.bias)

        # 初始化瓶颈层第二个卷积
        stdv = 0.1 * (self.bottleneck[2].weight.shape[1] ** -0.5)
        self.bottleneck[2].weight.data.uniform_(-stdv, stdv)
        init.zeros_(self.bottleneck[2].bias)

    def forward(self, x):
        # 编码路径
        x0 = self.enc_conv0(x)
        x1 = self.enc_block1(x0)
        x2 = self.downsample(x1)

        # 瓶颈
        x3 = self.bottleneck(x2)

        # 解码路径
        x4 = self.upsample(x3)
        x5 = torch.cat([x4, x1], dim=1)
        x6 = self.dec_conv1(x5)
        x7 = self.dec_block1(x6)

        # 输出 (残差连接)
        return self.out_conv(x7) + x