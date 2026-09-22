import torch.nn as nn
import torch
import math
from mmcv.cnn import build_norm_layer

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size//2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out

class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = build_norm_layer(norm_layer, self.outdim)[1]

    def forward(self, x):  # x = [B, C, H, W]
        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 2C, H/2, W/2]
        return x

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = (xx ** 2 + yy ** 2 - 2 * sigma ** 2) / (2 * math.pi * sigma ** 4) * torch.exp(
            -(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                             groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]

    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x

class Stem(nn.Module):
    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd1 = DRFD(out_c12, norm_layer, act_layer)
        self.drfd2 = DRFD(out_c12 * 2, norm_layer, act_layer)
        self.drfd3 = DRFD(out_c12 * 4, norm_layer, act_layer)
    def forward(self, x):
        x = self.LoG(x)
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd1(x)
        x = self.drfd2(x)
        x = self.drfd3(x)
        return x

if __name__ == '__main__':
    attn = Stem(in_chans=3, stem_dim=24, act_layer=nn.ReLU, norm_layer=dict(type='BN', requires_grad=True))
    x = torch.randn(8, 3, 256, 256)
    output = attn(x)
    print(output.shape)  # Expected output shape: torch.Size([8, 96, 16, 16])