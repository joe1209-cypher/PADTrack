'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

import torch.nn as nn
import torch

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        # 初始化权重和偏置




    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x





def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale
class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )
        self.sigma= ElementScale(768, init_value=1e-5, requires_grad=True)

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)
        x=self.sigma(x)

        return x + residual

    def init_weights(self,m):
        """初始化模型参数"""
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,(nn.GroupNorm,nn.InstanceNorm3d)):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)