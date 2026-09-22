import  torch.nn as nn
from .kernel_warehouse import Warehouse_Manager
class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, warehouse_manager=None):
        super(CNN, self).__init__()
        self.warehouse_manager = warehouse_manager or Warehouse_Manager()
        #warehouse_manager是一个模块管理类，负责模块的创建，缓存，复用和存储
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        self.conv2 = nn.Sequential(
            self._kwconv3x3_2(inplanes, 2 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            self._kwconv3x3_3(2 * inplanes, 4 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            self._kwconv3x3_4(4 * inplanes, 4 * inplanes, stride=2, padding=1),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.warehouse_manager.store()
        self.warehouse_manager.allocate(self)

    def _kwconv3x3_2(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer2'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def _kwconv3x3_3(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer3'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def _kwconv3x3_4(self, in_planes, out_planes, stride=1, padding=1):
        warehouse_name = 'conv3x3_layer4'
        return self.warehouse_manager.reserve(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
            warehouse_name=warehouse_name, enabled=True, bias=False)

    def forward(self, x):
        B, C, H, W  = x.shape
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        # c11 = c1.cpu()
        # c11 = c11.view(1, H * W // 16, 768).mean(dim=2).squeeze().view(H // 4, W // 4)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c11, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s

        # c21 = c2.cpu()
        # c21 = c21.view(1, H * W // 64, 768).mean(dim=2).squeeze().view(H // 8, W // 8)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c21, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s

        # c31 = c3.cpu()
        # c31 = c31.view(1, H * W // 256, 768).mean(dim=2).squeeze().view(H // 16, W // 16)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c31, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        # c41 = c4.cpu()
        # c41 = c41.view(1, H * W // 1024, 768).mean(dim=2).squeeze().view(H // 32, W // 32)
        # # 绘制图像并设置值范围和插值方法
        # plt.imshow(c41, cmap='viridis', interpolation='bilinear')
        # # 添加颜色条
        # plt.colorbar()
        # # 显示图像
        # plt.show()

        return c1, c2, c3, c4