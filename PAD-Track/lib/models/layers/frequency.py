import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyExpertLayer(nn.Module):
    """
    频率专家层 - 集成StableFrequencySelector
    创新点：多专家选择 + 稳定注意力机制
    """

    def __init__(self, in_channels, num_experts=6, temperature=0.1):
        super(FrequencyExpertLayer, self).__init__()

        self.num_experts = num_experts
        self.in_channels = in_channels

        # 创建多个频率特征选择器专家
        self.experts = nn.ModuleList([
            StableFrequencySelector(in_channels, temperature)
            for _ in range(num_experts)
        ])

        # 专家路由网络 - 轻量级，决定哪个专家处理输入
        self.routing_network = nn.Sequential(
            # 输入两个特征的拼接
            nn.Conv2d(in_channels * 2, in_channels // 8, 1),
            nn.GroupNorm(max(1, in_channels // 32), in_channels // 8),
            nn.ReLU(inplace=True),
            # 输出每个专家的权重
            nn.Conv2d(in_channels // 8, num_experts, 1),
            nn.Softmax(dim=1)
        )

        # 全局协调器 - 整合多个专家的输出
        self.coordinator = nn.Sequential(
            nn.Conv2d(in_channels * num_experts, in_channels, 1),
            nn.GroupNorm(max(1, in_channels // 8), in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # 残差连接门控
        self.residual_gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # 专家重要性权重（可学习的偏置）
        self.expert_bias = nn.Parameter(torch.zeros(1, num_experts, 1, 1))

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        # 初始化专家偏置，让所有专家初始时权重相等
        nn.init.constant_(self.expert_bias, 0.0)

        # 初始化残差门控
        nn.init.constant_(self.residual_gate, -2.0)  # 初始偏向残差连接

        # 路由网络初始化
        for m in self.routing_network.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat1, feat2):
        """
        专家层前向传播
        输入: 两个频率特征 [B, C, H, W]
        输出: 专家选择后的特征 [B, C, H, W]
        """
        B, C, H, W = feat1.shape

        # 1. 计算专家路由权重
        combined_input = torch.cat([feat1, feat2], dim=1)
        routing_weights = self.routing_network(combined_input)  # [B, num_experts, H, W]

        # 添加专家偏置，并进行softmax
        routing_weights = routing_weights + self.expert_bias
        routing_weights = F.softmax(routing_weights / 0.3, dim=1)  # 使用温度参数平滑

        # 2. 每个专家处理输入
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 获取当前专家的权重图
            expert_weight = routing_weights[:, i:i + 1, :, :]  # [B, 1, H, W]

            # 专家处理
            expert_feat = expert(feat1, feat2)  # [B, C, H, W]

            # 加权
            weighted_expert_feat = expert_feat * expert_weight

            expert_outputs.append(weighted_expert_feat)

        # 3. 整合专家输出
        if self.num_experts > 1:
            # 拼接所有专家输出
            all_expert_feats = torch.cat(expert_outputs, dim=1)  # [B, C*num_experts, H, W]
            # 协调器整合
            integrated_feat = self.coordinator(all_expert_feats)
        else:
            integrated_feat = expert_outputs[0]

        # 4. 残差连接（确保稳定性）
        residual = (feat1 + feat2) / 2
        gate = torch.sigmoid(self.residual_gate)
        output = integrated_feat * gate + residual * (1 - gate)

        # 5. 最终归一化
        output = F.layer_norm(output.permute(0, 2, 3, 1), [C]).permute(0, 3, 1, 2)

        return output # 返回输出和路由权重（用于可视化）


class StableFrequencySelector(nn.Module):
    """
    稳定的频率特征选择器
    专门设计用于解决训练不稳定问题
    """

    def __init__(self, in_channels, temperature=0.1):
        super(StableFrequencySelector, self).__init__()

        self.temperature = temperature

        # 1. 特征分析网络 - 使用GroupNorm代替BatchNorm更稳定
        self.analyzer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 4, 1),
            nn.GroupNorm(max(1, in_channels // 32), in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.GroupNorm(max(1, in_channels // 8), in_channels)
        )

        # 2. 稳定的注意力生成 - 使用可学习的偏置和缩放
        self.attention_conv = nn.Conv2d(in_channels, 2, 1)
        self.attention_bias = nn.Parameter(torch.zeros(1, 2, 1, 1))
        self.attention_scale = nn.Parameter(torch.ones(1, 2, 1, 1) * 0.5)

        # 3. 混合门控 - 控制新旧特征的混合比例
        self.gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # 4. 输出归一化
        self.norm = nn.LayerNorm([in_channels])

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """特殊的权重初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化门控参数
        nn.init.constant_(self.gate, 0.1)  # 小权重开始

    def forward(self, feat1, feat2):
        """
        稳定的前向传播
        创新点：软注意力 + 残差门控 + 温度缩放
        """
        B, C, H, W = feat1.shape

        # 1. 特征拼接和分析
        combined = torch.cat([feat1, feat2], dim=1)
        analyzed = self.analyzer(combined)

        # 2. 生成注意力权重 - 使用softmax + 温度参数
        # 温度参数使softmax更平滑，避免过大的梯度
        attention_logits = self.attention_conv(analyzed) * self.attention_scale + self.attention_bias
        attention_weights = F.softmax(attention_logits / self.temperature, dim=1)

        # 3. 特征混合 - 使用加权平均
        # 将注意力权重扩展到与特征相同的通道数
        weight1 = attention_weights[:, 0:1, :, :].repeat(1, C, 1, 1)
        weight2 = attention_weights[:, 1:2, :, :].repeat(1, C, 1, 1)

        mixed_feat = feat1 * weight1 + feat2 * weight2

        # 4. 残差连接 - 确保训练稳定性
        # 使用可学习的门控参数控制新旧特征的比例
        residual_feat = (feat1 + feat2) / 2  # 原始特征的均值作为残差
        gate_sigmoid = torch.sigmoid(self.gate)

        # 输出 = 残差 * (1 - gate) + 混合特征 * gate
        output = residual_feat * (1 - gate_sigmoid) + mixed_feat * gate_sigmoid

        # 5. 空间维度展平，进行LayerNorm，再恢复
        output = output.permute(0, 2, 3, 1)  # [B, H, W, C]
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # [B, C, H, W]

        return output