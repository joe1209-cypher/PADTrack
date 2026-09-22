import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyExpertLayer(nn.Module):
    """
    [修改版] 频率专家融合层
    专为 HighFreqTIM (高频) 和 LowFreqTIM (低频) 的融合设计。
    特点：
    1. 异构专家初始化 (高频偏好/低频偏好/平衡)。
    2. 基于高频能量的动态路由。
    """

    def __init__(self, in_channels, num_experts=6, temperature=0.1):
        super(FrequencyExpertLayer, self).__init__()

        self.num_experts = num_experts

        # --- 1. 创建异构专家组 ---
        # 我们不再随机生成专家，而是预设他们的专长
        # 假设 num_experts = 6:
        # 2个专家专注细节 (High-Focus)
        # 2个专家专注结构 (Low-Focus)
        # 2个专家专注融合 (Balanced)
        self.experts = nn.ModuleList()
        expert_types = ['high'] * 2 + ['low'] * 2 + ['balanced'] * (num_experts - 4)

        for e_type in expert_types:
            self.experts.append(
                StableFrequencySelector(in_channels, temperature, focus_type=e_type)
            )

        # --- 2. 能量感知路由网络 ---
        # 输入不仅是特征，还有"高频能量图"，帮助路由判断是否是边缘区域
        self.routing_network = nn.Sequential(
            # in_channels * 2 (特征) + 1 (能量图)
            nn.Conv2d(in_channels * 2 + 1, in_channels // 4, 1),
            nn.GroupNorm(8, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_experts, 1)
        )

        # --- 3. 协调器 (Feature Refinement) ---
        self.coordinator = nn.Sequential(
            nn.Conv2d(in_channels * num_experts, in_channels, 1),
            nn.GroupNorm(8, in_channels),
            nn.SiLU(inplace=True),  # 使用 SiLU 这种更平滑的激活函数
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # --- 4. 偏置门控 ---
        # 初始偏向 0，让网络自己学习融合比例
        self.residual_gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, high_freq_feat, low_freq_feat):
        """
        Args:
            high_freq_feat: 来自 HighFreqTIM 的输出 (边缘、纹理)
            low_freq_feat:  来自 LowFreqTIM 的输出 (热分布、结构)
        """
        B, C, H, W = high_freq_feat.shape

        # --- Step 1: 计算高频能量 (用于辅助路由) ---
        # 计算高频特征的空间能量 (L2 norm over channels)
        # 能量高的地方通常是目标边缘，需要"细节专家"介入
        high_energy = torch.mean(high_freq_feat ** 2, dim=1, keepdim=True)  # [B, 1, H, W]

        # --- Step 2: 路由决策 ---
        combined_input = torch.cat([high_freq_feat, low_freq_feat, high_energy], dim=1)
        routing_logits = self.routing_network(combined_input)

        # 使用 Gumbel-Softmax 或带温度的 Softmax 进行软路由
        routing_weights = F.softmax(routing_logits, dim=1)  # [B, num_experts, H, W]

        # --- Step 3: 专家并行处理 ---
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 获取该专家负责的权重区域
            w = routing_weights[:, i:i + 1, :, :]

            # 专家融合 (传入高低频特征)
            out = expert(high_freq_feat, low_freq_feat)

            # 加权
            expert_outputs.append(out * w)

        # --- Step 4: 整合 ---
        # 拼接并降维
        concat_feats = torch.cat(expert_outputs, dim=1)
        integrated = self.coordinator(concat_feats)

        # --- Step 5: 非对称残差连接 ---
        # 物理直觉：低频特征是图像的"底色/结构"，高频是"修饰"。
        # 因此残差的主体应该是 low_freq_feat，而不是两者的平均。
        gate = torch.sigmoid(self.residual_gate)

        # 最终输出 = 专家融合结果 * gate + 低频结构基底 * (1-gate) + 高频细节注入
        # 这种写法保证了在 gate=0 时，至少保留了基础的热红外结构和原始高频信息
        final_out = integrated * gate + low_freq_feat * (1 - gate) + high_freq_feat * 0.1

        return final_out


class StableFrequencySelector(nn.Module):
    """
    [修改版] 具备物理偏好的频率选择器
    """

    def __init__(self, in_channels, temperature=0.1, focus_type='balanced'):
        super(StableFrequencySelector, self).__init__()
        self.temperature = temperature
        self.focus_type = focus_type  # 'high', 'low', 'balanced'

        # 特征交互分析
        self.analyzer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.GroupNorm(4, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, 1)  # 输出2个通道对应两个输入的权重
        )

        # 针对不同频率特征的独立变换 (不再是简单的加权，而是变换后融合)
        self.high_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise保持细节
            nn.GroupNorm(8, in_channels)
        )
        self.low_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),  # Pointwise保持结构
            nn.GroupNorm(8, in_channels)
        )

        self._init_bias()

    def _init_bias(self):
        """根据专家类型初始化偏置，使其初始状态具备物理倾向"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 核心逻辑：修改 analyzer 最后一层的 bias，预设注意力的倾向
        final_conv = self.analyzer[-1]
        nn.init.constant_(final_conv.bias, 0)

        # bias shape: [2] -> [High_weight, Low_weight]
        if self.focus_type == 'high':
            # 初始偏好 HighFreq (索引0)
            with torch.no_grad():
                final_conv.bias[0] += 1.0
        elif self.focus_type == 'low':
            # 初始偏好 LowFreq (索引1)
            with torch.no_grad():
                final_conv.bias[1] += 1.0
        # balanced 保持为 0

    def forward(self, high_feat, low_feat):
        # 1. 生成注意力
        combined = torch.cat([high_feat, low_feat], dim=1)
        # [B, 2, H, W]
        attn_logits = self.analyzer(combined)

        # 2. 温度 Softmax
        attn_weights = F.softmax(attn_logits / self.temperature, dim=1)
        w_high = attn_weights[:, 0:1, :, :]
        w_low = attn_weights[:, 1:2, :, :]

        # 3. 变换与融合
        # 专家不仅选择，还会对特征进行微调
        high_out = self.high_transform(high_feat)
        low_out = self.low_transform(low_feat)

        # 4. 加权输出
        out = high_out * w_high + low_out * w_low

        return out