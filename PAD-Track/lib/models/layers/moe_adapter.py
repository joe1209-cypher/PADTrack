import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn import Attention
class moe_adapter(nn.Module):
    def __init__(self, in_features=768,out_features=768, hidden_features=8,num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.in_features=in_features
        self.hidden_futures=hidden_features
        self.out_features=out_features
        # 每个专家是一个简单的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
                nn.Linear(hidden_features,out_features)
            ) for _ in range(num_experts)
        ])
        # 门控网络（保持序列结构）
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden] = [16, 320, 768]
        bs,N,C = x.shape
        # 合并批次和序列维度
        x = x.view(-1, self.in_features)  # [16*320, 768]
        # 计算门控权重
        gate_logits = self.gate(x)  # [16*320, num_experts]
        weights = F.softmax(gate_logits, dim=-1)  # [16*320, num_experts]

        # 计算各专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [16*320, 768]
            expert_outputs.append(expert_out)

        # 组合维度 [num_experts, 16*320, 768]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 加权求和 (爱因斯坦求和简化计算)
        output = torch.bmm(weights.unsqueeze(1), expert_outputs)

        # 恢复原始维度
        out=output.view(bs, 320,768)
        return  out

