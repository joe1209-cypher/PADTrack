import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalSourceAwareTGRI(nn.Module):
    """
    Lateral inhibition module combining [Template Catalysis] + [Thermal Source Physics].
    Core improvement: Uses divergence of infrared gradients to distinguish "real thermal sources"
    from "flat high-intensity distractors".
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        # 1. Template Catalyst
        self.catalyst_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. Reaction-Diffusion components
        self.excite = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.inhibit = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.time_step = nn.Parameter(torch.zeros(1))

        # --- Thermal Source Gate ---
        # Calculates thermal divergence to identify source structure
        # Fixed Laplacian kernel, non-learnable physical operator
        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]], dtype=torch.float32)
        # Expand to [C, 1, 3, 3] for Depthwise Conv
        self.register_buffer('laplacian_k', laplacian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

        # Map physical divergence to 0-1 gate signal
        self.source_gate_act = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),  # Simple adaptive weighting
            nn.Sigmoid()
        )

    def forward(self, search_feat, template_feat):
        residual = search_feat

        # A. Calculate Template Catalyst (Semantic/Channel Attention)
        gamma = self.catalyst_net(template_feat)

        # B. Calculate Thermal Source Constraint (Physical/Spatial Attention)
        # 1. Calculate Laplacian response (approximate thermal divergence)
        # Real thermal sources are usually local maxima, resulting in strong negative Laplacian response.
        thermal_div = F.conv2d(search_feat, self.laplacian_k, padding=1, groups=search_feat.shape[1])

        # 2. Convert to gate signal
        # Focus on areas with "significant change" (large |Laplacian|)
        # Flat backgrounds, even if bright, will result in small Laplacian and get suppressed.
        S_gate = self.source_gate_act(torch.abs(thermal_div))

        # C. Inject dual constraints
        # Features are activated only if they match [Template] AND possess [Thermal Source Structure]
        modulated_feat = search_feat * gamma * S_gate

        # D. Execute Lateral Inhibition (Reaction-Diffusion)
        u = F.relu(self.excite(modulated_feat))  # Excitatory
        v = torch.sigmoid(self.inhibit(u))  # Inhibitory

        # Competition: Excitatory - Inhibitory (Shunting)
        reaction = u - (u * v)

        # E. Update
        delta = self.project(reaction)
        out = residual + self.time_step * delta

        return F.relu(out)