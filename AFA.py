import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_layer(in_dim, out_dim, use_relu=True):
    """创建线性层"""
    layers = [
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim)
    ]
    if use_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class AFA(nn.Module):
    """
    时间特征融合模块 - 适用于BTD维度
    输入: x: (B, T, D)
    输出: (B, T, D)
    """

    def __init__(self, d_model, hidden_dim=None):
        super(AFA, self).__init__()
        if hidden_dim is None:
            hidden_dim = d_model

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # 时间差异计算的投影层
        self.diff_proj = linear_layer(d_model, hidden_dim)

        # 特征融合层
        self.fusion_proj_A = linear_layer(d_model + hidden_dim, hidden_dim)
        self.fusion_proj_B = linear_layer(d_model + hidden_dim, hidden_dim)

        # 门控机制
        self.gate_A = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gate_B = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 最终输出层
        self.output_proj = linear_layer(hidden_dim * 2, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # 计算时间步之间的差异
        if T > 1:
            # 相邻时间步差异
            x_shifted = torch.roll(x, shifts=1, dims=1)  # 向右移动一位
            x_diff = x - x_shifted  # (B, T, D)
            x_diff[:, 0, :] = 0  # 第一个时间步没有前一步，设为0
        else:
            x_diff = torch.zeros_like(x)

        # 投影时间差异
        diff_features = self.diff_proj(x_diff)  # (B, T, hidden_dim)

        # 构造A和B特征（当前帧和差异特征的融合）
        x_concat_A = torch.cat([x, diff_features], dim=-1)  # (B, T, D + hidden_dim)
        x_concat_B = torch.cat([x, diff_features], dim=-1)  # (B, T, D + hidden_dim)

        # 融合特征
        x_fused_A = self.fusion_proj_A(x_concat_A)  # (B, T, hidden_dim)
        x_fused_B = self.fusion_proj_B(x_concat_B)  # (B, T, hidden_dim)

        # 门控机制
        gate_weight_A = self.gate_A(x_fused_A)  # (B, T, 1)
        gate_weight_B = self.gate_B(x_fused_B)  # (B, T, 1)

        # 应用门控
        x_gated_A = gate_weight_A * x_fused_A  # (B, T, hidden_dim)
        x_gated_B = gate_weight_B * x_fused_B  # (B, T, hidden_dim)

        # 拼接并输出
        x_combined = torch.cat([x_gated_A, x_gated_B], dim=-1)  # (B, T, hidden_dim * 2)
        output = self.output_proj(x_combined)  # (B, T, D)

        return output




