import torch
import torch.nn as nn
from einops import rearrange



class MTM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MTM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        # self.qkv2 = nn.Conv1d(in_channels=D, out_channels=3*D, kernel_size=1, bias=bias)


        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
          
          
        self.project_out = nn.Linear(dim, dim, bias=bias)


        self.context_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias),
        )


        self.fc = nn.Linear(3 * self.num_heads, 9, bias=True)


        self.dep_conv = nn.Conv1d(9 * dim // self.num_heads, dim, kernel_size=3, bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        """
        输入: x, shape (B, T, D)
        输出: out, shape (B, T, D)
        """
        b, t, d = x.shape

        # 获取 QKV 特征
        qkv = self.qkv(x)  # (B, T, 3*D)

        # 时间维度卷积需要转置为 (B, 3*D, T)
        qkv_conv = qkv.transpose(1, 2)  # (B, 3*D, T)
        qkv_conv = self.qkv_dwconv(qkv_conv)  # (B, 3*D, T)
        qkv = qkv_conv.transpose(1, 2)  # (B, T, 3*D)

        # 分支1:short branch
        f_all = qkv.reshape(b, t, 3 * self.num_heads, -1)  # (B, T, 3*heads, D//heads)
        f_all = self.fc(f_all.transpose(-2, -1))  # (B, T, D//heads, 9)
        f_all = f_all.transpose(-2, -1)  # (B, T, 9, D//heads)

        f_conv = f_all.reshape(b, t, 9 * d // self.num_heads)  # (B, T, 9*D//heads)
        f_conv = f_conv.transpose(1, 2)  # (B, 9*D//heads, T)
        out_conv = self.dep_conv(f_conv)  # (B, D, T)
        out_conv = out_conv.transpose(1, 2)  # (B, T, D)

        # 分支2:long branch
        q, k, v = qkv.chunk(3, dim=-1)  # 每个: (B, T, D)
        q = rearrange(q, 'b t (head c) -> b head c t', head=self.num_heads)
        k = rearrange(k, 'b t (head c) -> b head c t', head=self.num_heads)
        v = rearrange(v, 'b t (head c) -> b head c t', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out_attention = (attn @ v)
        out_attention = rearrange(out_attention, 'b head c t -> b t (head c)', head=self.num_heads)


        # 分支3:mid branch
        x_temporal = x.transpose(1, 2)  # (B, D, T)
        regional_context = self.context_conv(x_temporal)  # (B, D, T)
        regional_context = regional_context.transpose(1, 2)  # (B, T, D)

        # 融合三个分支的输出
        out = regional_context + out_conv + out_attention
        return out

