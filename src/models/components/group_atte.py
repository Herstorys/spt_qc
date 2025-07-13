import torch
from torch import nn
import torch.nn.functional as F


class GroupAttention(nn.Module):
    """分组注意力模块"""

    def __init__(self, embed_dim, num_heads, num_groups, group_mode='spatial'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.group_mode = group_mode
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

        self.heads_per_group = num_heads // num_groups

        # 为每个组创建独立的注意力权重
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 组分配网络
        if group_mode == 'learnable':
            self.group_assignment = nn.Linear(embed_dim, num_groups)

    def _assign_groups(self, x, pos=None):
        """根据不同策略将节点分配到组"""
        batch_size, seq_len, _ = x.size()

        if self.group_mode == 'spatial' and pos is not None:
            # 基于空间位置的分组
            pos_norm = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
            # 使用K-means风格的分组
            centroids = pos_norm.mean(dim=0, keepdim=True).repeat(self.num_groups, 1, 1)
            distances = torch.cdist(pos_norm.unsqueeze(0), centroids)
            group_ids = distances.argmin(dim=-1)
        elif self.group_mode == 'feature':
            # 基于特征相似性的分组
            x_norm = F.normalize(x, dim=-1)
            # 简单的特征聚类
            centroids = x_norm.mean(dim=1, keepdim=True).repeat(1, self.num_groups, 1)
            similarities = torch.bmm(x_norm, centroids.transpose(-2, -1))
            group_ids = similarities.argmax(dim=-1)
        elif self.group_mode == 'learnable':
            # 可学习的分组
            group_logits = self.group_assignment(x)
            group_ids = group_logits.argmax(dim=-1)
        else:  # random
            # 随机分组
            group_ids = torch.randint(0, self.num_groups, (batch_size, seq_len), device=x.device)

        return group_ids

    def forward(self, x, pos=None, edge_index=None, edge_attr=None):
        batch_size, seq_len, embed_dim = x.size()

        # 生成Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 分配组
        group_ids = self._assign_groups(x, pos)

        # 为每个组执行注意力
        outputs = []
        for group_idx in range(self.num_groups):
            # 获取当前组的头
            head_start = group_idx * self.heads_per_group
            head_end = head_start + self.heads_per_group

            group_q = q[:, :, head_start:head_end, :]  # [B, S, H_g, D]
            group_k = k[:, :, head_start:head_end, :]
            group_v = v[:, :, head_start:head_end, :]

            # 获取当前组的节点掩码
            group_mask = (group_ids == group_idx).unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]

            # 计算注意力权重
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_weights = torch.matmul(group_q, group_k.transpose(-2, -1)) * scale

            # 应用组掩码 - 只允许组内注意力
            group_attn_mask = torch.matmul(group_mask.float(), group_mask.transpose(-2, -1).float())
            attn_weights = attn_weights.masked_fill(group_attn_mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1)
            group_output = torch.matmul(attn_weights, group_v)

            outputs.append(group_output)

        # 合并所有组的输出
        output = torch.cat(outputs, dim=2)  # [B, S, H, D]
        output = output.view(batch_size, seq_len, embed_dim)

        return self.out_proj(output)