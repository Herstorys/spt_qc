import torch
from torch import nn
import torch.nn.functional as F


class GroupAttention(nn.Module):
    """分组注意力模块，支持SPT的2D张量格式 (N, C)"""

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
        """根据不同策略将节点分配到组，x: (N, C), pos: (N, 3)"""
        N = x.size(0)

        if self.group_mode == 'spatial' and pos is not None:
            # 基于空间位置的分组：将节点分配到最近的空间质心
            pos_norm = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
            # 均匀初始化 num_groups 个质心（沿第一主轴等分）
            idx_centroids = torch.linspace(0, N - 1, self.num_groups,
                                           dtype=torch.long, device=x.device)
            centroids = pos_norm[idx_centroids]  # (G, 3)
            distances = torch.cdist(pos_norm, centroids)  # (N, G)
            group_ids = distances.argmin(dim=-1)  # (N,)
        elif self.group_mode == 'feature':
            # 基于特征相似性的分组
            x_norm = F.normalize(x, dim=-1)  # (N, C)
            idx_centroids = torch.linspace(0, N - 1, self.num_groups,
                                           dtype=torch.long, device=x.device)
            centroids = x_norm[idx_centroids]  # (G, C)
            similarities = torch.mm(x_norm, centroids.t())  # (N, G)
            group_ids = similarities.argmax(dim=-1)  # (N,)
        elif self.group_mode == 'learnable':
            # 可学习的分组
            group_logits = self.group_assignment(x)  # (N, G)
            group_ids = group_logits.argmax(dim=-1)  # (N,)
        else:  # random
            group_ids = torch.randint(0, self.num_groups, (N,), device=x.device)

        return group_ids

    def forward(self, x, pos=None, edge_index=None, edge_attr=None):
        """
        :param x: (N, C) — SPT节点特征（2D格式）
        :param pos: (N, 3) — 节点位置，用于空间分组
        """
        N, embed_dim = x.size()

        # 生成Q, K, V: (N, H, D)
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # 分配组: (N,)
        group_ids = self._assign_groups(x, pos)

        scale = 1.0 / (self.head_dim ** 0.5)
        output = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)

        for group_idx in range(self.num_groups):
            # 当前组的节点索引
            mask = (group_ids == group_idx)  # (N,)
            if mask.sum() == 0:
                continue

            # 当前组使用的注意力头范围
            head_start = group_idx * self.heads_per_group
            head_end = head_start + self.heads_per_group

            group_q = q[mask, head_start:head_end, :]  # (Ng, Hg, D)
            group_k = k[mask, head_start:head_end, :]
            group_v = v[mask, head_start:head_end, :]

            # 计算组内注意力: (Ng, Hg, Ng) -> (Ng, Hg, D)
            attn_weights = torch.einsum('ihd,jhd->ihj', group_q, group_k) * scale  # (Ng, Hg, Ng)
            attn_weights = F.softmax(attn_weights, dim=-1)
            group_output = torch.einsum('ihj,jhd->ihd', attn_weights, group_v)  # (Ng, Hg, D)

            output[mask, head_start:head_end, :] = group_output

        output = output.view(N, embed_dim)
        return self.out_proj(output)
