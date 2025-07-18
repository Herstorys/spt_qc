import torch
import torch.nn.functional as F
from torch import nn
from src.utils import listify_with_reference
from src.nn import Stage, PointStage, DownNFuseStage, UpNFuseStage, \
    BatchNorm, CatFusion, MLP, LayerNorm
from src.nn.pool import BaseAttentivePool
from src.nn.pool import pool_factory

__all__ = ['SPT']


class PyramidSampler(nn.Module):
    """金字塔采样模块"""

    def __init__(self, scales, fusion='concat', channels=None):
        super().__init__()
        self.scales = scales
        self.fusion = fusion
        self.channels = channels

    def forward(self, x, pos, batch_idx):
        """
        对输入进行金字塔采样

        Args:
            x: 输入特征 [N, D]
            pos: 位置信息 [N, 3]
            batch_idx: 批次索引 [N]

        Returns:
            sampled_x: 采样后的特征
            sampled_pos: 采样后的位置
        """
        pyramid_features = []
        pyramid_positions = []

        # 为每个批次分别处理
        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            batch_x = x[mask]
            batch_pos = pos[mask]

            batch_pyramid_features = []
            batch_pyramid_positions = []

            # 对每个尺度进行采样
            for scale in self.scales:
                if scale == 1.0:
                    # 保持原始尺度
                    scale_x = batch_x
                    scale_pos = batch_pos
                else:
                    # 随机采样到指定尺度
                    n_samples = max(1, int(len(batch_x) * scale))
                    if n_samples >= len(batch_x):
                        scale_x = batch_x
                        scale_pos = batch_pos
                    else:
                        # 使用FPS采样或随机采样
                        indices = self._farthest_point_sampling(batch_pos, n_samples)
                        scale_x = batch_x[indices]
                        scale_pos = batch_pos[indices]

                batch_pyramid_features.append(scale_x)
                batch_pyramid_positions.append(scale_pos)

            pyramid_features.append(batch_pyramid_features)
            pyramid_positions.append(batch_pyramid_positions)

        # 融合不同尺度的特征
        return self._fuse_pyramid_features(pyramid_features, pyramid_positions)

    def _farthest_point_sampling(self, pos, n_samples):
        """最远点采样"""
        n_points = len(pos)
        if n_samples >= n_points:
            return torch.arange(n_points, device=pos.device)

        # 简化的FPS实现
        centroids = torch.zeros(n_samples, dtype=torch.long, device=pos.device)
        distance = torch.ones(n_points, device=pos.device) * 1e10
        farthest = torch.randint(0, n_points, (1,), device=pos.device)

        for i in range(n_samples):
            centroids[i] = farthest
            centroid = pos[farthest, :]
            dist = torch.sum((pos - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def _fuse_pyramid_features(self, pyramid_features, pyramid_positions):
        """融合金字塔特征"""
        if self.fusion == 'concat':
            # 拼接所有尺度的特征
            all_features = []
            all_positions = []

            for batch_features, batch_positions in zip(pyramid_features, pyramid_positions):
                batch_concat_features = torch.cat(batch_features, dim=0)
                batch_concat_positions = torch.cat(batch_positions, dim=0)
                all_features.append(batch_concat_features)
                all_positions.append(batch_concat_positions)

            fused_features = torch.cat(all_features, dim=0)
            fused_positions = torch.cat(all_positions, dim=0)

        elif self.fusion == 'max':
            # 对应位置取最大值（需要插值到相同尺度）
            # 这里简化处理，使用最大尺度的特征
            all_features = []
            all_positions = []

            for batch_features, batch_positions in zip(pyramid_features, pyramid_positions):
                # 选择最大尺度（第一个，因为scale=1.0在第一个）
                all_features.append(batch_features[0])
                all_positions.append(batch_positions[0])

            fused_features = torch.cat(all_features, dim=0)
            fused_positions = torch.cat(all_positions, dim=0)

        else:  # 'first' - 使用第一个尺度
            all_features = []
            all_positions = []

            for batch_features, batch_positions in zip(pyramid_features, pyramid_positions):
                all_features.append(batch_features[0])
                all_positions.append(batch_positions[0])

            fused_features = torch.cat(all_features, dim=0)
            fused_positions = torch.cat(all_positions, dim=0)

        return fused_features, fused_positions


class SPT(nn.Module):
    """Superpoint Transformer with Pyramid Sampling support."""

    def __init__(
            self,

            point_mlp=None,
            point_drop=None,

            nano=False,

            down_dim=None,
            down_pool_dim=None,
            down_in_mlp=None,
            down_out_mlp=None,
            down_mlp_drop=None,
            down_num_heads=1,
            down_num_blocks=0,
            down_ffn_ratio=4,
            down_residual_drop=None,
            down_attn_drop=None,
            down_drop_path=None,

            up_dim=None,
            up_in_mlp=None,
            up_out_mlp=None,
            up_mlp_drop=None,
            up_num_heads=1,
            up_num_blocks=0,
            up_ffn_ratio=4,
            up_residual_drop=None,
            up_attn_drop=None,
            up_drop_path=None,

            node_mlp=None,
            h_edge_mlp=None,
            v_edge_mlp=None,
            mlp_activation=nn.LeakyReLU(),
            mlp_norm=BatchNorm,
            qk_dim=8,
            qkv_bias=True,
            qk_scale=None,
            in_rpe_dim=18,
            activation=nn.LeakyReLU(),
            norm=LayerNorm,
            pre_norm=True,
            no_sa=False,
            no_ffn=False,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            k_delta_rpe=False,
            q_delta_rpe=False,
            qk_share_rpe=False,
            q_on_minus_rpe=False,
            share_hf_mlps=False,
            stages_share_rpe=False,
            blocks_share_rpe=False,
            heads_share_rpe=False,

            use_pos=True,
            use_node_hf=True,
            use_diameter=False,
            use_diameter_parent=False,
            pool='max',
            unpool='index',
            fusion='cat',
            norm_mode='graph',
            output_stage_wise=False,

            # 新增金字塔采样参数
            sampling_strategy='hierarchical',  # 'hierarchical', 'pyramid', 'hybrid'
            pyramid_scales=[1.0, 0.5, 0.25],
            pyramid_fusion='concat',
            pyramid_down_enabled=True,
            pyramid_up_enabled=True):

        super().__init__()

        self.nano = nano
        self.use_pos = use_pos
        self.use_node_hf = use_node_hf
        self.use_diameter = use_diameter
        self.use_diameter_parent = use_diameter_parent
        self.norm_mode = norm_mode
        self.stages_share_rpe = stages_share_rpe
        self.blocks_share_rpe = blocks_share_rpe
        self.heads_share_rpe = heads_share_rpe
        self.output_stage_wise = output_stage_wise

        # 金字塔采样相关参数
        self.sampling_strategy = sampling_strategy
        self.pyramid_scales = pyramid_scales
        self.pyramid_fusion = pyramid_fusion
        self.pyramid_down_enabled = pyramid_down_enabled
        self.pyramid_up_enabled = pyramid_up_enabled

        # Convert input arguments to nested lists
        (
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path
        ) = listify_with_reference(
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path)

        (
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path
        ) = listify_with_reference(
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path)

        # Local helper variables describing the architecture
        num_down = len(down_dim) - self.nano
        num_up = len(up_dim)
        needs_h_edge_hf = any(x > 0 for x in down_num_blocks + up_num_blocks)
        needs_v_edge_hf = num_down > 0 and isinstance(
            pool_factory(pool, down_pool_dim[0]), BaseAttentivePool)

        # 初始化金字塔采样器
        if self.sampling_strategy in ['pyramid', 'hybrid']:
            self.pyramid_downsampler = PyramidSampler(
                pyramid_scales, pyramid_fusion) if pyramid_down_enabled else None
            self.pyramid_upsampler = PyramidSampler(
                pyramid_scales, pyramid_fusion) if pyramid_up_enabled else None
        else:
            self.pyramid_downsampler = None
            self.pyramid_upsampler = None

        # Build MLPs that will be used to process handcrafted segment
        # and edge features
        node_mlp = node_mlp if use_node_hf else None
        self.node_mlps = _build_mlps(
            node_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        h_edge_mlp = h_edge_mlp if needs_h_edge_hf else None
        self.h_edge_mlps = _build_mlps(
            h_edge_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        v_edge_mlp = v_edge_mlp if needs_v_edge_hf else None
        self.v_edge_mlps = _build_mlps(
            v_edge_mlp,
            num_down,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        # Module operating on Level-0 points in isolation
        if self.nano:
            self.first_stage = Stage(
                down_dim[0],
                num_blocks=down_num_blocks[0],
                in_mlp=down_in_mlp[0],
                out_mlp=down_out_mlp[0],
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0],
                num_heads=down_num_heads[0],
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=down_ffn_ratio[0],
                residual_drop=down_residual_drop[0],
                attn_drop=down_attn_drop[0],
                drop_path=down_drop_path[0],
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                use_pos=use_pos,
                use_diameter=use_diameter,
                use_diameter_parent=use_diameter_parent,
                blocks_share_rpe=blocks_share_rpe,
                heads_share_rpe=heads_share_rpe)
        else:
            self.first_stage = PointStage(
                point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                use_pos=use_pos,
                use_diameter_parent=use_diameter_parent)

        # Operator to append the features
        self.feature_fusion = CatFusion()

        # 根据采样策略构建不同的阶段
        if self.sampling_strategy == 'pyramid':
            self._build_pyramid_only_stages(
                down_dim, down_num_blocks, down_in_mlp, down_out_mlp,
                down_mlp_drop, down_num_heads, down_ffn_ratio,
                down_residual_drop, down_attn_drop, down_drop_path,
                up_dim, up_num_blocks, up_in_mlp, up_out_mlp,
                up_mlp_drop, up_num_heads, up_ffn_ratio,
                up_residual_drop, up_attn_drop, up_drop_path,
                qk_dim, qkv_bias, qk_scale, in_rpe_dim,
                activation, norm, pre_norm, no_sa, no_ffn,
                k_rpe, q_rpe, v_rpe, k_delta_rpe, q_delta_rpe,
                qk_share_rpe, q_on_minus_rpe, mlp_activation, mlp_norm,
                pool, unpool, fusion)
        else:
            self._build_hierarchical_stages(
                num_down, num_up, down_dim, down_pool_dim, down_in_mlp, down_out_mlp,
                down_mlp_drop, down_num_heads, down_num_blocks, down_ffn_ratio,
                down_residual_drop, down_attn_drop, down_drop_path,
                up_dim, up_in_mlp, up_out_mlp, up_mlp_drop, up_num_heads,
                up_num_blocks, up_ffn_ratio, up_residual_drop, up_attn_drop,
                up_drop_path, qk_dim, qkv_bias, qk_scale, in_rpe_dim,
                activation, norm, pre_norm, no_sa, no_ffn,
                k_rpe, q_rpe, v_rpe, k_delta_rpe, q_delta_rpe,
                qk_share_rpe, q_on_minus_rpe, mlp_activation, mlp_norm,
                pool, unpool, fusion)

        # 验证配置
        assert self.num_up_stages > 0 or not self.output_stage_wise, \
            "At least one up stage is needed for output_stage_wise=True"

        if self.sampling_strategy == 'hierarchical':
            assert bool(self.down_stages) != bool(self.up_stages) \
                   or self.num_down_stages >= self.num_up_stages, \
                "The number of Up stages should be <= the number of Down stages."
            assert self.nano or self.num_down_stages > self.num_up_stages, \
                "The number of Up stages should be < the number of Down stages."

    def _build_pyramid_only_stages(self, down_dim, down_num_blocks, down_in_mlp,
                                  down_out_mlp, down_mlp_drop, down_num_heads,
                                  down_ffn_ratio, down_residual_drop, down_attn_drop,
                                  down_drop_path, up_dim, up_num_blocks, up_in_mlp,
                                  up_out_mlp, up_mlp_drop, up_num_heads, up_ffn_ratio,
                                  up_residual_drop, up_attn_drop, up_drop_path,
                                  qk_dim, qkv_bias, qk_scale, in_rpe_dim,
                                  activation, norm, pre_norm, no_sa, no_ffn,
                                  k_rpe, q_rpe, v_rpe, k_delta_rpe, q_delta_rpe,
                                  qk_share_rpe, q_on_minus_rpe, mlp_activation, mlp_norm,
                                  pool, unpool, fusion):
        """构建纯金字塔采样架构"""
        # 简化的金字塔架构，主要用金字塔采样器处理多尺度
        self.down_stages = None
        self.up_stages = None

        # 为每个金字塔尺度构建处理器
        self.pyramid_processors = nn.ModuleList([
            Stage(
                down_dim[0] if len(down_dim) > 0 else 64,
                num_blocks=down_num_blocks[0] if len(down_num_blocks) > 0 else 1,
                in_mlp=down_in_mlp[0] if len(down_in_mlp) > 0 else None,
                out_mlp=down_out_mlp[0] if len(down_out_mlp) > 0 else None,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0] if len(down_mlp_drop) > 0 else 0.0,
                num_heads=down_num_heads[0] if len(down_num_heads) > 0 else 1,
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=down_ffn_ratio[0] if len(down_ffn_ratio) > 0 else 4,
                residual_drop=down_residual_drop[0] if len(down_residual_drop) > 0 else 0.0,
                attn_drop=down_attn_drop[0] if len(down_attn_drop) > 0 else 0.0,
                drop_path=down_drop_path[0] if len(down_drop_path) > 0 else 0.0,
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                use_pos=self.use_pos,
                use_diameter=self.use_diameter,
                use_diameter_parent=self.use_diameter_parent,
                blocks_share_rpe=self.blocks_share_rpe,
                heads_share_rpe=self.heads_share_rpe)
            for _ in self.pyramid_scales
        ])

    def _build_hierarchical_stages(self, num_down, num_up, down_dim, down_pool_dim,
                                  down_in_mlp, down_out_mlp, down_mlp_drop,
                                  down_num_heads, down_num_blocks, down_ffn_ratio,
                                  down_residual_drop, down_attn_drop, down_drop_path,
                                  up_dim, up_in_mlp, up_out_mlp, up_mlp_drop,
                                  up_num_heads, up_num_blocks, up_ffn_ratio,
                                  up_residual_drop, up_attn_drop, up_drop_path,
                                  qk_dim, qkv_bias, qk_scale, in_rpe_dim,
                                  activation, norm, pre_norm, no_sa, no_ffn,
                                  k_rpe, q_rpe, v_rpe, k_delta_rpe, q_delta_rpe,
                                  qk_share_rpe, q_on_minus_rpe, mlp_activation, mlp_norm,
                                  pool, unpool, fusion):
        """构建传统的分层架构"""
        # Transformer encoder (down) Stages operating on Level-i data
        if num_down > 0:
            # Build the RPE encoders here if shared across all stages
            down_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_down, 18, qk_dim, self.stages_share_rpe)

            down_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_down, 18, qk_dim,
                self.stages_share_rpe)

            if self.nano:
                down_k_rpe = [None] + down_k_rpe
                down_q_rpe = [None] + down_q_rpe

            self.down_stages = nn.ModuleList([
                DownNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    pool=pool_factory(pool, pool_dim),
                    fusion=fusion,
                    use_pos=self.use_pos,
                    use_diameter=self.use_diameter,
                    use_diameter_parent=self.use_diameter_parent,
                    blocks_share_rpe=self.blocks_share_rpe,
                    heads_share_rpe=self.heads_share_rpe)
                for
                    i_down,
                    (dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe,
                    pool_dim)
                in enumerate(zip(
                    down_dim,
                    down_num_blocks,
                    down_in_mlp,
                    down_out_mlp,
                    down_mlp_drop,
                    down_num_heads,
                    down_ffn_ratio,
                    down_residual_drop,
                    down_attn_drop,
                    down_drop_path,
                    down_k_rpe,
                    down_q_rpe,
                    down_pool_dim))
                if i_down >= self.nano])
        else:
            self.down_stages = None

        # Transformer decoder (up) Stages operating on Level-i data
        if num_up > 0:
            up_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_up, 18, qk_dim, self.stages_share_rpe)

            up_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_up, 18, qk_dim,
                self.stages_share_rpe)

            self.up_stages = nn.ModuleList([
                UpNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    unpool=unpool,
                    fusion=fusion,
                    use_pos=self.use_pos,
                    use_diameter=self.use_diameter,
                    use_diameter_parent=self.use_diameter_parent,
                    blocks_share_rpe=self.blocks_share_rpe,
                    heads_share_rpe=self.heads_share_rpe)
                for dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe
                in zip(
                    up_dim,
                    up_num_blocks,
                    up_in_mlp,
                    up_out_mlp,
                    up_mlp_drop,
                    up_num_heads,
                    up_ffn_ratio,
                    up_residual_drop,
                    up_attn_drop,
                    up_drop_path,
                    up_k_rpe,
                    up_q_rpe)])
        else:
            self.up_stages = None

    @property
    def num_down_stages(self):
        if self.sampling_strategy == 'pyramid':
            return len(self.pyramid_processors) if hasattr(self, 'pyramid_processors') else 0
        return len(self.down_stages) if self.down_stages is not None else 0

    @property
    def num_up_stages(self):
        if self.sampling_strategy == 'pyramid':
            return len(self.pyramid_processors) if hasattr(self, 'pyramid_processors') else 0
        return len(self.up_stages) if self.up_stages is not None else 0

    @property
    def out_dim(self):
        if self.sampling_strategy == 'pyramid':
            if hasattr(self, 'pyramid_processors') and self.pyramid_processors:
                # 为了与语义分割模块兼容，返回列表格式
                base_dim = self.pyramid_processors[0].out_dim
                if self.pyramid_fusion == 'concat':
                    # 如果是拼接融合，输出维度是所有尺度的和
                    total_dim = base_dim * len(self.pyramid_processors)
                else:
                    # 其他融合方式保持原维度
                    total_dim = base_dim

                # 返回与output_stage_wise=True格式一致的列表
                if self.output_stage_wise:
                    return [total_dim] * len(self.pyramid_scales)
                else:
                    return [total_dim]  # 包装成列表以保持一致性

            # 如果没有金字塔处理器，使用first_stage的输出维度
            return [self.first_stage.out_dim]

        if self.output_stage_wise:
            out_dim = [stage.out_dim for stage in self.up_stages][::-1]
            out_dim += [self.down_stages[-1].out_dim]
            return out_dim
        if self.up_stages is not None:
            return [self.up_stages[-1].out_dim]
        if self.down_stages is not None:
            return [self.down_stages[-1].out_dim]
        return [self.first_stage.out_dim]

    def forward(self, nag):
        if self.sampling_strategy == 'pyramid':
            return self._forward_pyramid(nag)
        else:
            return self._forward_hierarchical(nag)

    def _forward_pyramid(self, nag):
        """金字塔采样的前向传播"""
        if self.nano:
            nag = nag[1:]

        # 处理第一阶段的手工特征
        if self.nano:
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                nag[0].x = self.node_mlps[0](nag[0].x, batch=norm_index)
            if self.h_edge_mlps is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                norm_index = norm_index[nag[0].edge_index[0]]
                nag[0].edge_attr = self.h_edge_mlps[0](
                    nag[0].edge_attr, batch=norm_index)

        # 编码level-0数据
        x, diameter = self.first_stage(
            nag[0].x if self.use_node_hf else None,
            nag[0].norm_index(mode=self.norm_mode),
            pos=nag[0].pos,
            diameter=None,
            node_size=getattr(nag[0], 'node_size', None),
            super_index=nag[0].super_index,
            edge_index=nag[0].edge_index,
            edge_attr=nag[0].edge_attr)

        # 应用金字塔采样
        if self.pyramid_downsampler is not None:
            batch_idx = nag[0].norm_index(mode=self.norm_mode)
            x, sampled_pos = self.pyramid_downsampler(x, nag[0].pos, batch_idx)

            # 更新位置信息
            nag[0].pos = sampled_pos

        # 使用金字塔处理器处理不同尺度的特征
        pyramid_outputs = []
        for processor in self.pyramid_processors:
            processed_x, _ = processor(
                nag[0].x if self.use_node_hf else None,
                x,
                nag[0].norm_index(mode=self.norm_mode),
                nag[0].super_index,
                pos=nag[0].pos,
                diameter=diameter,
                node_size=getattr(nag[0], 'node_size', None),
                super_index=nag[0].super_index,
                edge_index=nag[0].edge_index,
                edge_attr=nag[0].edge_attr)
            pyramid_outputs.append(processed_x)

        # 融合金字塔输出
        if self.pyramid_fusion == 'concat':
            final_output = torch.cat(pyramid_outputs, dim=-1)
        elif self.pyramid_fusion == 'max':
            stacked_outputs = torch.stack(pyramid_outputs, dim=0)
            final_output = torch.max(stacked_outputs, dim=0)[0]
        else:  # 'first' or default
            final_output = pyramid_outputs[0]

        # 根据output_stage_wise设置返回格式
        if self.output_stage_wise:
            # 返回列表格式，每个尺度一个输出
            return pyramid_outputs if len(pyramid_outputs) > 1 else [final_output]

        return final_output

    def _forward_hierarchical(self, nag):
        """传统分层采样的前向传播（原有逻辑）"""
        if self.nano:
            nag = nag[1:]

        # Apply the first MLPs on the handcrafted features
        if self.nano:
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                nag[0].x = self.node_mlps[0](nag[0].x, batch=norm_index)
            if self.h_edge_mlps is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                norm_index = norm_index[nag[0].edge_index[0]]
                nag[0].edge_attr = self.h_edge_mlps[0](
                    nag[0].edge_attr, batch=norm_index)

        # Encode level-0 data
        x, diameter = self.first_stage(
            nag[0].x if self.use_node_hf else None,
            nag[0].norm_index(mode=self.norm_mode),
            pos=nag[0].pos,
            diameter=None,
            node_size=getattr(nag[0], 'node_size', None),
            super_index=nag[0].super_index,
            edge_index=nag[0].edge_index,
            edge_attr=nag[0].edge_attr)

        # Add the diameter to the next level's attributes
        nag[1].diameter = diameter

        # Iteratively encode level-1 and above
        down_outputs = []
        if self.nano:
            down_outputs.append(x)
        if self.down_stages is not None:

            enum = enumerate(zip(
                self.down_stages,
                self.node_mlps[int(self.nano):],
                self.h_edge_mlps[int(self.nano):],
                self.v_edge_mlps))

            for i_stage, (stage, node_mlp, h_edge_mlp, v_edge_mlp) in enum:

                # Forward on the down stage and the corresponding NAG level
                i_level = i_stage + 1

                # Process handcrafted node and edge features
                if node_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    nag[i_level].x = node_mlp(nag[i_level].x, batch=norm_index)
                if h_edge_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    norm_index = norm_index[nag[i_level].edge_index[0]]
                    edge_attr = getattr(nag[i_level], 'edge_attr', None)
                    if edge_attr is not None:
                        nag[i_level].edge_attr = h_edge_mlp(
                            edge_attr, batch=norm_index)
                if v_edge_mlp is not None:
                    norm_index = nag[i_level - 1].norm_index(mode=self.norm_mode)
                    v_edge_attr = getattr(nag[i_level], 'v_edge_attr', None)
                    if v_edge_attr is not None:
                        nag[i_level - 1].v_edge_attr = v_edge_mlp(
                            v_edge_attr, batch=norm_index)

                # Forward on the DownNFuseStage
                x, diameter = self._forward_down_stage(stage, nag, i_level, x)
                down_outputs.append(x)

                # End here if we reached the last NAG level
                if i_level == nag.num_levels - 1:
                    continue

                # Add the diameter to the next level's attributes
                nag[i_level + 1].diameter = diameter

        # Iteratively decode level-num_down_stages and below
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1
                x_skip = down_outputs[-(2 + i_stage)]
                x, _ = self._forward_up_stage(stage, nag, i_level, x, x_skip)
                up_outputs.append(x)

        # Different types of output signatures
        if self.output_stage_wise:
            out = [x] + up_outputs[::-1][1:] + [down_outputs[-1]]
            return out

        return x

    def _forward_down_stage(self, stage, nag, i_level, x):
        """下采样阶段前向传播"""
        # 应用金字塔下采样（如果启用混合模式）
        if (self.sampling_strategy == 'hybrid' and
            self.pyramid_downsampler is not None and
            self.pyramid_down_enabled):
            batch_idx = nag[i_level].norm_index(mode=self.norm_mode)
            x, pyramid_pos = self.pyramid_downsampler(x, nag[i_level].pos, batch_idx)
            nag[i_level].pos = pyramid_pos

        is_last_level = (i_level == nag.num_levels - 1)
        x_handcrafted = nag[i_level].x if self.use_node_hf else None
        return stage(
            x_handcrafted,
            x,
            nag[i_level].norm_index(mode=self.norm_mode),
            nag[i_level - 1].super_index,
            pos=nag[i_level].pos,
            diameter=nag[i_level].diameter,
            node_size=nag[i_level].node_size,
            super_index=nag[i_level].super_index if not is_last_level else None,
            edge_index=nag[i_level].edge_index,
            edge_attr=nag[i_level].edge_attr,
            v_edge_attr=nag[i_level - 1].v_edge_attr,
            num_super=nag[i_level].num_nodes)

    def _forward_up_stage(self, stage, nag, i_level, x, x_skip):
        """上采样阶段前向传播"""
        # 应用金字塔上采样（如果启用混合模式）
        if (self.sampling_strategy == 'hybrid' and
            self.pyramid_upsampler is not None and
            self.pyramid_up_enabled):
            batch_idx = nag[i_level].norm_index(mode=self.norm_mode)
            x, pyramid_pos = self.pyramid_upsampler(x, nag[i_level].pos, batch_idx)
            nag[i_level].pos = pyramid_pos

        x_handcrafted = nag[i_level].x if self.use_node_hf else None
        return stage(
            self.feature_fusion(x_skip, x_handcrafted),
            x,
            nag[i_level].norm_index(mode=self.norm_mode),
            nag[i_level].super_index,
            pos=nag[i_level].pos,
            diameter=nag[i_level - self.nano].diameter,
            node_size=nag[i_level].node_size,
            super_index=nag[i_level].super_index,
            edge_index=nag[i_level].edge_index,
            edge_attr=nag[i_level].edge_attr)


def _build_shared_rpe_encoders(
        rpe, num_stages, in_dim, out_dim, stages_share):
    """Local helper to build RPE encoders for spt. The main goal is to
    make shared encoders construction easier.

    Note that setting stages_share=True will make all stages, blocks and
    heads use the same RPE encoder.
    """
    if not isinstance(rpe, bool):
        assert stages_share, \
            "If anything else but a boolean is passed for the RPE encoder, " \
            "this value will be passed to all Stages and `stages_share` " \
            "should be set to True."
        return [rpe] * num_stages

    # If all stages share the same RPE encoder, all blocks and all heads
    # too. We copy the same module instance to be shared across all
    # stages and blocks
    if stages_share and rpe:
        return [nn.Linear(in_dim, out_dim)] * num_stages

    return [rpe] * num_stages


def _build_mlps(layers, num_stage, activation, norm, shared):
    if layers is None:
        return [None] * num_stage

    if shared:
        return nn.ModuleList([
            MLP(layers, activation=activation, norm=norm)] * num_stage)

    return nn.ModuleList([
        MLP(layers, activation=activation, norm=norm)
        for _ in range(num_stage)])
