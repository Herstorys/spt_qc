import torch


class IterativePartitionRefinement:
    """迭代式超点分区细化模块"""

    def __init__(self,
                 max_iterations=3,
                 confidence_threshold=0.8,
                 purity_threshold=0.9,
                 split_size_ratio=0.3):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.purity_threshold = purity_threshold
        self.split_size_ratio = split_size_ratio

    def refine_partition(self, nag, node_features, node_logits,
                        edge_index, edge_weights):
        """迭代式分区细化主函数"""
        current_partition = nag.sub.clone()

        for iteration in range(self.max_iterations):
            # 1. 分析当前分区质量
            partition_quality = self._analyze_partition_quality(
                current_partition, node_features, node_logits)

            # 2. 识别需要细化的超点
            superpoints_to_refine = self._identify_problematic_superpoints(
                partition_quality)

            if len(superpoints_to_refine) == 0:
                break

            # 3. 对问题超点进行分割或合并
            refined_partition = self._refine_superpoints(
                current_partition, superpoints_to_refine,
                node_features, node_logits, edge_index, edge_weights)

            # 4. 更新分区
            current_partition = refined_partition

        return current_partition

    def _analyze_partition_quality(self, partition, node_features, node_logits):
        """分析每个超点的质量指标"""
        unique_partitions = torch.unique(partition)
        quality_metrics = {}

        for sp_id in unique_partitions:
            sp_mask = (partition == sp_id)
            sp_features = node_features[sp_mask]
            sp_logits = node_logits[sp_mask]

            # 计算特征一致性（特征方差）
            feature_variance = torch.var(sp_features, dim=0).mean()

            # 计算语义一致性（标签熵）
            sp_probs = torch.softmax(sp_logits, dim=1)
            mean_probs = sp_probs.mean(dim=0)
            semantic_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # 计算预测置信度
            max_probs = torch.max(sp_probs, dim=1)[0]
            confidence = max_probs.mean()

            quality_metrics[sp_id.item()] = {
                'feature_variance': feature_variance,
                'semantic_entropy': semantic_entropy,
                'confidence': confidence,
                'size': sp_mask.sum().item()
            }

        return quality_metrics

    def _identify_problematic_superpoints(self, quality_metrics):
        """识别需要细化的超点"""
        problematic_sp = []

        for sp_id, metrics in quality_metrics.items():
            # 基于多个指标判断是否需要细化
            needs_refinement = (
                metrics['feature_variance'] > 0.5 or  # 特征不一致
                metrics['semantic_entropy'] > 1.0 or   # 语义不纯
                metrics['confidence'] < self.confidence_threshold  # 置信度低
            )

            if needs_refinement and metrics['size'] > 10:  # 超点足够大才值得分割
                problematic_sp.append(sp_id)

        return problematic_sp

    def _refine_superpoints(self, partition, problematic_sp,
                           node_features, node_logits, edge_index, edge_weights):
        """对问题超点进行细化"""
        refined_partition = partition.clone()
        next_sp_id = partition.max() + 1

        for sp_id in problematic_sp:
            sp_mask = (partition == sp_id)
            sp_indices = torch.where(sp_mask)[0]

            if len(sp_indices) < 4:  # 太小的超点不分割
                continue

            # 在超点内部运行子图聚类
            sub_partition = self._subgraph_clustering(
                sp_indices, node_features, node_logits, edge_index, edge_weights)

            # 更新全局分区
            unique_sub_parts = torch.unique(sub_partition)
            for i, sub_part_id in enumerate(unique_sub_parts):
                if i == 0:  # 保持原始ID
                    continue
                sub_mask = (sub_partition == sub_part_id)
                global_indices = sp_indices[sub_mask]
                refined_partition[global_indices] = next_sp_id
                next_sp_id += 1

        return refined_partition

    def _subgraph_clustering(self, sp_indices, node_features, node_logits,
                           edge_index, edge_weights):
        """在超点内部进行子图聚类"""
        # 提取子图
        sub_features = node_features[sp_indices]
        sub_logits = node_logits[sp_indices]

        # 构建索引映射
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(sp_indices)}

        # 过滤边到子图
        mask = torch.isin(edge_index[0], sp_indices) & torch.isin(edge_index[1], sp_indices)
        sub_edge_index = edge_index[:, mask]
        sub_edge_weights = edge_weights[mask]

        # 重新映射边索引
        for i in range(sub_edge_index.shape[1]):
            sub_edge_index[0, i] = old_to_new[sub_edge_index[0, i].item()]
            sub_edge_index[1, i] = old_to_new[sub_edge_index[1, i].item()]

        # 使用简单的谱聚类进行分割
        from sklearn.cluster import SpectralClustering

        # 构建相似度矩阵
        n_nodes = len(sp_indices)
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        for i in range(sub_edge_index.shape[1]):
            src, tgt = sub_edge_index[0, i], sub_edge_index[1, i]
            weight = sub_edge_weights[i]
            adj_matrix[src, tgt] = weight
            adj_matrix[tgt, src] = weight

        # 谱聚类
        n_clusters = min(3, max(2, n_nodes // 5))  # 自适应簇数
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42)

        try:
            sub_labels = clustering.fit_predict(adj_matrix.cpu().numpy())
            return torch.from_numpy(sub_labels).to(sub_features.device)
        except:
            # 如果谱聚类失败，返回原始分区
            return torch.zeros(n_nodes, dtype=torch.long, device=sub_features.device)


class LearnablePartitionModule(torch.nn.Module):
    """可学习的超点分区模块"""

    def __init__(self,
                 feature_dim=256,
                 hidden_dim=128,
                 num_layers=3):
        super().__init__()

        # 节点特征编码器
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # 边权重预测器
        self.edge_weight_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

        # 分割决策网络
        self.split_decision_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for quality metrics
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, node_features, edge_index, current_partition):
        """前向传播"""
        # 编码节点特征
        encoded_features = self.node_encoder(node_features)

        # 预测边权重
        edge_features = torch.cat([
            encoded_features[edge_index[0]],
            encoded_features[edge_index[1]]
        ], dim=1)
        edge_weights = self.edge_weight_predictor(edge_features).squeeze()

        # 为每个超点计算质量指标和分割决策
        unique_partitions = torch.unique(current_partition)
        split_decisions = {}

        for sp_id in unique_partitions:
            sp_mask = (current_partition == sp_id)
            sp_features = encoded_features[sp_mask]

            # 计算质量指标
            feature_std = torch.std(sp_features, dim=0).mean()
            feature_mean_norm = torch.norm(sp_features.mean(dim=0))
            sp_size = sp_mask.sum().float()

            # 超点级别特征
            sp_repr = sp_features.mean(dim=0)
            quality_metrics = torch.stack([feature_std, feature_mean_norm, sp_size])

            # 分割决策
            decision_input = torch.cat([sp_repr, quality_metrics])
            split_prob = self.split_decision_net(decision_input)

            split_decisions[sp_id.item()] = split_prob

        return edge_weights, split_decisions

    def compute_partition_loss(self, predicted_weights, ground_truth_weights,
                              split_decisions, ground_truth_splits):
        """计算分区损失"""
        # 边权重损失
        weight_loss = torch.nn.functional.mse_loss(predicted_weights, ground_truth_weights)

        # 分割决策损失
        split_loss = 0
        for sp_id, pred_split in split_decisions.items():
            if sp_id in ground_truth_splits:
                gt_split = torch.tensor(ground_truth_splits[sp_id],
                                      device=pred_split.device, dtype=pred_split.dtype)
                split_loss += torch.nn.functional.binary_cross_entropy(pred_split, gt_split)

        split_loss = split_loss / len(split_decisions) if len(split_decisions) > 0 else 0

        return weight_loss + split_loss