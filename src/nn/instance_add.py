import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


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
            quality_metrics = self._analyze_partition_quality(
                current_partition, node_features, node_logits)

            problematic_sp = self._identify_problematic_superpoints(quality_metrics)

            if len(problematic_sp) == 0:
                break

            current_partition = self._refine_superpoints(
                current_partition, problematic_sp,
                node_features, node_logits, edge_index, edge_weights)

        return current_partition

    def _analyze_partition_quality(self, partition, node_features, node_logits):
        """分析每个超点的质量指标"""
        unique_partitions = torch.unique(partition)
        quality_metrics = {}

        for sp_id in unique_partitions:
            mask = (partition == sp_id)
            sp_features = node_features[mask]
            sp_logits = node_logits[mask]

            # 计算特征一致性
            feature_std = torch.std(sp_features, dim=0).mean()

            # 计算语义一致性
            semantic_pred = torch.argmax(sp_logits, dim=1)
            semantic_entropy = self._compute_entropy(semantic_pred)

            # 计算大小
            sp_size = mask.sum().item()

            quality_metrics[sp_id.item()] = {
                'feature_consistency': 1.0 / (1.0 + feature_std.item()),
                'semantic_consistency': 1.0 - semantic_entropy,
                'size': sp_size,
                'confidence': torch.softmax(sp_logits, dim=1).max(dim=1)[0].mean().item()
            }

        return quality_metrics

    def _compute_entropy(self, labels):
        """计算标签的熵"""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        probs = counts.float() / len(labels)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()

    def _identify_problematic_superpoints(self, quality_metrics):
        """识别需要细化的超点"""
        problematic_sp = []

        for sp_id, metrics in quality_metrics.items():
            # 判断标准：语义一致性低、特征一致性低或大小过大
            if (metrics['semantic_consistency'] < self.purity_threshold or
                metrics['feature_consistency'] < self.confidence_threshold or
                metrics['size'] > 100):  # 大小阈值可调
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

            if len(sp_indices) > 1:
                sub_clusters = self._subgraph_clustering(
                    sp_indices, node_features, node_logits, edge_index, edge_weights)

                # 重新分配标签
                for i, cluster_id in enumerate(sub_clusters):
                    if cluster_id > 0:  # 保持第一个簇的原始标签
                        refined_partition[sp_indices[i]] = next_sp_id + cluster_id - 1

                next_sp_id += sub_clusters.max().item()

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
            src, dst = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()
            adj_matrix[src, dst] = sub_edge_weights[i]
            adj_matrix[dst, src] = sub_edge_weights[i]

        # 谱聚类
        n_clusters = min(3, max(2, n_nodes // 5))  # 自适应簇数
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )

        try:
            clusters = clustering.fit_predict(adj_matrix.numpy())
            return torch.tensor(clusters, dtype=torch.long)
        except:
            # 如果聚类失败，返回原始分组
            return torch.zeros(n_nodes, dtype=torch.long)


class DifferentiableGraphClustering(MessagePassing):
    """可微分图聚类模块，基于GNN的迭代聚类"""

    def __init__(self, feature_dim, hidden_dim=128, num_iterations=5,
                 max_clusters=50, temperature=0.1):
        super().__init__(aggr='mean')
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.max_clusters = max_clusters
        self.temperature = temperature

        # 特征变换网络
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 聚类中心学习网络
        self.cluster_centers = nn.Parameter(
            torch.randn(max_clusters, hidden_dim) * 0.1
        )

        # 消息传递网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 聚类分配网络
        self.assignment_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_clusters)
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: 节点特征 [N, feature_dim]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E]
        Returns:
            soft_assignment: 软聚类分配 [N, max_clusters]
            cluster_logits: 聚类logits [N, max_clusters]
        """
        # 添加自环
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=x.size(0)
        )

        # 特征变换
        h = self.feature_transform(x)

        # 迭代消息传递和聚类更新
        for i in range(self.num_iterations):
            # 消息传递
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)

            # 更新聚类中心
            soft_assignment = F.softmax(
                self.assignment_net(h) / self.temperature, dim=1
            )
            self._update_cluster_centers(h, soft_assignment)

        # 最终聚类分配
        cluster_logits = self.assignment_net(h)
        soft_assignment = F.softmax(cluster_logits / self.temperature, dim=1)

        return soft_assignment, cluster_logits

    def message(self, x_j, edge_weight=None):
        """消息函数"""
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out, x):
        """更新函数"""
        return self.update_net(torch.cat([x, aggr_out], dim=1))

    def _update_cluster_centers(self, node_features, soft_assignment):
        """可微分的聚类中心更新"""
        # 使用软分配更新聚类中心
        cluster_weights = soft_assignment.sum(dim=0, keepdim=True).T  # [max_clusters, 1]
        weighted_features = torch.mm(soft_assignment.T, node_features)  # [max_clusters, hidden_dim]

        # 避免除零
        cluster_weights = torch.clamp(cluster_weights, min=1e-8)
        new_centers = weighted_features / cluster_weights

        # 指数移动平均更新
        momentum = 0.9
        self.cluster_centers.data = momentum * self.cluster_centers.data + (1 - momentum) * new_centers


class OptimalTransportClustering(nn.Module):
    """基于最优传输的可微分聚类"""

    def __init__(self, feature_dim, max_clusters=50, reg=0.1, max_iter=100):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_clusters = max_clusters
        self.reg = reg
        self.max_iter = max_iter

        # 聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(max_clusters, feature_dim) * 0.1
        )

        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        """
        使用Sinkhorn算法进行可微分最优传输聚类
        """
        batch_size = x.size(0)

        # 编码特征
        encoded_x = self.feature_encoder(x)

        # 计算距离矩阵
        dist_matrix = self._compute_distance_matrix(encoded_x, self.cluster_centers)

        # Sinkhorn算法求解最优传输
        transport_plan = self._sinkhorn(dist_matrix)

        return transport_plan

    def _compute_distance_matrix(self, x, centers):
        """计算特征到聚类中心的距离矩阵"""
        # x: [N, D], centers: [K, D]
        # 返回: [N, K]
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        centers_norm = (centers ** 2).sum(dim=1, keepdim=True).T  # [1, K]
        dot_product = torch.mm(x, centers.T)  # [N, K]

        distances = x_norm + centers_norm - 2 * dot_product
        return distances

    def _sinkhorn(self, cost_matrix):
        """Sinkhorn算法实现可微分最优传输"""
        N, K = cost_matrix.shape

        # 初始化
        u = torch.zeros(N, device=cost_matrix.device)
        v = torch.zeros(K, device=cost_matrix.device)

        # 目标分布（均匀分布）
        a = torch.ones(N, device=cost_matrix.device) / N
        b = torch.ones(K, device=cost_matrix.device) / K

        # Sinkhorn迭代
        for _ in range(self.max_iter):
            u_prev = u.clone()

            # 更新u
            K_tilde = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.reg)
            u = self.reg * (torch.log(a) - torch.logsumexp(
                (u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.reg, dim=1))

            # 更新v
            v = self.reg * (torch.log(b) - torch.logsumexp(
                (u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.reg, dim=0))

            # 检查收敛
            if torch.norm(u - u_prev) < 1e-6:
                break

        # 计算传输计划
        transport_plan = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.reg)

        return transport_plan


class EndToEndInstanceSegmentation(nn.Module):
    """端到端可微分实例分割模块"""

    def __init__(self, feature_dim, hidden_dim=128, max_clusters=50,
                 clustering_method='gnn', use_optimal_transport=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_clusters = max_clusters
        self.clustering_method = clustering_method
        self.use_optimal_transport = use_optimal_transport

        # 特征提取器
        self.feature_extractor = None

        # 边权重预测器
        self.edge_weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 选择聚类方法
        if clustering_method == 'gnn':
            self.clustering_module = DifferentiableGraphClustering(
                hidden_dim, hidden_dim, max_clusters=max_clusters
            )
        elif clustering_method == 'optimal_transport':
            self.clustering_module = OptimalTransportClustering(
                hidden_dim, max_clusters=max_clusters
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        # 如果同时使用最优传输作为后处理
        if use_optimal_transport and clustering_method != 'optimal_transport':
            self.ot_postprocess = OptimalTransportClustering(
                hidden_dim, max_clusters=max_clusters
            )
        else:
            self.ot_postprocess = None

    def _create_feature_extractor(self, input_dim):
        """根据实际输入维度创建特征提取器"""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, node_x, node_logits, edge_index, edge_affinity_logits=None):
        """
        端到端前向传播

        Args:
            node_x: 节点特征 [N, feature_dim]
            node_logits: 节点语义logits [N, num_classes]
            edge_index: 边索引 [2, E]
            edge_affinity_logits: 边亲和力logits [E]

        Returns:
            cluster_assignment: 聚类分配 [N, max_clusters]
            refined_features: 细化的特征 [N, hidden_dim]
            edge_weights: 边权重 [E]
            cluster_logits: 聚类logits [N, max_clusters]
        """
        # 动态创建特征提取器（仅在第一次调用时）
        if self.feature_extractor is None:
            actual_input_dim = node_x.size(1)
            self.feature_extractor = self._create_feature_extractor(actual_input_dim).to(node_x.device)
        # 提取和细化特征
        refined_features = self.feature_extractor(node_x)

        # 预测边权重
        if edge_affinity_logits is not None:
            edge_weights = torch.sigmoid(edge_affinity_logits)
        else:
            # 基于特征相似度计算边权重
            edge_features = torch.cat([
                refined_features[edge_index[0]],
                refined_features[edge_index[1]]
            ], dim=1)
            edge_weights = self.edge_weight_predictor(edge_features).squeeze()

        # 可微分聚类
        if self.clustering_method == 'gnn':
            cluster_assignment, cluster_logits = self.clustering_module(
                refined_features, edge_index, edge_weights
            )
        else:  # optimal_transport
            cluster_assignment = self.clustering_module(refined_features)
            cluster_logits = torch.log(cluster_assignment + 1e-8)

        # 可选的最优传输后处理
        if self.ot_postprocess is not None:
            cluster_assignment = self.ot_postprocess(refined_features)

        return cluster_assignment, refined_features, edge_weights, cluster_logits

    def compute_loss(self, cluster_assignment, cluster_logits, node_logits,
                    ground_truth_instances, semantic_loss_weight=1.0,
                    clustering_loss_weight=1.0, regularization_weight=0.01):
        """
        计算端到端损失

        Args:
            cluster_assignment: 预测的聚类分配 [N, max_clusters]
            cluster_logits: 聚类logits [N, max_clusters]
            node_logits: 节点语义logits [N, num_classes]
            ground_truth_instances: 真值实例标签 [N]
        """
        device = cluster_assignment.device

        # 1. 语义分割损失（保持原有的语义监督）
        semantic_loss = F.cross_entropy(node_logits, ground_truth_instances)

        # 2. 聚类一致性损失
        # 构建真值聚类分配矩阵
        unique_instances = torch.unique(ground_truth_instances)
        num_instances = len(unique_instances)

        if num_instances <= self.max_clusters:
            # 直接构建one-hot编码
            gt_assignment = torch.zeros(len(ground_truth_instances), self.max_clusters, device=device)
            for i, instance_id in enumerate(unique_instances):
                mask = (ground_truth_instances == instance_id)
                gt_assignment[mask, i] = 1.0

            clustering_loss = F.kl_div(
                F.log_softmax(cluster_logits, dim=1),
                gt_assignment,
                reduction='batchmean'
            )
        else:
            # 实例数量超过最大聚类数，使用软聚类损失
            clustering_loss = self._compute_soft_clustering_loss(
                cluster_assignment, ground_truth_instances
            )

        # 3. 正则化损失（鼓励聚类分布均匀）
        cluster_sizes = cluster_assignment.sum(dim=0)
        regularization_loss = torch.var(cluster_sizes)

        # 总损失
        total_loss = (semantic_loss_weight * semantic_loss +
                     clustering_loss_weight * clustering_loss +
                     regularization_weight * regularization_loss)

        return {
            'total_loss': total_loss,
            'semantic_loss': semantic_loss,
            'clustering_loss': clustering_loss,
            'regularization_loss': regularization_loss
        }

    def _compute_soft_clustering_loss(self, cluster_assignment, ground_truth_instances):
        """计算软聚类损失，适用于实例数量超过聚类数的情况"""
        unique_instances = torch.unique(ground_truth_instances)

        # 计算每个实例的平均聚类分配
        instance_cluster_means = {}
        for instance_id in unique_instances:
            mask = (ground_truth_instances == instance_id)
            instance_cluster_means[instance_id.item()] = cluster_assignment[mask].mean(dim=0)

        # 计算实例间的分离度和实例内的紧凑度
        separation_loss = 0
        compactness_loss = 0

        for i, instance_id in enumerate(unique_instances):
            mask = (ground_truth_instances == instance_id)
            instance_assignments = cluster_assignment[mask]
            instance_mean = instance_cluster_means[instance_id.item()]

            # 紧凑度：实例内节点应该有相似的聚类分配
            compactness_loss += F.mse_loss(instance_assignments, instance_mean.unsqueeze(0).expand_as(instance_assignments))

            # 分离度：不同实例应该有不同的聚类分配
            for j, other_instance_id in enumerate(unique_instances):
                if i != j:
                    other_mean = instance_cluster_means[other_instance_id.item()]
                    separation_loss -= F.cosine_similarity(instance_mean, other_mean, dim=0)

        return compactness_loss + separation_loss / len(unique_instances)

    def get_hard_assignment(self, cluster_assignment):
        """将软分配转换为硬分配"""
        return torch.argmax(cluster_assignment, dim=1)


class AdaptiveInstanceSegmentation(nn.Module):
    """自适应实例分割，根据输入数据选择最佳方法"""

    def __init__(self, feature_dim, hidden_dim=128, max_clusters=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_clusters = max_clusters

        # 方法选择器网络
        self.method_selector = None

        # 端到端GNN模型
        self.gnn_model = EndToEndInstanceSegmentation(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_clusters=max_clusters,
            clustering_method='gnn'
        )

        # 端到端最优传输模型
        self.ot_model = EndToEndInstanceSegmentation(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_clusters=max_clusters,
            clustering_method='optimal_transport'
        )

    def _create_method_selector(self, input_dim):
        """根据实际输入维度创建方法选择器"""
        return nn.Sequential(
            nn.Linear(input_dim + 3, self.hidden_dim),  # +3 for graph statistics
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, node_x, node_logits, edge_index, edge_affinity_logits=None,
                initial_partition=None, ground_truth_instances=None, return_loss=False):
        """
        自适应前向传播，根据输入特征选择最佳方法
        """
        # 动态创建方法选择器
        if self.method_selector is None:
            actual_input_dim = node_x.size(1)
            self.method_selector = self._create_method_selector(actual_input_dim).to(node_x.device)
        # 计算图统计信息
        num_nodes = node_x.size(0)
        num_edges = edge_index.size(1)
        avg_degree = num_edges * 2.0 / num_nodes if num_nodes > 0 else 0
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

        # 计算特征统计
        feature_mean = node_x.mean(dim=0)
        feature_std = node_x.std(dim=0).mean()

        # 构建选择器输入
        selector_input = torch.cat([
            feature_mean,
            torch.tensor([avg_degree, edge_density, feature_std], device=node_x.device)
        ]).unsqueeze(0)

        # 选择方法
        method_probs = self.method_selector(selector_input)
        selected_method = torch.argmax(method_probs, dim=1).item()

        # 根据选择的方法执行分割
        if selected_method == 0:  # 传统方法
            if initial_partition is not None:
                result = initial_partition
            else:
                # 简单的K-means作为fallback
                from sklearn.cluster import KMeans
                n_clusters = min(self.max_clusters, max(2, num_nodes // 20))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                result = torch.tensor(
                    kmeans.fit_predict(node_x.cpu().numpy()),
                    device=node_x.device, dtype=torch.long
                )

            if return_loss:
                return result, {'method_selection_loss': torch.tensor(0.0)}, {'selected_method': 0}
            return result

        elif selected_method == 1:  # 端到端GNN
            cluster_assignment, refined_features, edge_weights, cluster_logits = \
                self.gnn_model(node_x, node_logits, edge_index, edge_affinity_logits)

            hard_assignment = self.gnn_model.get_hard_assignment(cluster_assignment)

            if return_loss and ground_truth_instances is not None:
                loss_dict = self.gnn_model.compute_loss(
                    cluster_assignment, cluster_logits, node_logits, ground_truth_instances
                )
                additional_info = {
                    'selected_method': 1,
                    'cluster_assignment': cluster_assignment,
                    'refined_features': refined_features,
                    'edge_weights': edge_weights
                }
                return hard_assignment, loss_dict, additional_info

            return hard_assignment

        else:  # 端到端最优传输
            cluster_assignment, refined_features, edge_weights, cluster_logits = \
                self.ot_model(node_x, node_logits, edge_index, edge_affinity_logits)

            hard_assignment = self.ot_model.get_hard_assignment(cluster_assignment)

            if return_loss and ground_truth_instances is not None:
                loss_dict = self.ot_model.compute_loss(
                    cluster_assignment, cluster_logits, node_logits, ground_truth_instances
                )
                additional_info = {
                    'selected_method': 2,
                    'cluster_assignment': cluster_assignment,
                    'refined_features': refined_features,
                    'edge_weights': edge_weights
                }
                return hard_assignment, loss_dict, additional_info

            return hard_assignment