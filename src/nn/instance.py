import torch
from torch import nn
from copy import copy
from itertools import product
from src.utils.instance import instance_cut_pursuit
from src.nn.instance_add import (
    IterativePartitionRefinement,
    EndToEndInstanceSegmentation,
    AdaptiveInstanceSegmentation
)
import torch.nn.functional as F

__all__ = ['InstancePartitioner', 'DifferentiableInstancePartitioner', 'AdaptiveInstancePartitioner']


class InstancePartitioner(nn.Module):
    """Partition a graph into instances using cut-pursuit.

    Extended to support differentiable clustering methods for end-to-end training.
    """

    def __init__(
            self,
            loss_type='l2_kl',
            regularization=10,
            x_weight=1e-2,
            p_weight=1,
            cutoff=1,
            parallel=True,
            iterations=10,
            trim=False,
            discrepancy_epsilon=1e-4,
            temperature=1,
            dampening=0,
            # 新增参数
            use_iterative_refinement=False,
            use_end_to_end=False,
            refinement_params=None,
            # 端到端参数
            feature_dim=None,
            hidden_dim=128,
            max_clusters=50,
            clustering_method='gnn',
            use_optimal_transport=False
    ):
        super().__init__()
        self.loss_type = loss_type
        self.regularization = regularization
        self.x_weight = x_weight
        self.p_weight = p_weight
        self.cutoff = cutoff
        self.parallel = parallel
        self.iterations = iterations
        self.trim = trim
        self.discrepancy_epsilon = discrepancy_epsilon
        self.temperature = temperature
        self.dampening = dampening

        # 改进功能参数
        self.use_iterative_refinement = use_iterative_refinement
        self.use_end_to_end = use_end_to_end
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_clusters = max_clusters
        self.clustering_method = clustering_method
        self.use_optimal_transport = use_optimal_transport

        self.refinement_params = refinement_params or {
            'max_iterations': 3,
            'confidence_threshold': 0.8,
            'purity_threshold': 0.9
        }

        # 初始化模型
        self.refinement_module = None
        self.end_to_end_model = None

        # 如果启用迭代细化
        if use_iterative_refinement:
            self.refinement_module = IterativePartitionRefinement(**self.refinement_params)

        # 如果指定了特征维度，立即初始化端到端模型
        if feature_dim is not None and use_end_to_end:
            self._initialize_end_to_end_model()

    def _initialize_end_to_end_model(self):
        """初始化端到端可微分模型"""
        if self.use_end_to_end and self.feature_dim is not None:
            self.end_to_end_model = EndToEndInstanceSegmentation(
                feature_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                max_clusters=self.max_clusters,
                clustering_method=self.clustering_method,
                use_optimal_transport=self.use_optimal_transport
            )

    def _dynamic_initialize(self, node_x):
        """根据输入特征动态初始化模型"""
        if self.feature_dim is None:
            self.feature_dim = node_x.shape[1]
            if self.use_end_to_end:
                self._initialize_end_to_end_model()

    def forward(
            self,
            batch,
            node_x,
            node_logits,
            stuff_classes,
            node_size,
            edge_index,
            edge_affinity_logits,
            grid=None,
            ground_truth_instances=None,
            return_loss=False):
        """Extended forward method supporting end-to-end differentiable clustering.

        :param ground_truth_instances: Tensor of shape [num_nodes]
            Ground truth instance labels for end-to-end training
        :param return_loss: bool
            Whether to return loss information for training
        """
        # 动态初始化（如果需要）
        if self.use_end_to_end and self.end_to_end_model is None:
            self._dynamic_initialize(node_x)

        # If grid is passed, multiple partition will be computed on the parameter grid
        if grid is not None and len(grid) > 0:
            return self._grid_forward(
                batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits, grid,
                ground_truth_instances, return_loss)

        # 选择使用的方法
        if self.use_end_to_end and self.end_to_end_model is not None:
            return self._forward_end_to_end(
                batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits,
                ground_truth_instances, return_loss)
        else:
            # 使用传统方法
            return self._forward_traditional(
                batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits)

    def _forward_traditional(self, batch, node_x, node_logits, stuff_classes,
                           node_size, edge_index, edge_affinity_logits):
        """传统cut-pursuit方法"""
        partition = instance_cut_pursuit(
            batch, node_x, node_logits, stuff_classes,
            node_size, edge_index, edge_affinity_logits,
            loss_type=self.loss_type,
            regularization=self.regularization,
            x_weight=self.x_weight,
            p_weight=self.p_weight,
            cutoff=self.cutoff,
            parallel=self.parallel,
            iterations=self.iterations,
            trim=self.trim,
            discrepancy_epsilon=self.discrepancy_epsilon,
            temperature=self.temperature,
            dampening=self.dampening)

        # 应用迭代细化（如果启用）
        if self.use_iterative_refinement and self.refinement_module is not None:
            # 这里需要根据实际的NAG结构调整
            # partition = self.refinement_module.refine_partition(
            #     nag, node_x, node_logits, edge_index, edge_affinity_logits)
            pass

        return partition

    def _forward_end_to_end(self, batch, node_x, node_logits, stuff_classes,
                          node_size, edge_index, edge_affinity_logits,
                          ground_truth_instances=None, return_loss=False):
        """端到端可微分前向传播"""
        # 使用端到端模型进行聚类
        cluster_assignment, refined_features, edge_weights, cluster_logits = \
            self.end_to_end_model(node_x, node_logits, edge_index, edge_affinity_logits)

        # 获取硬分配
        hard_assignment = self.end_to_end_model.get_hard_assignment(cluster_assignment)

        # 应用stuff类合并逻辑
        final_partition = self._apply_stuff_merging(
            hard_assignment, node_logits, stuff_classes, batch)

        if return_loss and ground_truth_instances is not None:
            # 计算端到端损失
            loss_dict = self.end_to_end_model.compute_loss(
                cluster_assignment, cluster_logits, node_logits, ground_truth_instances)

            additional_info = {
                'cluster_assignment': cluster_assignment,
                'refined_features': refined_features,
                'edge_weights': edge_weights,
                'soft_assignment': cluster_assignment
            }

            return final_partition, loss_dict, additional_info

        return final_partition

    def _apply_stuff_merging(self, partition, node_logits, stuff_classes, batch):
        """应用stuff类合并逻辑，保持与原有逻辑一致"""
        from src.utils.instance import get_stuff_mask, scatter_mean_weighted, consecutive_cluster

        # 计算每个分区的平均logits
        node_size = torch.ones_like(partition, dtype=torch.float)
        obj_logits = scatter_mean_weighted(node_logits, partition, node_size)
        obj_y = obj_logits.argmax(dim=1)
        obj_is_stuff = get_stuff_mask(obj_y, stuff_classes)

        node_obj_y = obj_y[partition]
        node_is_stuff = obj_is_stuff[partition]

        batch = batch if batch is not None else torch.zeros_like(partition)
        num_batch_items = batch.max() + 1
        final_obj_index = partition.clone()
        final_obj_index[node_is_stuff] = \
            partition.max() + 1 \
            + node_obj_y[node_is_stuff] * num_batch_items \
            + batch[node_is_stuff]
        final_obj_index, perm = consecutive_cluster(final_obj_index)

        return final_obj_index

    def _grid_forward(self, batch, node_x, node_logits, stuff_classes,
                     node_size, edge_index, edge_affinity_logits, grid,
                     ground_truth_instances=None, return_loss=False):
        """Run multiple forward calls for grid-searching optimal settings."""
        # 扩展支持的网格搜索参数
        supported_params = {
            'regularization', 'x_weight', 'p_weight', 'cutoff',
            'parallel', 'iterations', 'trim', 'discrepancy_epsilon',
            'temperature', 'dampening', 'use_end_to_end', 'clustering_method'
        }

        keys = list(grid.keys())
        for k in keys:
            if k not in supported_params:
                raise ValueError(
                    f"'{k}' is not a supported grid search parameter for {self.__class__.__name__}")

        # Backup the current attributes
        attr_bckp = copy(self.__dict__)
        grid_outputs = []

        for values in product(*grid.values()):
            # Update attributes
            for k, v in zip(keys, values):
                setattr(self, k, v)

            # Re-initialize models if needed
            if 'use_end_to_end' in keys or 'clustering_method' in keys:
                if self.use_end_to_end:
                    self._initialize_end_to_end_model()

            # Compute partition
            result = self.forward(
                batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits,
                grid=None, ground_truth_instances=ground_truth_instances,
                return_loss=return_loss)

            grid_outputs.append(({k: v for k, v in zip(keys, values)}, result))

        # Restore the initial attributes
        for k, v in attr_bckp.items():
            setattr(self, k, v)

        return grid_outputs

    def enable_end_to_end(self, clustering_method='gnn', max_clusters=50):
        """启用端到端可微分聚类"""
        self.use_end_to_end = True
        self.clustering_method = clustering_method
        self.max_clusters = max_clusters
        if self.feature_dim is not None:
            self._initialize_end_to_end_model()

    def disable_end_to_end(self):
        """禁用端到端聚类"""
        self.use_end_to_end = False
        self.end_to_end_model = None

    def get_trainable_parameters(self):
        """获取可训练参数"""
        params = []
        if self.end_to_end_model is not None:
            params.extend(list(self.end_to_end_model.parameters()))
        return params

    def extra_repr(self) -> str:
        keys = [
            'regularization', 'x_weight', 'cutoff', 'parallel',
            'iterations', 'trim', 'discrepancy_epsilon',
            'use_end_to_end', 'clustering_method'
        ]
        return ', '.join([f'{k}={getattr(self, k)}' for k in keys])


class DifferentiableInstancePartitioner(InstancePartitioner):
    """专门用于端到端训练的可微分实例分区器"""

    def __init__(self, feature_dim, hidden_dim=128, max_clusters=50,
                 clustering_method='gnn', use_optimal_transport=False,
                 semantic_loss_weight=1.0, clustering_loss_weight=1.0,
                 regularization_weight=0.01, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_clusters=max_clusters,
            clustering_method=clustering_method,
            use_end_to_end=True,
            use_optimal_transport=use_optimal_transport,
            **kwargs
        )

        self.semantic_loss_weight = semantic_loss_weight
        self.clustering_loss_weight = clustering_loss_weight
        self.regularization_weight = regularization_weight

    def forward(self, batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits,
                ground_truth_instances=None, return_loss=False, **kwargs):
        """专门的端到端前向传播"""
        # 强制使用端到端方法
        return self._forward_end_to_end(
            batch, node_x, node_logits, stuff_classes,
            node_size, edge_index, edge_affinity_logits,
            ground_truth_instances, return_loss)

    def compute_panoptic_loss(self, predictions, targets, node_logits):
        """计算全景分割专用损失"""
        device = predictions.device

        # 语义分割损失
        semantic_loss = F.cross_entropy(node_logits, targets)

        # 实例分割质量损失
        instance_loss = self._compute_instance_quality_loss(predictions, targets)

        # 总损失
        total_loss = (self.semantic_loss_weight * semantic_loss +
                     self.clustering_loss_weight * instance_loss)

        return {
            'total_loss': total_loss,
            'semantic_loss': semantic_loss,
            'instance_loss': instance_loss
        }

    def _compute_instance_quality_loss(self, predictions, targets):
        """计算实例分割质量损失"""
        unique_targets = torch.unique(targets)
        total_loss = 0

        for target_id in unique_targets:
            target_mask = (targets == target_id)
            pred_labels = predictions[target_mask]

            # 计算该实例的纯度
            if len(pred_labels) > 0:
                unique_preds, counts = torch.unique(pred_labels, return_counts=True)
                max_count = counts.max()
                purity = max_count.float() / len(pred_labels)
                total_loss += (1 - purity) ** 2

        return total_loss / len(unique_targets) if len(unique_targets) > 0 else torch.tensor(0.0, device=predictions.device)


class AdaptiveInstancePartitioner(InstancePartitioner):
    """自适应实例分区器，根据输入数据特征选择最佳方法"""

    def __init__(self, feature_dim, hidden_dim=128, max_clusters=50, **kwargs):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_clusters=max_clusters,
            **kwargs
        )

        # 创建自适应模型
        self.adaptive_model = AdaptiveInstanceSegmentation(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_clusters=max_clusters
        )

    def forward(self, batch, node_x, node_logits, stuff_classes,
                node_size, edge_index, edge_affinity_logits,
                ground_truth_instances=None, return_loss=False, **kwargs):
        """自适应前向传播"""
        # 计算初始分区（作为备选）
        initial_partition = self._forward_traditional(
            batch, node_x, node_logits, stuff_classes,
            node_size, edge_index, edge_affinity_logits)

        result = self.adaptive_model(
            node_x, node_logits, edge_index, edge_affinity_logits,
            initial_partition=initial_partition,
            ground_truth_instances=ground_truth_instances,
            return_loss=return_loss
        )

        if return_loss:
            partition, loss_dict, additional_info = result
            # 应用stuff类合并
            final_partition = self._apply_stuff_merging(
                partition, node_logits, stuff_classes, batch)
            return final_partition, loss_dict, additional_info
        else:
            partition = result
            return self._apply_stuff_merging(
                partition, node_logits, stuff_classes, batch)

    def get_trainable_parameters(self):
        """获取所有可训练参数"""
        return list(self.adaptive_model.parameters())