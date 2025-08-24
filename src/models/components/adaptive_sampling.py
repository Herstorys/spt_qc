import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn_graph
from torch_cluster import radius_graph
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, List, Dict, Tuple
import numpy as np


class AdaptiveSampler(nn.Module):
    """
    Adaptive Point Sampler with density awareness
    """

    def __init__(self,
                 sample_ratio: float = 0.5,
                 radius: float = 0.1,
                 max_neighbors: int = 32,
                 density_aware: bool = True):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.density_aware = density_aware

        # Learnable parameters for density weighting
        if density_aware:
            self.density_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def compute_density(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute local point density for each point

        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]

        Returns:
            density: Local density for each point [N]
        """
        # Build radius graph
        edge_index = radius_graph(pos, r=self.radius, batch=batch,
                                  max_num_neighbors=self.max_neighbors)

        # Count neighbors for each point
        density = torch.zeros(pos.size(0), device=pos.device, dtype=torch.float)
        density.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))

        return density

    def compute_geometric_features(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric features like curvature, normal variation

        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]

        Returns:
            geo_features: Geometric features [N, F]
        """
        # Build k-NN graph for geometric analysis
        edge_index = knn_graph(pos, k=16, batch=batch)
        row, col = edge_index

        # Compute relative positions
        rel_pos = pos[col] - pos[row]  # [E, 3]

        # Compute distances
        distances = torch.norm(rel_pos, p=2, dim=-1)  # [E]

        # Estimate local surface normal (simplified PCA approach)
        # Group by source nodes
        normals = torch.zeros_like(pos)  # [N, 3]

        # For each point, compute covariance of its neighbors
        for i in range(pos.size(0)):
            neighbors_mask = (row == i)
            if neighbors_mask.sum() > 3:  # Need at least 3 neighbors
                neighbor_pos = pos[col[neighbors_mask]]  # [K, 3]
                centroid = neighbor_pos.mean(dim=0)  # [3]
                centered = neighbor_pos - centroid  # [K, 3]

                # Compute covariance matrix
                cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1)

                # Eigendecomposition (simplified - use smallest eigenvector as normal)
                try:
                    eigenvals, eigenvecs = torch.linalg.eigh(cov)
                    normals[i] = eigenvecs[:, 0]  # Smallest eigenvector
                except:
                    normals[i] = torch.tensor([0, 0, 1], device=pos.device, dtype=pos.dtype)

        # Compute curvature estimate (variation of normals)
        curvature = torch.zeros(pos.size(0), device=pos.device)

        for i in range(pos.size(0)):
            neighbors_mask = (row == i)
            if neighbors_mask.sum() > 1:
                neighbor_normals = normals[col[neighbors_mask]]
                current_normal = normals[i].unsqueeze(0)

                # Compute angular variation
                dot_products = torch.mm(current_normal, neighbor_normals.t()).squeeze()
                dot_products = torch.clamp(dot_products, -1, 1)
                angles = torch.acos(torch.abs(dot_products))
                curvature[i] = angles.std()

        return curvature.unsqueeze(-1)  # [N, 1]

    def adaptive_fps(self, pos: torch.Tensor, batch: torch.Tensor,
                     features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform density-aware Farthest Point Sampling

        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]
            features: Point features [N, F] (optional)

        Returns:
            idx: Indices of sampled points
        """
        if not self.density_aware:
            # Standard FPS
            return fps(pos, batch, ratio=self.sample_ratio)

        # Compute density and geometric features
        density = self.compute_density(pos, batch)
        geo_features = self.compute_geometric_features(pos, batch)

        # Normalize density
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)

        # Compute importance weights (favor low density + high curvature regions)
        density_weight = self.density_mlp(density_norm.unsqueeze(-1)).squeeze(-1)
        curvature_weight = geo_features.squeeze(-1)
        curvature_weight = (curvature_weight - curvature_weight.min()) / (
                    curvature_weight.max() - curvature_weight.min() + 1e-8)

        # Combine weights (favor low density, high curvature)
        importance = (1.0 - density_weight) * 0.6 + curvature_weight * 0.4

        # Weighted FPS (approximate by modifying positions slightly)
        # Add small random perturbation weighted by importance
        pos_weighted = pos + torch.randn_like(pos) * 0.01 * importance.unsqueeze(-1)

        # Perform FPS on weighted positions
        idx = fps(pos_weighted, batch, ratio=self.sample_ratio)

        return idx

    def forward(self, pos: torch.Tensor, batch: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of adaptive sampler

        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]
            features: Point features [N, F] (optional)

        Returns:
            sampled_pos: Sampled point positions
            sampled_batch: Sampled batch indices
            sampled_features: Sampled point features (if provided)
            idx: Indices of sampled points
        """
        idx = self.adaptive_fps(pos, batch, features)

        sampled_pos = pos[idx]
        sampled_batch = batch[idx]
        sampled_features = features[idx] if features is not None else None

        return sampled_pos, sampled_batch, sampled_features, idx


class HierarchicalSampler(nn.Module):
    """
    Hierarchical multi-scale adaptive sampler
    """

    def __init__(self,
                 sample_ratios: List[float] = [0.8, 0.4, 0.2],
                 radii: List[float] = [0.1, 0.2, 0.4],
                 max_neighbors: int = 32):
        super().__init__()

        assert len(sample_ratios) == len(radii), "sample_ratios and radii must have same length"

        self.sample_ratios = sample_ratios
        self.radii = radii
        self.num_levels = len(sample_ratios)

        # Create adaptive samplers for each level
        self.samplers = nn.ModuleList([
            AdaptiveSampler(ratio, radius, max_neighbors, density_aware=True)
            for ratio, radius in zip(sample_ratios, radii)
        ])

        # Feature propagation networks for each level
        self.feature_propagations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 128),  # Adjust dimensions as needed
                nn.ReLU(),
                nn.Linear(128, 64)
            ) for _ in range(self.num_levels - 1)
        ])

    def interpolate_features(self, pos_source: torch.Tensor, pos_target: torch.Tensor,
                             features_source: torch.Tensor, batch_source: torch.Tensor,
                             batch_target: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Interpolate features from source points to target points using k-NN

        Args:
            pos_source: Source point positions [N_s, 3]
            pos_target: Target point positions [N_t, 3]
            features_source: Source point features [N_s, F]
            batch_source: Source batch indices [N_s]
            batch_target: Target batch indices [N_t]
            k: Number of nearest neighbors for interpolation

        Returns:
            interpolated_features: Features interpolated to target points [N_t, F]
        """
        # Build k-NN graph from target to source
        edge_index = knn_graph(pos_source, k=k, batch=batch_source,
                               flow='target_to_source')

        # Compute distances for weighting
        row, col = edge_index
        distances = torch.norm(pos_target[row] - pos_source[col], p=2, dim=-1)

        # Compute weights (inverse distance weighting)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / scatter_add(weights, row, dim=0)[row]

        # Interpolate features
        weighted_features = features_source[col] * weights.unsqueeze(-1)
        interpolated_features = scatter_add(weighted_features, row, dim=0,
                                            dim_size=pos_target.size(0))

        return interpolated_features

    def forward(self, pos: torch.Tensor, batch: torch.Tensor,
                features: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass of hierarchical sampler

        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]
            features: Point features [N, F]

        Returns:
            hierarchical_data: List of dictionaries containing data for each level
        """
        hierarchical_data = []
        current_pos, current_batch, current_features = pos, batch, features

        for level, sampler in enumerate(self.samplers):
            # Sample points at current level
            sampled_pos, sampled_batch, sampled_features, idx = sampler(
                current_pos, current_batch, current_features
            )

            # Store data for current level
            level_data = {
                'pos': sampled_pos,
                'batch': sampled_batch,
                'features': sampled_features,
                'idx': idx,
                'level': level,
                'sample_ratio': self.sample_ratios[level]
            }
            hierarchical_data.append(level_data)

            # Update for next level
            current_pos = sampled_pos
            current_batch = sampled_batch
            current_features = sampled_features

        return hierarchical_data


class AdaptiveSamplingLoss(nn.Module):
    """
    Additional loss to encourage better sampling
    """

    def __init__(self, coverage_weight: float = 0.1, diversity_weight: float = 0.1):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.diversity_weight = diversity_weight

    def compute_coverage_loss(self, original_pos: torch.Tensor, sampled_pos: torch.Tensor,
                              batch_original: torch.Tensor, batch_sampled: torch.Tensor) -> torch.Tensor:
        """
        Compute coverage loss - ensure sampled points cover the original space well
        """
        # For each original point, find distance to nearest sampled point
        edge_index = knn_graph(sampled_pos, k=1, batch=batch_sampled)  # Find nearest sampled point

        # This is a simplified version - in practice, you'd want more sophisticated coverage metrics
        distances = torch.norm(original_pos.unsqueeze(1) - sampled_pos.unsqueeze(0), p=2, dim=-1)
        min_distances, _ = distances.min(dim=1)

        # Coverage loss is the mean distance to nearest sampled point
        coverage_loss = min_distances.mean()

        return coverage_loss

    def compute_diversity_loss(self, sampled_pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss - encourage sampled points to be well distributed
        """
        # Compute pairwise distances between sampled points
        distances = torch.cdist(sampled_pos, sampled_pos, p=2)

        # Remove diagonal (self-distances)
        mask = torch.eye(distances.size(0), device=distances.device).bool()
        distances = distances.masked_fill(mask, float('inf'))

        # Diversity loss encourages larger minimum distances
        min_distances, _ = distances.min(dim=1)
        diversity_loss = -min_distances.mean()  # Negative to encourage larger distances

        return diversity_loss

    def forward(self, original_pos: torch.Tensor, sampled_pos: torch.Tensor,
                batch_original: torch.Tensor, batch_sampled: torch.Tensor) -> torch.Tensor:
        """
        Compute total adaptive sampling loss
        """
        coverage_loss = self.compute_coverage_loss(original_pos, sampled_pos,
                                                   batch_original, batch_sampled)
        diversity_loss = self.compute_diversity_loss(sampled_pos, batch_sampled)

        total_loss = (self.coverage_weight * coverage_loss +
                      self.diversity_weight * diversity_loss)

        return total_loss



def test_adaptive_sampler():
    pos = torch.rand(100, 3)  # 100个随机点
    batch = torch.zeros(100, dtype=torch.long)  # 单一批次
    features = torch.rand(100, 16)  # 每个点16维特征

    sampler = AdaptiveSampler(sample_ratio=0.5, radius=0.1, density_aware=True)
    sampled_pos, sampled_batch, sampled_features, idx = sampler(pos, batch, features)

    assert sampled_pos.shape[0] == 50, "采样点数量不正确"
    assert sampled_pos.shape[1] == 3, "采样点位置维度不正确"
    assert sampled_features.shape[0] == 50, "采样特征数量不正确"
    assert idx.shape[0] == 50, "采样索引数量不正确"

if __name__ == '__main__':
    test_adaptive_sampler()
    print("AdaptiveSampler 测试通过")