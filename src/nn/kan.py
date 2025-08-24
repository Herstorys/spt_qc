import torch
from torch import nn
from kan import KAN as PyKAN


__all__ = ['KAN', 'KANClassifier']


class KAN(nn.Module):
    """KAN (Kolmogorov-Arnold Network) 替代 MLP。
    使用 pykan 库实现，保持与 MLP 相似的接口。
    """

    def __init__(
            self,
            dims,
            grid=2,
            k=2,
            noise_scale=0.1,
            seed=0,
            device='cpu'):
        """
        :param dims: List[int] - 层维度列表
        :param grid: int - 网格大小（减少以节省内存）
        :param k: int - B-spline 的阶数（减少以节省内存）
        :param noise_scale: float - 噪声缩放
        :param seed: int - 随机种子
        :param device: str - 设备
        """
        super().__init__()
        assert len(dims) >= 2

        self.kan = PyKAN(
            width=dims,
            grid=grid,
            k=k,
            noise_scale=noise_scale,
            seed=seed,
            device=device,
        )
        self.out_dim = dims[-1]

    def forward(self, x, batch=None):
        """前向传播，保持与 MLP 相同的接口"""
        output = self.kan(x)
        return output


class KANClassifier(nn.Module):
    """基于 KAN 的分类器，替代原始的 Classifier"""

    def __init__(
            self,
            in_dim,
            num_classes,
            hidden_dims=None,
            grid=2,
            k=2,
            noise_scale=0.1,
            seed=0,
            device='cpu'):
        """
        :param in_dim: int - 输入维度
        :param num_classes: int - 类别数量
        :param hidden_dims: List[int] - 隐藏层维度，可选
        :param grid: int - 网格大小
        :param k: int - B-spline 的阶数
        :param noise_scale: float - 噪声缩放
        :param seed: int - 随机种子
        :param device: str - 设备
        """
        super().__init__()

        # 构建维度列表，限制隐藏层大小
        if hidden_dims is None:
            dims = [in_dim, num_classes]
        else:
            # 限制隐藏层维度以节省内存
            limited_hidden_dims = [min(dim, 16) for dim in hidden_dims]
            dims = [in_dim] + limited_hidden_dims + [num_classes]

        self.kan = KAN(
            dims=dims,
            grid=grid,
            k=k,
            noise_scale=noise_scale,
            seed=seed,
            device=device
        )

    def forward(self, x):
        return self.kan(x)

