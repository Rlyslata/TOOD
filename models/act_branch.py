"""
ACT-Branch: 激活轨迹判别模型
2层MLP + LogitNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitNormLoss(nn.Module):
    """LogitNorm: 将logits归一化到超球面后计算交叉熵"""
    def __init__(self, tau=0.04):
        super().__init__()
        self.tau = tau

    def forward(self, logits, targets):
        # L2归一化logits
        norms = torch.norm(logits, p=2, dim=1, keepdim=True) + 1e-7
        logits_norm = logits / norms * (1.0 / self.tau)
        return F.cross_entropy(logits_norm, targets)


class ACTBranch(nn.Module):
    """
    轻量级全连接网络
    输入: 单点轨迹向量 [B, traj_dim]
    输出: logits [B, num_classes]
    """
    def __init__(self, traj_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def get_score(self, x):
        """
        推理时获取OOD分数
        使用负Energy score: -logsumexp(logits)
        ID样本分数应该更高（负energy更大）
        """
        logits = self.forward(x)
        energy = torch.logsumexp(logits, dim=1)
        return energy# ID高，OOD低