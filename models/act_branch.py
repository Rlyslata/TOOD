"""ACT-Branch: 激活轨迹判别模型
2层MLP + LogitNorm

输入: 12维连续轨迹向量 (4层 x 3信号)
输出: num_classes维logits
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
        logits_norm = logits / (norms * self.tau)
        return F.cross_entropy(logits_norm, targets)


class ACTBranch(nn.Module):
    """2层MLP: traj_dim -> hidden_dim -> num_classes"""
    def __init__(self, traj_dim=12, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def get_energy(self, x):
        """计算energy score: logsumexp(logits)"""
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)
