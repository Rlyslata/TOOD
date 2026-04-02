"""ACT-Branch: 激活轨迹判别模型
2层MLP + LogitNorm

输入: 36维连续轨迹向量 (12层 x 3信号)
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
        logits_norm = logits / norms * (1.0 / self.tau)
        return F.cross_entropy(logits_norm, targets)


class ACTBranch(nn.Module):
    """轨迹判别网络

    结构: Linear -> ReLU -> Dropout -> Linear
    训练时用LogitNorm损失
    推理时用energy score (logsumexp)
    """
    def __init__(self, traj_dim=36, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def get_energy(self, x):
        """推理时计算energy score
        Energy = logsumexp(logits)
        ID样本energy高, OOD样本energy低
        """
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)
