"""
单点轨迹提取器
训练阶段：
    1. 提取所有ID样本的 traj_val 和 traj_idx
    2. 按类别统计每层的典型索引（众数）→ class_typical_idx[c] shape [n_layers]
    3. 同时需要保存每层特征（用于B方案按索引取值）
评估阶段：
    对每个测试/OOD样本：
    1. 前向传播得到预测类别 c_pred、每层特征、traj_val、traj_idx
    2. 信号A：索引偏离度 — traj_idx 与 class_typical_idx[c_pred] 逐层比较，不匹配的层数越多 → 越OOD
    3. 信号B：用 class_typical_idx[c_pred] 的索引位置去每层特征取值，得到 aligned_val，值越低→ 越OOD
    4. 综合A、B得到最终Trajectory分数
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import numpy as np


# ============================================================
# 1. TrajectoryExtractor (改进版)
# ============================================================
class TrajectoryExtractor:
    def __init__(self, model, hook, device):
        self.model = model
        self.hook = hook
        self.device = device

    def extract_single_point(self, features_dict):
        """
        每层CLS token取最大值 + 最大值索引
        Args:
            features_dict: {layer_idx: [B, D]}
        Returns:
            traj_val: [B, n_layers]  最大值
            traj_idx: [B, n_layers]  最大值索引
        """
        val_points = []
        idx_points = []
        for layer_idx in sorted(features_dict.keys()):
            feat = features_dict[layer_idx]  # [B, D]
            max_result = feat.max(dim=1)
            val_points.append(max_result.values)   # [B]
            idx_points.append(max_result.indices)   # [B]
        traj_val = torch.stack(val_points, dim=1)  # [B, n_layers]
        traj_idx = torch.stack(idx_points, dim=1)  # [B, n_layers]
        return traj_val, traj_idx

    def extract_aligned_values(self, features_dict, target_indices):
        """
        信号B: 用指定索引位置去每层特征取值
        Args:
            features_dict: {layer_idx: [B, D]}
            target_indices: [B, n_layers] 每层要取的维度索引
        Returns:
            aligned_val: [B, n_layers]
        """
        aligned_points = []
        for i, layer_idx in enumerate(sorted(features_dict.keys())):
            feat = features_dict[layer_idx]  # [B, D]
            idx = target_indices[:, i]        # [B]
            # 用gather按索引取值
            val = feat.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]
            aligned_points.append(val)
        aligned_val = torch.stack(aligned_points, dim=1)  # [B, n_layers]
        return aligned_val

    @torch.no_grad()
    def extract_dataset(self, dataloader):
        """
        对整个数据集提取轨迹
        Returns:
            all_traj_val: [N, n_layers]
            all_traj_idx: [N, n_layers]
            all_labels:   [N]
        """
        self.model.eval()
        all_traj_val = []
        all_traj_idx = []
        all_labels = []

        for x, y in dataloader:
            x = x.to(self.device)
            self.hook.clear()
            logits, _ = self.model(x)

            features_dict = self.hook.get_features()
            traj_val, traj_idx = self.extract_single_point(features_dict)

            all_traj_val.append(traj_val.cpu())
            all_traj_idx.append(traj_idx.cpu())
            all_labels.append(y)

        all_traj_val = torch.cat(all_traj_val, dim=0)
        all_traj_idx = torch.cat(all_traj_idx, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_traj_val, all_traj_idx, all_labels

    def make_traj_loader(self, trajs, labels, batch_size, shuffle=True):
        dataset = TensorDataset(trajs, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 2. TrajectoryStatistics (新增)
#    训练阶段统计每个类别每层的典型索引
# ============================================================
class TrajectoryStatistics:
    def __init__(self, num_classes, n_layers):
        self.num_classes = num_classes
        self.n_layers = n_layers
        # 每个类别每层的典型索引 (众数)
        self.typical_idx = None  # [num_classes, n_layers]
        # 每个类别每层的索引分布 (用于计算概率)
        self.idx_distributions = None

    def fit(self, traj_idx, labels):
        """
        统计每个类别每层的典型索引
        Args:
            traj_idx: [N, n_layers] 所有ID样本的最大值索引
            labels:   [N] 对应标签
        """
        self.typical_idx = torch.zeros(self.num_classes, self.n_layers, dtype=torch.long)
        # 保存每类每层的索引频率分布, 用于信号A的概率打分
        self.idx_distributions = {}

        for c in range(self.num_classes):
            mask = (labels == c)
            class_idx = traj_idx[mask]  # [Nc, n_layers]

            if class_idx.shape[0] == 0:
                continue

            self.idx_distributions[c] = {}

            for layer in range(self.n_layers):
                layer_indices = class_idx[:, layer].numpy()
                counter = Counter(layer_indices)
                # 众数作为典型索引
                mode_idx = counter.most_common(1)[0][0]
                self.typical_idx[c, layer] = mode_idx
                # 保存频率分布
                total = len(layer_indices)
                self.idx_distributions[c][layer] = {
                    int(k): v / total for k, v in counter.items()
                }

        print(f"[TrajectoryStatistics] 拟合完成, {self.num_classes}类, {self.n_layers}层")
        return self

    def compute_index_deviation(self, traj_idx, pred_labels):
        """
        信号A: 计算索引偏离度
        索引与预测类别的典型索引不匹配的层数比例
        Args:
            traj_idx:    [B, n_layers]
            pred_labels: [B] 模型预测的类别
        Returns:
            deviation: [B]  范围[0,1], 越大越可能OOD
        """
        B = traj_idx.shape[0]
        typical = self.typical_idx[pred_labels]  # [B, n_layers]
        # 逐层比较: 不匹配为1, 匹配为0
        mismatch = (traj_idx != typical).float()  # [B, n_layers]
        # 不匹配层数比例
        deviation = mismatch.mean(dim=1)  # [B]
        return deviation

    def compute_index_probability(self, traj_idx, pred_labels):
        """
        信号A增强版: 基于概率的索引偏离
        用每层索引在该类别分布中的概率, 概率越低越异常
        Args:
            traj_idx:    [B, n_layers]
            pred_labels: [B]
        Returns:
            neg_log_prob: [B]  负对数概率, 越大越可能OOD
        """
        B = traj_idx.shape[0]
        log_probs = torch.zeros(B)
        smoothing = 1e-4  # 未见过的索引给一个小概率

        for i in range(B):
            c = pred_labels[i].item()
            total_log_p = 0.0
            for layer in range(self.n_layers):
                idx_val = traj_idx[i, layer].item()
                if c in self.idx_distributions and layer in self.idx_distributions[c]:
                    p = self.idx_distributions[c][layer].get(idx_val, smoothing)
                else:
                    p = smoothing
                total_log_p += np.log(p)
            log_probs[i] = total_log_p

        # 返回负对数概率: ID样本概率高 -> neg_log_prob低
        #                  OOD样本概率低 -> neg_log_prob高
        neg_log_prob = -log_probs
        return neg_log_prob

    def get_typical_indices(self, pred_labels):
        """
        获取预测类别对应的典型索引
        Args:
            pred_labels: [B]
        Returns:
            typical: [B, n_layers]
        """
        return self.typical_idx[pred_labels]

    def save(self, path):
        torch.save({
            'typical_idx': self.typical_idx,
            'idx_distributions': self.idx_distributions,
            'num_classes': self.num_classes,
            'n_layers': self.n_layers,
        }, path)
        print(f"[TrajectoryStatistics] 已保存到 {path}")

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=False)
        obj = cls(data['num_classes'], data['n_layers'])
        obj.typical_idx = data['typical_idx']
        obj.idx_distributions = data['idx_distributions']
        print(f"[TrajectoryStatistics] 已加载 {path}")
        return obj