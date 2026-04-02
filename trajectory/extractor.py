"""连续轨迹提取器
训练阶段：
    1. 提取所有ID样本的逐层CLS特征
    2. 计算每层L2范数 -> [N, 12]
    3. 按类别统计每层均值向量和共享协方差矩阵
    4. 计算余弦相似度和马氏距离 -> 拼接为 [N, 36] 轨迹
评估阶段：
    对每个测试/OOD样本：
    1. 前向传播得到预测类别和逐层特征
    2. 计算L2范数 + 余弦相似度 + 马氏距离 -> [B, 36]
    3. 送入ACT-Branch得到energy分数
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class TrajectoryExtractor:
    """提取逐层CLS token特征并计算连续轨迹信号"""

    def __init__(self, model, hook, device):
        """
        Args:
            model: DeiTBackbone实例
            hook: TransformerHook实例
            device: torch.device
        """
        self.model = model
        self.hook = hook
        self.device = device
        self.layer_indices = hook.layer_indices

    @torch.no_grad()
    def extract_dataset(self, dataloader):
        """
        提取整个数据集的逐层特征和L2范数轨迹

        Args:
            dataloader: DataLoader

        Returns:
            l2_trajectories: [N, n_layers] 每层CLS的L2范数
            labels: [N] 标签
            all_features: list of dict, 每个dict是 {layer_idx: [B, feat_dim]}
                          用于后续拟合统计量
        """
        self.model.eval()
        all_l2 = []
        all_labels = []
        all_features = []  # 存储每个batch的features_dict（CPU）

        for x, y in dataloader:
            x = x.to(self.device)
            self.model(x)  # forward触发hook
            features_dict = self.hook.get_features()  # {layer_idx: [B, 384]}

            # 计算每层L2范数
            batch_l2 = []
            batch_feat = {}
            for layer in self.layer_indices:
                feat = features_dict[layer]  # [B, 384]
                l2 = torch.norm(feat, p=2, dim=1)  # [B]
                batch_l2.append(l2)
                batch_feat[layer] = feat.cpu()  # 转CPU节省显存

            batch_l2 = torch.stack(batch_l2, dim=1)  # [B, n_layers]
            all_l2.append(batch_l2.cpu())
            all_labels.append(y)
            all_features.append(batch_feat)

        l2_trajectories = torch.cat(all_l2, dim=0)  # [N, n_layers]
        labels = torch.cat(all_labels, dim=0)  # [N]

        return l2_trajectories, labels, all_features

    def make_traj_loader(self, trajectories, labels, batch_size=256, shuffle=True):
        """
        将轨迹数据打包为DataLoader

        Args:
            trajectories: [N, traj_dim] 完整轨迹向量
            labels: [N] 标签
            batch_size: batch大小
            shuffle: 是否打乱

        Returns:
            DataLoader
        """
        dataset = TensorDataset(trajectories, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TrajectoryStatistics:
    """管理训练集的类级统计量（均值、协方差逆），用于计算余弦相似度和马氏距离"""

    def __init__(self, num_classes, n_layers):
        """
        Args:
            num_classes: 类别数
            n_layers: hook的层数
        """
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.class_means = {}   # {class_id: {layer_idx: [feat_dim]}}
        self.shared_cov_inv = {}  # {layer_idx: [feat_dim, feat_dim]}

    def fit(self, all_features, labels, layer_indices, shrinkage=0.1):
        """
        从训练集特征计算每类每层的均值向量和共享协方差矩阵的逆

        Args:
            all_features: list of dict, 每个dict是 {layer_idx: [B, feat_dim]}（来自extract_dataset）
            labels: [N] 所有样本的标签
            layer_indices: 层索引列表
            shrinkage: 协方差矩阵正则化系数
        """
        # 先按层拼接所有batch的特征
        layer_feats = {}  # {layer_idx: [N, feat_dim]}
        for layer in layer_indices:
            chunks = [batch_dict[layer] for batch_dict in all_features]
            layer_feats[layer] = torch.cat(chunks, dim=0)  # [N, feat_dim]

        feat_dim = layer_feats[layer_indices[0]].shape[1]

        # 按类计算每层均值
        for c in range(self.num_classes):
            mask = (labels == c)
            self.class_means[c] = {}
            for layer in layer_indices:
                class_feat = layer_feats[layer][mask]  # [Nc, feat_dim]
                self.class_means[c][layer] = class_feat.mean(dim=0)  # [feat_dim]

        # 按层计算共享协方差矩阵的逆（所有类共享）
        for layer in layer_indices:
            feats = layer_feats[layer]  # [N, feat_dim]
            mean = feats.mean(dim=0, keepdim=True)  # [1, feat_dim]
            centered = feats - mean  # [N, feat_dim]
            cov = (centered.T @ centered) / (feats.shape[0] - 1)  # [feat_dim, feat_dim]
            # shrinkage正则化防止奇异
            cov = cov + shrinkage * torch.eye(feat_dim)
            self.shared_cov_inv[layer] = torch.linalg.inv(cov)  # [feat_dim, feat_dim]

        print(f"  统计量拟合完成: {self.num_classes}类, {len(layer_indices)}层, 特征维度{feat_dim}")

    def compute_cosine_sim(self, features_dict, pred_labels, layer_indices):
        """
        计算每个样本与其预测类均值的余弦相似度

        Args:
            features_dict: {layer_idx: [B, feat_dim]}
            pred_labels: [B] 预测标签
            layer_indices: 层索引列表

        Returns:
            cos_sim: [B, n_layers]
        """
        B = pred_labels.shape[0]
        n_layers = len(layer_indices)
        cos_sim = torch.zeros(B, n_layers)

        for i, layer in enumerate(layer_indices):
            feat = features_dict[layer]  # [B, feat_dim]
            if feat.is_cuda:
                feat = feat.cpu()
            for j in range(B):
                c = pred_labels[j].item()
                mean = self.class_means[c][layer]  # [feat_dim]
                sim = torch.nn.functional.cosine_similarity(
                    feat[j].unsqueeze(0), mean.unsqueeze(0)
                )  # [1]
                cos_sim[j, i] = sim.item()

        return cos_sim  # [B, n_layers]

    def compute_mahalanobis(self, features_dict, pred_labels, layer_indices):
        """
        计算每个样本到其预测类中心的马氏距离

        Args:
            features_dict: {layer_idx: [B, feat_dim]}
            pred_labels: [B] 预测标签
            layer_indices: 层索引列表

        Returns:
            mahal: [B, n_layers]
        """
        B = pred_labels.shape[0]
        n_layers = len(layer_indices)
        mahal = torch.zeros(B, n_layers)

        for i, layer in enumerate(layer_indices):
            feat = features_dict[layer]  # [B, feat_dim]
            if feat.is_cuda:
                feat = feat.cpu()
            cov_inv = self.shared_cov_inv[layer]  # [feat_dim, feat_dim]

            for j in range(B):
                c = pred_labels[j].item()
                mean = self.class_means[c][layer]  # [feat_dim]
                diff = (feat[j] - mean).unsqueeze(0)  # [1, feat_dim]
                # 马氏距离: sqrt(diff @ cov_inv @ diff^T)
                m = torch.sqrt(diff @ cov_inv @ diff.T + 1e-8)  # [1, 1]
                mahal[j, i] = m.item()

        return mahal  # [B, n_layers]

    def build_trajectory(self, features_dict, pred_labels, l2_norms, layer_indices):
        """
        拼接完整的36维轨迹向量: [L2, cos_sim, mahalanobis] x 12层

        Args:
            features_dict: {layer_idx: [B, feat_dim]}
            pred_labels: [B] 预测标签
            l2_norms: [B, n_layers] L2范数
            layer_indices: 层索引列表

        Returns:
            trajectory: [B, n_layers * 3]
        """
        cos_sim = self.compute_cosine_sim(features_dict, pred_labels, layer_indices)
        mahal = self.compute_mahalanobis(features_dict, pred_labels, layer_indices)

        # 拼接: [B, n_layers] x 3 -> [B, n_layers * 3]
        trajectory = torch.cat([l2_norms, cos_sim, mahal], dim=1)
        return trajectory

    def save(self, path):
        """保存统计量到文件"""
        state = {
            'num_classes': self.num_classes,
            'n_layers': self.n_layers,
            'class_means': self.class_means,
            'shared_cov_inv': self.shared_cov_inv,
        }
        torch.save(state, path)
        print(f"  统计量已保存: {path}")

    @classmethod
    def load(cls, path):
        """从文件加载统计量"""
        state = torch.load(path, map_location='cpu')
        obj = cls(state['num_classes'], state['n_layers'])
        obj.class_means = state['class_means']
        obj.shared_cov_inv = state['shared_cov_inv']
        print(f"  统计量已加载: {path}")
        return obj
