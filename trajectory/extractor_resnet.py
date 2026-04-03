"""ResNet版连续轨迹提取器

与DeiT版的核心区别:
- DeiT: 12层, 每层维度相同(384), 共享一个feat_dim
- ResNet: 4层, 每层维度不同(64/128/256/512), 需要按层独立处理

训练阶段:
    1. 提取所有ID样本的逐层GAP特征
    2. 计算每层L2范数 -> [N, 4]
    3. 按类别统计每层均值向量和协方差矩阵(每层维度独立)
    4. 计算余弦相似度和马氏距离 -> 拼接为 [N, 12] 轨迹
评估阶段:
    对每个测试/OOD样本:
    1. 前向传播得到预测类别和逐层特征
    2. 计算L2范数 + 余弦相似度 + 马氏距离 -> [B, 12]
    3. 送入ACT-Branch得到energy分数

放置路径: trajectory/extractor_resnet.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class TrajectoryExtractorResNet:
    """提取逐层GAP特征并计算连续轨迹信号(ResNet版)"""

    def __init__(self, model, hook, device):
        """
        Args:
            model: ResNetBackbone实例
            hook: ResNetHook实例
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

        Returns:
            l2_trajectories: [N, n_layers] 每层GAP特征的L2范数
            labels: [N] 标签
            all_features: list of dict, 每个dict是 {layer_idx: [B, C_layer]}
        """
        self.model.eval()
        all_l2 = []
        all_labels = []
        all_features = []

        for x, y in dataloader:
            x = x.to(self.device)
            self.model(x)  # forward触发hook
            features_dict = self.hook.get_features()  # {layer_idx: [B, C]}

            # 计算每层L2范数
            batch_l2 = []
            batch_feat = {}
            for layer in self.layer_indices:
                feat = features_dict[layer]  # [B, C_layer]
                l2 = torch.norm(feat, p=2, dim=1)  # [B]
                batch_l2.append(l2)
                batch_feat[layer] = feat.cpu()

            batch_l2 = torch.stack(batch_l2, dim=1)  # [B, n_layers]
            all_l2.append(batch_l2.cpu())
            all_labels.append(y)
            all_features.append(batch_feat)

        l2_trajectories = torch.cat(all_l2, dim=0)  # [N, n_layers]
        labels = torch.cat(all_labels, dim=0)  # [N]

        return l2_trajectories, labels, all_features

    def make_traj_loader(self, trajectories, labels, batch_size=256, shuffle=True):
        """将轨迹数据打包为DataLoader"""
        dataset = TensorDataset(trajectories, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TrajectoryStatisticsResNet:
    """管理训练集的类级统计量 (ResNet版, 每层维度独立)

    与DeiT版的区别:
    - DeiT: 所有层共享feat_dim=384, 一个cov_inv矩阵���小相同
    - ResNet: layer0=64维, layer1=128维, layer2=256维, layer3=512维
              每层的均值向量和协方差逆矩阵大小不同
    """

    def __init__(self, num_classes, n_layers, layer_dims):
        """
        Args:
            num_classes: 类别数
            n_layers: hook的层数 (4)
            layer_dims: 各层特征维度列表 [64, 128, 256, 512]
        """
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.layer_dims = layer_dims  # 新增: 记录每层维度
        self.class_means = {}   # {class_id: {layer_idx: [C_layer]}}
        self.shared_cov_inv = {}  # {layer_idx: [C_layer, C_layer]}

    def fit(self, all_features, labels, layer_indices, shrinkage=0.001):
        """
        从训练集特征计算每类每层的均值向量和共享协方差矩阵的逆

        Args:
            all_features: list of dict, 每个dict是 {layer_idx: [B, C_layer]}
            labels: [N] 所有样本的标签
            layer_indices: 层索引列表 [0, 1, 2, 3]
            shrinkage: 协方差矩阵正则化系数
        """
        # 先按层拼接所有batch的特征
        layer_feats = {}  # {layer_idx: [N, C_layer]}
        for layer in layer_indices:
            chunks = [batch_dict[layer] for batch_dict in all_features]
            layer_feats[layer] = torch.cat(chunks, dim=0)

        # 按类计算每层均值
        for c in range(self.num_classes):
            mask = (labels == c)
            self.class_means[c] = {}
            for layer in layer_indices:
                class_feat = layer_feats[layer][mask]  # [Nc, C_layer]
                self.class_means[c][layer] = class_feat.mean(dim=0)  # [C_layer]

        # 按层计算共享协方差矩阵的逆 (每层维度不同!)
        for i, layer in enumerate(layer_indices):
            feats = layer_feats[layer]  # [N, C_layer]
            feat_dim = feats.shape[1]   # 该层的维度
            mean = feats.mean(dim=0, keepdim=True)  # [1, C_layer]
            centered = feats - mean  # [N, C_layer]
            cov = (centered.T @ centered) / (feats.shape[0] - 1)  # [C_layer, C_layer]
            # shrinkage正则化
            cov = cov + shrinkage * torch.eye(feat_dim)
            self.shared_cov_inv[layer] = torch.linalg.inv(cov)  # [C_layer, C_layer]

        dims_str = ', '.join([f'layer{i}={layer_feats[l].shape[1]}' for i, l in enumerate(layer_indices)])
        print(f"  统计量拟合完成: {self.num_classes}类, {len(layer_indices)}层, 维度[{dims_str}]")

    def compute_cosine_sim(self, features_dict, pred_labels, layer_indices):
        """
        计算每个样本与其预测类均值的余弦相似度

        Args:
            features_dict: {layer_idx: [B, C_layer]}
            pred_labels: [B] 预测标签
            layer_indices: 层索引列表

        Returns:
            cos_sim: [B, n_layers]
        """
        B = pred_labels.shape[0]
        n_layers = len(layer_indices)
        cos_sim = torch.zeros(B, n_layers)

        for i, layer in enumerate(layer_indices):
            feat = features_dict[layer]  # [B, C_layer]
            if feat.is_cuda:
                feat = feat.cpu()

            # 向量化计算: 收集每个样本对应的类均值
            means = torch.stack(
                [self.class_means[pred_labels[j].item()][layer] for j in range(B)],
                dim=0
            )  # [B, C_layer]
            cos_sim[:, i] = F.cosine_similarity(feat, means, dim=1)

        return cos_sim  # [B, n_layers]

    def compute_mahalanobis(self, features_dict, pred_labels, layer_indices):
        """
        计算每个样本到其预测类中心的马氏距离

        Args:
            features_dict: {layer_idx: [B, C_layer]}
            pred_labels: [B] 预测标签
            layer_indices: 层索引列表

        Returns:
            mahal: [B, n_layers]
        """
        B = pred_labels.shape[0]
        n_layers = len(layer_indices)
        mahal = torch.zeros(B, n_layers)

        for i, layer in enumerate(layer_indices):
            feat = features_dict[layer]  # [B, C_layer]
            if feat.is_cuda:
                feat = feat.cpu()
            cov_inv = self.shared_cov_inv[layer]  # [C_layer, C_layer]

            # 向量化计算
            means = torch.stack(
                [self.class_means[pred_labels[j].item()][layer] for j in range(B)],
                dim=0
            )  # [B, C_layer]
            diff = feat - means  # [B, C_layer]
            # 马氏距离: sqrt(diff @ cov_inv @ diff^T) 对角线
            # 等价于 sqrt(sum(diff * (diff @ cov_inv), dim=1))
            m_sq = torch.sum(diff * (diff @ cov_inv), dim=1)  # [B]
            mahal[:, i] = torch.sqrt(m_sq.clamp(min=1e-8))

        return mahal  # [B, n_layers]

    def build_trajectory(self, features_dict, pred_labels, l2_norms, layer_indices):
        """
        拼接完整的12维轨迹向量: [L2, cos_sim, mahalanobis] x 4层

        Args:
            features_dict: {layer_idx: [B, C_layer]}
            pred_labels: [B] 预测标签
            l2_norms: [B, n_layers] L2范数
            layer_indices: 层索引列表

        Returns:
            trajectory: [B, n_layers * 3]
        """
        cos_sim = self.compute_cosine_sim(features_dict, pred_labels, layer_indices)
        mahal = self.compute_mahalanobis(features_dict, pred_labels, layer_indices)

        # 拼接: [B, 4] x 3 -> [B, 12]
        trajectory = torch.cat([l2_norms, cos_sim, mahal], dim=1)
        return trajectory

    def save(self, path):
        """保存统计量到文件"""
        state = {
            'num_classes': self.num_classes,
            'n_layers': self.n_layers,
            'layer_dims': self.layer_dims,
            'class_means': self.class_means,
            'shared_cov_inv': self.shared_cov_inv,
        }
        torch.save(state, path)
        print(f"  统计量已保存: {path}")

    @classmethod
    def load(cls, path):
        """从文件加载统计量"""
        state = torch.load(path, map_location='cpu')
        obj = cls(state['num_classes'], state['n_layers'], state['layer_dims'])
        obj.class_means = state['class_means']
        obj.shared_cov_inv = state['shared_cov_inv']
        print(f"  统计量已加载: {path}")
        return obj
