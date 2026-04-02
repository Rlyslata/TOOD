"""OOD评分模块
S1: Energy score (来自DeiT分类头)
S2: ACT-Branch轨迹分数 (基于连续信号)
S_fused: 加权融合
"""

import torch


@torch.no_grad()
def compute_energy_scores(model, dataloader, device):
    """计算Energy score
    Energy = logsumexp(logits)
    ID样本energy高, OOD样本energy低

    Args:
        model: DeiTBackbone
        dataloader: 数据加载器
        device: 设备

    Returns:
        scores: tensor[N]
    """
    model.eval()
    scores = []
    for x, _ in dataloader:
        x = x.to(device)
        logits, _ = model(x)
        energy = torch.logsumexp(logits, dim=1)
        scores.append(energy.cpu())
    return torch.cat(scores, dim=0)


@torch.no_grad()
def compute_traj_scores(act_model, model, hook, traj_stats,
                       dataloader, device, layer_indices):
    """计算轨迹分数

    流程:
      1. 前向传播得到logits和hook特征
      2. 计算L2范数
      3. 用预测类别 + 类统计量计算余弦相似度和马氏距离
      4. 拼接36维轨迹向量
      5. 送入ACT-Branch得到energy分数

    Args:
        act_model: ACTBranch
        model: DeiTBackbone
        hook: TransformerHook
        traj_stats: TrajectoryStatistics (已fit)
        dataloader: 数据加载器
        device: 设备
        layer_indices: 层索引列表

    Returns:
        scores: tensor[N]
    """
    model.eval()
    act_model.eval()
    scores = []

    for x, _ in dataloader:
        x = x.to(device)
        logits, _ = model(x)
        features_dict = hook.get_features()  # {layer_idx: [B, 384]}

        # 将特征移到CPU计算(与统计量一致)
        feat_cpu = {k: v.cpu() for k, v in features_dict.items()}

        # 预测类别
        pred_labels = logits.argmax(dim=1).cpu()  # [B]

        # 计算L2范数 [B, n_layers]
        batch_l2 = []
        for layer in layer_indices:
            feat = feat_cpu[layer]  # [B, 384]
            l2 = torch.norm(feat, p=2, dim=1)  # [B]
            batch_l2.append(l2)
        l2_norms = torch.stack(batch_l2, dim=1)  # [B, n_layers]

        # 拼接完整轨迹(L2 + 余弦 + 马氏)
        trajectory = traj_stats.build_trajectory(
            feat_cpu, pred_labels, l2_norms, layer_indices
        )  # [B, 36]

        # ACT-Branch energy
        energy = act_model.get_energy(trajectory.to(device))  # [B]
        scores.append(energy.cpu())

    return torch.cat(scores, dim=0)


def compute_fused_scores(id_energy, id_traj, ood_energy, ood_traj, alpha=0.5, beta=0.5):
    """加权融合两个分数

    使用ID+OOD合并后的统计量做min-max归一化，保证尺度一致
    ID分数高, OOD分数低

    Args:
        id_energy: tensor[N_id] ID的energy分数
        id_traj: tensor[N_id] ID的轨迹分数
        ood_energy: tensor[N_ood] OOD的energy分数
        ood_traj: tensor[N_ood] OOD的轨迹分数
        alpha: Energy权重
        beta: Trajectory权重

    Returns:
        id_fused: tensor[N_id]
        ood_fused: tensor[N_ood]
    """
    def normalize_together(id_s, ood_s):
        all_s = torch.cat([id_s, ood_s], dim=0)
        s_min = all_s.min()
        s_max = all_s.max()
        if s_max - s_min < 1e-8:
            return torch.zeros_like(id_s), torch.zeros_like(ood_s)
        id_norm = (id_s - s_min) / (s_max - s_min)
        ood_norm = (ood_s - s_min) / (s_max - s_min)
        return id_norm, ood_norm

    id_e_norm, ood_e_norm = normalize_together(id_energy, ood_energy)
    id_t_norm, ood_t_norm = normalize_together(id_traj, ood_traj)

    id_fused = alpha * id_e_norm + beta * id_t_norm
    ood_fused = alpha * ood_e_norm + beta * ood_t_norm
    return id_fused, ood_fused
