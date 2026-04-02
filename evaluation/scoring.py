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
def compute_traj_scores(act_model, model, hook, traj_extractor,
                        traj_stats, dataloader, device):
    """计算轨迹分数

    流程:
      1. 前向传播得到logits和hook特征
      2. 提取L2范数
      3. 用预测类别 + 类统计量计算余弦相似度和马氏距离
      4. 拼接36维轨迹向量
      5. 送入ACT-Branch得到energy分数

    Args:
        act_model: ACTBranch
        model: DeiTBackbone
        hook: TransformerHook
        traj_extractor: TrajectoryExtractor
        traj_stats: TrajectoryStatistics (已fit)
        dataloader: 数据加载器
        device: 设备

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

        # 提取L2范数
        l2_norms = traj_extractor.extract_l2(feat_cpu)  # [B, 12]

        # 拼接完整轨迹(L2 + 余弦 + 马氏)
        trajectory = traj_stats.build_trajectory(
            feat_cpu, pred_labels, l2_norms
        )  # [B, 36]

        # ACT-Branch energy
        energy = act_model.get_energy(trajectory.to(device))  # [B]
        scores.append(energy.cpu())

    return torch.cat(scores, dim=0)


def compute_fused_scores(energy_scores, traj_scores, alpha=0.5, beta=0.5):
    """加权融合两个分数

    先对两组分数做min-max归一化, 再加权求和
    ID分数高, OOD分数低

    Args:
        energy_scores: tensor[N]
        traj_scores: tensor[N]
        alpha: Energy权重
        beta: Trajectory权重

    Returns:
        fused: tensor[N]
    """
    def normalize(s):
        s_min = s.min()
        s_max = s.max()
        if s_max - s_min < 1e-8:
            return torch.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    e_norm = normalize(energy_scores)
    t_norm = normalize(traj_scores)
    return alpha * e_norm + beta * t_norm
