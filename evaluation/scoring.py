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


# ============================================================
#  融合分数
# ============================================================
def compute_fused_scores(id_energy, id_traj, ood_energy, ood_traj,
                         alpha=0.5, beta=0.5, mode="fixed"):
    """融合Energy和Trajectory分数

    三种模式:
    - "fixed":           min-max归一化 + 固定权重加权 (原始方式)
    - "zscore_max":      用ID分布z-score统一量纲后逐样本取max
    - "zscore_weighted": 用ID分布z-score统一量纲后按ID紧致度自动加权

    z-score统一量纲原理:
      normalized = (x - id_mean) / id_std
      变换后两个信号都表示"偏离ID中心多少个标准差"
      energy z-score=-3 与 traj z-score=-3 含义相同
      只依赖ID分布统计量, 不依赖OOD数据, 实际可部署

    ID分数高, OOD分数低

    Args:
        id_energy:  tensor[N_id]  ID的energy分数
        id_traj:    tensor[N_id]  ID的轨迹分数
        ood_energy: tensor[N_ood] OOD的energy分数
        ood_traj:   tensor[N_ood] OOD的轨迹分数
        alpha: Energy权重 (仅fixed模式)
        beta:  Trajectory权重 (仅fixed模式)
        mode:  "fixed" / "zscore_max" / "zscore_weighted"

    Returns:
        id_fused:  tensor[N_id]
        ood_fused: tensor[N_ood]
    """

    if mode == "fixed":
        # ---------- 原始方式(s_min/max使用了ood数据，属于oracle方法，不可实际部署): min-max + 固定权重 ----------
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

    # ---------- z-score 统一量纲 ----------
    # 只用ID分布的均值和标准差做标准化
    # 变换后含义: "该样本偏离ID中心多少个标准差"
    # ID样本: z-score集中在0附近
    # OOD样本: z-score向负方向偏移 (因为ID分数高, OOD分数低)

    e_mean, e_std = id_energy.mean(), id_energy.std()
    t_mean, t_std = id_traj.mean(), id_traj.std()

    # 防止除零
    e_std = e_std if e_std > 1e-8 else torch.tensor(1.0)
    t_std = t_std if t_std > 1e-8 else torch.tensor(1.0)

    # z-score变换: 对ID和OOD都用ID的统计量
    id_e_z = (id_energy - e_mean) / e_std
    ood_e_z = (ood_energy - e_mean) / e_std
    id_t_z = (id_traj - t_mean) / t_std
    ood_t_z = (ood_traj - t_mean) / t_std

    if mode == "zscore_max":
        # ---------- z-score + 逐样本取max ----------
        # ID样本: 两个z-score都在0附近, max约0
        # OOD样本: 至少有一个z-score很负, 但max取较大的那个
        #          只要有一个信号认为它是ID(z高), max就高 -> 不够好
        # 改用min: 两个信号都认为是ID才得高分, 有一个认为OOD就拉低
        id_fused = torch.min(id_e_z, id_t_z)
        ood_fused = torch.min(ood_e_z, ood_t_z)

    elif mode == "zscore_weighted":
        # ---------- z-score + 按紧致度自动加权 ----------
        # 原理: ID分布标准差越小, 说明该信号对ID越"确定",
        #       OOD偏移相对更显著, 应该给更大权重
        # 权重 = 1/std (紧致度), 归一化为概率
        w_e = 1.0 / e_std
        w_t = 1.0 / t_std
        w_sum = w_e + w_t
        w_e = w_e / w_sum
        w_t = w_t / w_sum

        id_fused = w_e * id_e_z + w_t * id_t_z
        ood_fused = w_e * ood_e_z + w_t * ood_t_z
    # ============ 自适应选择 ============
    elif mode == "zscore_adaptive":
        # 哪个z-score偏离ID更远（更负），就听哪个
        # 对每个样本，选z-score更小的那个作为最终分数
        # 与zscore_max(min)不同：这里不是固定取min，而是取绝对偏离更大的
        id_fused = torch.where(id_e_z.abs() > id_t_z.abs(), id_e_z, id_t_z)
        ood_fused = torch.where(ood_e_z.abs() > ood_t_z.abs(), ood_e_z, ood_t_z)

    # ============ 乘法融合 ============
    elif mode == "zscore_multiply":
        # 带符号乘法：保留方向
        # sign = 两个z-score的平均符号方向
        # magnitude = 两个abs的乘积
        # ID: (+2)(+1.8) -> 正方向，幅度大-> 高分
        # OOD: (-3)(-2) -> 负方向（都偏离ID），幅度大 -> 低分
        # 混合: (-3)(+0.1) -> 幅度小 -> 接近0

        id_sign = torch.sign(id_e_z + id_t_z)
        id_fused = id_sign * (id_e_z.abs() * id_t_z.abs())

        ood_sign = torch.sign(ood_e_z + ood_t_z)
        ood_fused = ood_sign * (ood_e_z.abs() * ood_t_z.abs())
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
    return id_fused, ood_fused
