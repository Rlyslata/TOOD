"""
OOD评分模块
S1: Energy score (来自DeiT分类头)
S2: ACT-Branch轨迹分数
S_fused: 加权融合
"""

import torch


@torch.no_grad()
def compute_energy_scores(model, dataloader, device):
    """
    计算Energy score
    Energy = logsumexp(logits)
    ID样本energy高，OOD样本energy低
    """
    model.eval()
    scores = []
    for x, _ in dataloader:
        x = x.to(device)
        logits, _ = model(x)
        energy = torch.logsumexp(logits, dim=1)
        scores.append(energy.cpu())
    return torch.cat(scores, dim=0)



# ============================================================
# 3. compute_traj_scores (改进版)
# ============================================================
@torch.no_grad()
def compute_traj_scores(
    act_model, model, hook, traj_extractor, traj_stats,
    dataloader, device,
    weight_energy=0.4, weight_deviation=0.3, weight_aligned=0.3
):
    """
    计算综合轨迹OOD分数, 融合三个信号:
      - ACT Energy (原有)
      - 信号A: 索引偏离度 (概率版)
      - 信号B: 对齐取值的Energy

    Args:
        act_model: ACTBranch
        model: DeiTBackbone
        hook: TransformerHook
        traj_extractor: TrajectoryExtractor
        traj_stats: TrajectoryStatistics (训练阶段拟合好的)
        dataloader: 测试数据
        device: torch.device
        weight_energy: ACT Energy权重
        weight_deviation: 索引偏离信号权重
        weight_aligned: 对齐取值信号权重
    Returns:
        fused_scores: [N]  ID高, OOD低
    """
    model.eval()
    act_model.eval()

    all_energy = []
    all_deviation = []
    all_aligned_energy = []

    for x, _ in dataloader:
        x = x.to(device)
        hook.clear()
        logits, _ = model(x)
        pred_labels = logits.argmax(dim=1).cpu()  # [B]

        features_dict = hook.get_features()

        # --- 原有: ACT Energy ---
        traj_val, traj_idx = traj_extractor.extract_single_point(features_dict)
        energy_score = act_model.get_score(traj_val.to(device))  # [B], ID高OOD低
        all_energy.append(energy_score.cpu())

        # --- 信号A: 索引偏离度 (概率版) ---
        deviation = traj_stats.compute_index_probability(
            traj_idx.cpu(), pred_labels
        )  # [B], ID低OOD高
        all_deviation.append(deviation)

        # --- 信号B: 用典型索引对齐取值 ---
        typical_indices = traj_stats.get_typical_indices(pred_labels)  # [B, n_layers]
        aligned_val = traj_extractor.extract_aligned_values(
            features_dict, typical_indices.to(device)
        )  # [B, n_layers]
        # 对齐值送入ACT得到energy
        aligned_energy = act_model.get_score(aligned_val.to(device))  # [B]
        all_aligned_energy.append(aligned_energy.cpu())

    # 拼接
    all_energy = torch.cat(all_energy, dim=0)            # ID高OOD低
    all_deviation = torch.cat(all_deviation, dim=0)      # ID低OOD高
    all_aligned_energy = torch.cat(all_aligned_energy, dim=0)  # ID高OOD低

    # 统一方向: 全部转为 "ID高, OOD低"
    # deviation需要取反
    all_deviation_inv = -all_deviation  # 现在ID高OOD低

    # Min-Max归一化
    def minmax(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    e_norm = minmax(all_energy)
    d_norm = minmax(all_deviation_inv)
    a_norm = minmax(all_aligned_energy)

    # 加权融合
    fused = (
        weight_energy * e_norm +
        weight_deviation * d_norm +
        weight_aligned * a_norm
    )
     # ===== 诊断打印 =====
    print(f"    [信号诊断] energy:mean={all_energy.mean():.4f}, std={all_energy.std():.4f}, min={all_energy.min():.4f}, max={all_energy.max():.4f}")
    print(f"    [信号诊断] deviation: mean={all_deviation.mean():.4f}, std={all_deviation.std():.4f}, min={all_deviation.min():.4f}, max={all_deviation.max():.4f}")
    print(f"    [信号诊断] aligned:   mean={all_aligned_energy.mean():.4f}, std={all_aligned_energy.std():.4f}, min={all_aligned_energy.min():.4f}, max={all_aligned_energy.max():.4f}")
    # ===== 诊断结束 =====
    return fused


# ============================================================
# 4. fuse_scores (Energy + Trajectory 最终融合)
# ============================================================
def fuse_scores(energy_scores, traj_scores, lam=0.5):
    """
    Energy分支 + Trajectory分支 融合
    两者都应该是: ID高, OOD低
    """
    def minmax(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    s1 = minmax(energy_scores)
    s2 = minmax(traj_scores)
    return lam * s1 + (1 - lam) * s2


# ============================================================
# 5. 使用示例
# ============================================================
"""
# === 训练阶段 ===

# 1. 提取ID训练集轨迹
traj_val, traj_idx, labels = traj_extractor.extract_dataset(train_loader)

# 2. 拟合统计量
traj_stats = TrajectoryStatistics(num_classes=10, n_layers=len(hook.layer_indices))
traj_stats.fit(traj_idx, labels)
traj_stats.save('checkpoints/traj_stats.pt')

# 3. 用traj_val训练ACTBranch (和原来一样)
traj_loader = traj_extractor.make_traj_loader(traj_val, labels, batch_size=128)
# ... 训练ACT ...


# === 评估阶段 ===

# 1. 加载统计量
traj_stats = TrajectoryStatistics.load('checkpoints/traj_stats.pt')

# 2. 计算Trajectory综合分数
traj_scores = compute_traj_scores(
    act_model, model, hook, traj_extractor, traj_stats,
    test_loader, device,
    weight_energy=0.4, weight_deviation=0.3, weight_aligned=0.3
)

# 3. 计算Energy分数 (原有方法)
energy_scores = compute_energy_scores(model, test_loader, device)

# 4. 最终融合
final_scores = fuse_scores(energy_scores, traj_scores, lam=0.6)
# lam=0.6 偏向Energy, 因为Energy本身效果好
"""
