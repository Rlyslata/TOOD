"""TrajOOD 连续轨迹方案
DeiT-Small + 12层连续信号(L2+余弦相似度+马氏距离) + CIFAR-10 vs OOD
一键运行: python main.py
"""

import torch
import random
import numpy as np
import os

import config as cfg
from datasets.loader import get_cifar10_loaders, get_ood_loader
from models.deit_backbone import DeiTBackbone
from models.hook import TransformerHook
from models.act_branch import ACTBranch
from trajectory.extractor import TrajectoryExtractor, TrajectoryStatistics
from trainers.finetune import finetune
from trainers.train_act import train_act_branch
from evaluation.scoring import compute_energy_scores, compute_traj_scores, compute_fused_scores
from evaluation.metrics import compute_all_metrics


def set_seed(seed):
    # 1. Python 内置随机数生成器
    random.seed(seed)

    # 2. NumPy 随机数生成器
    np.random.seed(seed)

    # 3. PyTorch CPU 随机数生成器
    torch.manual_seed(seed)

    # 4. PyTorch GPU 随机数生成器（所有GPU）
    torch.cuda.manual_seed_all(seed)

    # 5. cuDNN 确定性模式（保证卷积结果可重复）
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")

    # ============ Step 1: 数据加载 ============
    print("\n[Step 1] 加载数据集...")
    train_loader, test_loader = get_cifar10_loaders(
        cfg.DATA_ROOT, cfg.BATCH_SIZE, cfg.NUM_WORKERS
    )
    print(f"  CIFAR-10 训练集: {len(train_loader.dataset)} 样本")
    print(f"  CIFAR-10 测试集: {len(test_loader.dataset)} 样本")

    # ============ Step 2: 加载DeiT-Small ============
    print("\n[Step 2] 加载DeiT-Small预训练模型...")
    model = DeiTBackbone(num_classes=cfg.NUM_CLASSES, freeze_backbone=True)
    model.to(device)
    print(f"  特征维度: {model.feat_dim}")

    # ============ Step 3: 微调分类头 ============
    print("\n[Step 3] 微调分类头...")
    ckpt_path = os.path.join(cfg.SAVE_DIR, "deit_cifar10.pth")
    if os.path.exists(ckpt_path):
        print(f"  加载已有checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        model = finetune(
            model, train_loader, device,
            epochs=cfg.FINETUNE_EPOCHS,
            lr=cfg.FINETUNE_LR,
            weight_decay=cfg.FINETUNE_WEIGHT_DECAY
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"  模型已保存: {ckpt_path}")

    # 测试分类准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"  CIFAR-10 测试准确率: {correct/total * 100:.2f}%")

    # ============ Step 4: 注册Hook ============
    print("\n[Step 4] 注册Transformer层Hook...")
    hook = TransformerHook(model, cfg.TRAJ_LAYERS)
    print(f"  Hook层: {cfg.TRAJ_LAYERS}")

    # ============ Step 5: 提取训练集轨迹 ============
    print("\n[Step 5] 提取训练集轨迹...")

    # 5.1 提取逐层特征和L2范数
    print("  [5.1] 提取逐层CLS特征和L2范数...")
    traj_extractor = TrajectoryExtractor(model, hook, device)
    l2_trajectories, labels, all_features = traj_extractor.extract_dataset(train_loader)
    print(f"  L2轨迹形状: {l2_trajectories.shape}")  # [N, 12]

    # 5.2 拟合类级统计量（均值+协方差逆）
    print("  [5.2] 拟合类级统计量...")
    traj_stats = TrajectoryStatistics(
        num_classes=cfg.NUM_CLASSES,
        n_layers=len(cfg.TRAJ_LAYERS)
    )
    traj_stats.fit(
        all_features, labels, cfg.TRAJ_LAYERS,
        shrinkage=cfg.SHRINKAGE
    )
    traj_stats.save(os.path.join(cfg.SAVE_DIR, "traj_stats.pt"))

    # 5.3 构建完整36维轨迹（需要预测标签）
    print("  [5.3] 构建完整轨迹向量...")
    # 训练集用真实标签作为pred_labels
    # 需要逐batch重建features_dict来计算余弦和马氏
    all_traj = []
    offset = 0
    for batch_feat in all_features:
        B = batch_feat[cfg.TRAJ_LAYERS[0]].shape[0]
        batch_labels = labels[offset:offset + B]
        batch_l2 = l2_trajectories[offset:offset + B]
        traj = traj_stats.build_trajectory(
            batch_feat, batch_labels, batch_l2, cfg.TRAJ_LAYERS
        )
        all_traj.append(traj)
        offset += B
    full_trajectories = torch.cat(all_traj, dim=0)  # [N, 36]
    print(f"  完整轨迹形状: {full_trajectories.shape}")
    print(f"  轨迹示例(前3维): {full_trajectories[0, :3]}")

    # 构建ACT训练用DataLoader
    traj_train_loader = traj_extractor.make_traj_loader(
        full_trajectories, labels, batch_size=256, shuffle=True
    )

    # ============ Step 6: 训练ACT-Branch ============
    print("\n[Step 6] 训练ACT-Branch...")
    act_ckpt_path = os.path.join(cfg.SAVE_DIR, "act_branch.pth")
    act_model = ACTBranch(
        traj_dim=cfg.TRAJ_DIM,
        hidden_dim=cfg.ACT_HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES
    )

    if os.path.exists(act_ckpt_path):
        print(f"  加载已有checkpoint: {act_ckpt_path}")
        act_model.load_state_dict(torch.load(act_ckpt_path, map_location=device))
        act_model.to(device)
    else:
        act_model = train_act_branch(
            act_model, traj_train_loader, device,
            epochs=cfg.ACT_EPOCHS,
            lr=cfg.ACT_LR,
            weight_decay=cfg.ACT_WEIGHT_DECAY,
            tau=cfg.LOGITNORM_TAU
        )
        torch.save(act_model.state_dict(), act_ckpt_path)
        print(f"  模型已保存: {act_ckpt_path}")

    # ============ Step 7: OOD评估 ============
    print("\n[Step 7] OOD评估...")
    print("=" * 70)
    print(f"{'OOD数据集':<12} {'方法':<15} {'AUROC':>8} {'FPR@95':>8} {'AUPR':>8}")
    print("=" * 70)

    # 重新加载统计量（验证save/load一致性）
    traj_stats = TrajectoryStatistics.load(
        os.path.join(cfg.SAVE_DIR, "traj_stats.pt")
    )

    # ID测试集分数
    id_energy = compute_energy_scores(model, test_loader, device)
    id_traj = compute_traj_scores(
        act_model, model, hook, traj_extractor, traj_stats,
        test_loader, device
    )
    id_fused = compute_fused_scores(id_energy, id_traj, cfg.FUSION_LAMBDA)

    for ood_name in ["svhn", "textures", "lsun"]:
        ood_loader = get_ood_loader(
            ood_name, cfg.DATA_ROOT, cfg.BATCH_SIZE, cfg.NUM_WORKERS
        )
        if ood_loader is None:
            continue

        # Energy score
        ood_energy = compute_energy_scores(model, ood_loader, device)
        m1 = compute_all_metrics(id_energy.numpy(), ood_energy.numpy())
        print(f"{ood_name:<12} {'Energy':<15} {m1['auroc']:>7.2f}% {m1['fpr95']:>7.2f}% {m1['aupr']:>7.2f}%")

        # 轨迹分数
        ood_traj = compute_traj_scores(
            act_model, model, hook, traj_extractor, traj_stats,
            ood_loader, device
        )
        m2 = compute_all_metrics(id_traj.numpy(), ood_traj.numpy())
        print(f"{'':<12} {'Trajectory':<15} {m2['auroc']:>7.2f}% {m2['fpr95']:>7.2f}% {m2['aupr']:>7.2f}%")

        # 融合分数
        ood_fused = compute_fused_scores(ood_energy, ood_traj, cfg.FUSION_LAMBDA)
        m3 = compute_all_metrics(id_fused.numpy(), ood_fused.numpy())
        print(f"{'':<12} {'Fusion':<15} {m3['auroc']:>7.2f}% {m3['fpr95']:>7.2f}% {m3['aupr']:>7.2f}%")
        print("-" * 70)

    # 清理hook
    hook.remove()
    print("\n实验完成。")


if __name__ == "__main__":
    main()
