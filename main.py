"""TrajOOD - ResNet18版本
ResNet18 + 4层连续信号(L2+余弦相似度+马氏距离) + CIFAR-10 vs OOD
一键运行: python main_resnet.py

与DeiT版本的核心区别:
1. 4个ResNet layer替代12个Transformer block
2. 各层特征维度不同(64/128/256/512)，需要独立处理协方差
3. 轨迹维度: 4层 x 3信号 = 12维 (DeiT是36维)
4. 输入尺寸224(使用ImageNet预训练权重)
"""

import torch
import random
import numpy as np
import os
import sys

# ============ 导入配置 ============
import config_resnet as cfg

# ============ 导入模块 ============
from datasets.loader import get_cifar10_loaders, get_ood_loader
from models.resnet_backbone import ResNet18Backbone
from models.resnet_hook import ResNetHook
from models.act_branch import ACTBranch
from trajectory.extractor_resnet import TrajectoryExtractor, TrajectoryStatistics
from trainers.finetune_resnet import finetune
from trainers.train_act import train_act_branch
from evaluation.scoring import compute_energy_scores, compute_traj_scores_resnet
from evaluation.metrics import compute_all_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")
    print(f"ResNet18 Trajectory OOD Detection")
    print(f"轨迹维度: {cfg.TRAJ_DIM} (4层 x 3信号)")

    # =========== Step 1: 数据加载 ===========
    print("\n[Step 1] 加载数据集...")
    train_loader, test_loader = get_cifar10_loaders(
        cfg.DATA_ROOT, cfg.BATCH_SIZE, cfg.NUM_WORKERS
    )
    print(f"  CIFAR-10 训练集: {len(train_loader.dataset)} 样本")
    print(f"  CIFAR-10 测试集: {len(test_loader.dataset)} 样本")

    # =========== Step 2: 加载ResNet18 ===========
    print("\n[Step 2] 加载ResNet18预训练模型...")
    model = ResNet18Backbone(num_classes=cfg.NUM_CLASSES, freeze_backbone=True)
    model.to(device)
    print(f"  特征维度: {model.feat_dim}")
    print(f"  各层维度: {cfg.LAYER_DIMS}")

    # =========== Step 3: 微调分类头 ===========
    print("\n[Step 3] 微调分类头...")
    ckpt_path = os.path.join(cfg.SAVE_DIR, "resnet18_cifar10.pth")
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

    # =========== Step 4: 注册Hook ===========
    print("\n[Step 4] 注册ResNet层Hook...")
    hook = ResNetHook(model, cfg.TRAJ_LAYERS)
    print(f"  Hook层: {cfg.TRAJ_LAYERS}")
    print(f"  对应维度: {[cfg.LAYER_DIMS[i] for i in range(len(cfg.TRAJ_LAYERS))]}")

    # =========== Step 5: 提取训练集轨迹 ===========
    print("\n[Step 5] 提取训练集轨迹...")

    # 5.1 提取逐层特征和L2范数
    print("  [5.1] 提取逐层特征和L2范数...")
    traj_extractor = TrajectoryExtractor(model, hook, device)
    l2_trajectories, labels, all_features = traj_extractor.extract_dataset(train_loader)
    print(f"  L2轨迹形状: {l2_trajectories.shape}")  # [N, 4]

    # 5.2 拟合类级统计量（均值+协方差逆）
    print("  [5.2] 拟合类级统计量...")
    traj_stats = TrajectoryStatistics(
        num_classes=cfg.NUM_CLASSES,
        n_layers=len(cfg.TRAJ_LAYERS),
        layer_dims=cfg.LAYER_DIMS
    )
    traj_stats.fit(
        all_features, labels, cfg.TRAJ_LAYERS,
        shrinkage=cfg.SHRINKAGE
    )
    traj_stats.save(os.path.join(cfg.SAVE_DIR, "traj_stats_resnet.pt"))

    # 5.3 构建完整12维轨迹
    print("  [5.3] 构建完整轨迹向量...")
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
    full_trajectories = torch.cat(all_traj, dim=0)  # [N, 12]
    print(f"  完整轨迹形状: {full_trajectories.shape}")
    print(f"  轨迹示例(前6维): {full_trajectories[0, :6]}")

    # 构建ACT训练用DataLoader
    traj_train_loader = traj_extractor.make_traj_loader(
        full_trajectories, labels, batch_size=256, shuffle=True
    )

    # =========== Step 6: 训练ACT-Branch ===========
    print("\n[Step 6] 训练ACT-Branch...")
    act_ckpt_path = os.path.join(cfg.SAVE_DIR, "act_branch_resnet.pth")
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

    # =========== Step 7: OOD评估 ===========
    print("\n[Step 7] OOD评估...")
    fusion_mode = cfg.FUSION_MODE
    print(f"  融合模式: {fusion_mode}")
    print("=" * 70)
    print(f"{'OOD数据集':<12} {'方法':<15} {'AUROC':>8} {'FPR@95':>8} {'AUPR':>8}")
    print("=" * 70)

    # 重新加载统计量（验证save/load一致性）
    traj_stats = TrajectoryStatistics.load(
        os.path.join(cfg.SAVE_DIR, "traj_stats_resnet.pt")
    )

    # ID测试集分数
    id_energy = compute_energy_scores(model, test_loader, device)
    id_traj = compute_traj_scores_resnet(
        act_model, model, hook, traj_stats,
        test_loader, device, cfg.TRAJ_LAYERS
    )

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
        ood_traj = compute_traj_scores_resnet(
            act_model, model, hook, traj_stats,
            ood_loader, device, cfg.TRAJ_LAYERS
        )
        m2 = compute_all_metrics(id_traj.numpy(), ood_traj.numpy())
        print(f"{'':.<12} {'Trajectory':<15} {m2['auroc']:>7.2f}% {m2['fpr95']:>7.2f}% {m2['aupr']:>7.2f}%")

        # 融合分数
        from evaluation.scoring import compute_fused_scores
        id_fused, ood_fused = compute_fused_scores(
            id_energy, id_traj, ood_energy, ood_traj,
            alpha=cfg.FUSION_LAMBDA, beta=1.0 - cfg.FUSION_LAMBDA,
            mode=fusion_mode
        )
        m3 = compute_all_metrics(id_fused.numpy(), ood_fused.numpy())
        fusion_label = f"Fusion({fusion_mode})"
        print(f"{'':.<12} {fusion_label:<15} {m3['auroc']:>7.2f}% {m3['fpr95']:>7.2f}% {m3['aupr']:>7.2f}%")
        print("-" * 70)

    # 清理hook
    hook.remove()
    print("\n实验完成。")


if __name__ == "__main__":
    main()
