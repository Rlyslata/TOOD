"""全局配置 - ResNet18版本
ResNet18 + 4层连续信号(L2范数+余弦相似度+马氏距离) + CIFAR-10

放置路径: config_resnet.py (项目根目录)
"""

import torch
import os

# ============ 路径 ============
DATA_ROOT = os.path.expanduser("~/data")
SAVE_DIR = "./checkpoints_resnet"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============ 设备 ============
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ============ 随机种子 ============
SEED = 42

# ============ 数据 ============
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224  # ResNet输入尺寸 (与DeiT保持一致，便于对比)
NUM_CLASSES = 10  # CIFAR-10

# ============ ResNet微调 ============
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-3      # ResNet分类头可以用稍大学习率
FINETUNE_WEIGHT_DECAY = 1e-4

# ============ 轨迹 ============
TRAJ_LAYERS = [0, 1, 2, 3]           # ResNet18的4个layer
LAYER_DIMS = [64, 128, 256, 512]     # 各层特征维度
NUM_SIGNALS = 3                       # L2范数, 余弦相似度, 马氏距离
TRAJ_DIM = len(TRAJ_LAYERS) * NUM_SIGNALS  # 4 * 3 = 12
SHRINKAGE = 0.001  # 协方差矩阵正则化系数 (维度较小，可用更小的shrinkage)

# ============ ACT-Branch ============
ACT_EPOCHS = 100
ACT_LR = 1e-3
ACT_HIDDEN_DIM = 64      # 输入只有12维，隐藏层不需要太大
ACT_WEIGHT_DECAY = 1e-4
LOGITNORM_TAU = 0.04

# ============ 评分融合 ============
FUSION_LAMBDA = 0.7

# ============ 自适应融合 ============
# 融合模式: "fixed" / "zscore_max" / "zscore_weighted" / "zscore_adaptive" / "zscore_multiply"
FUSION_MODE = "zscore_multiply"
