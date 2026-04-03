"""全局配置 - 连续轨迹方案
DeiT-Small + 12层连续信号(L2范数+余弦相似度+马氏距离) + CIFAR-10
"""

import torch
import os

# ============ 路径 ============
DATA_ROOT = os.path.expanduser("~/data")  # 数据集存放路径
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============ 设备 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 随机种子 ============
SEED = 42

# ============ 数据 ============
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224  # DeiT输入尺寸
NUM_CLASSES = 10  # CIFAR-10

# ============ DeiT微调 ============
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
FINETUNE_WEIGHT_DECAY = 1e-4

# ============ 轨迹 ============
TRAJ_LAYERS = list(range(12))  # 全部12层
FEAT_DIM = 384  # DeiT-Small hidden dim
NUM_SIGNALS = 3  # L2范数, 余弦相似度, 马氏距离
TRAJ_DIM = len(TRAJ_LAYERS) * NUM_SIGNALS  # 12 * 3 = 36
SHRINKAGE = 0.0001  # 协方差矩阵正则化系数

# ============ ACT-Branch ============
ACT_EPOCHS = 100         # 50 -> 100
ACT_LR = 1e-3
ACT_HIDDEN_DIM = 128     # 64 -> 128
ACT_WEIGHT_DECAY = 1e-4
LOGITNORM_TAU = 0.04

# ============ 评分融合 ============
FUSION_LAMBDA = 0.7  # energy与trajectory的融合权重

# ============ 自适应融合 ============
# 融合模式:
#   "fixed"           - min-max归一化 + 固定权重 (原始方式)
#   "zscore_max"      - 用ID分布z-score统一量纲后逐样本取min(保守策略)
#   "zscore_weighted" - 用ID分布z-score统一量纲后按紧致度自动加权
#   "zscore_adaptive" - 自适应选择更偏离的z-score
#   "zscore_multiply" - 带符号乘法融合
FUSION_MODE = "zscore_multiply"

