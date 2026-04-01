"""
全局配置 - 最小可行实验
DeiT-Small + 单点轨迹 + CIFAR-10
"""

import torch
import os

# ============ 路径 ============
DATA_ROOT = os.path.expanduser("~/data")# 数据集存放路径
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============ 设备 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据 ============
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224  # DeiT输入尺寸
NUM_CLASSES = 10  # CIFAR-10

# ============ DeiT微调 ============
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
FINETUNE_WEIGHT_DECAY = 1e-4

# ============ 轨迹提取 ============
# DeiT-Small有12个Transformer block，选取哪些层提取轨迹
# 选取第2, 5, 8, 11层（0-indexed），覆盖浅、中、深层
TRAJ_LAYERS = [2, 5, 8, 11]
TRAJ_DIM = len(TRAJ_LAYERS)  # 单点轨迹维度 = 层数

# ============ ACT-Branch============
ACT_HIDDEN_DIM = 64
ACT_EPOCHS = 50
ACT_LR = 1e-3
ACT_WEIGHT_DECAY = 1e-4
LOGITNORM_TAU = 0.04# LogitNorm温度参数

# ============ 融合 ============
FUSION_LAMBDA = 0.5  # S = lambda  S1_energy + (1-lambda)  S2_traj

# ============ 随机种子 ============
SEED = 42