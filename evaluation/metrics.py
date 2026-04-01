"""
OOD检测评估指标
AUROC, FPR@95TPR, AUPR
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def compute_all_metrics(id_scores, ood_scores):
    """
    计算完整的OOD检测指标
    约定: ID分数高, OOD分数低

    Args:
        id_scores: tensor/array, ID样本的分数
        ood_scores: tensor/array, OOD样本的分数

    Returns:
        dict: {auroc, fpr95, aupr}
    """
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)

    # 标签: ID=1, OOD=0
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC
    auroc = roc_auc_score(labels, scores)

    # FPR@95TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    # AUPR
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)

    return {
        "auroc": auroc * 100,
        "fpr95": fpr95 * 100,
        "aupr": aupr * 100,
    }