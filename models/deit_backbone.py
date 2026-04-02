"""DeiT-Small backbone + 可微调分类头
使用timm加载预训练权重，冻结backbone，微调分类头
"""

import torch
import torch.nn as nn
import timm


class DeiTBackbone(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        # 加载预训练DeiT-Small (384维hidden, 12层, 6头)
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0  # 去掉原始分类头
        )
        self.feat_dim = self.backbone.embed_dim  # 384

        # 分类头
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        返回:
            logits: [B, num_classes]
            features: [B, feat_dim] (CLS token)
        """
        features = self.backbone(x)  # [B, feat_dim]
        logits = self.classifier(features)  # [B, num_classes]
        return logits, features

    def get_blocks(self):
        """返回Transformer blocks列表，供hook使用"""
        return self.backbone.blocks