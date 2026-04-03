"""ResNet18 backbone + 可微调分类头
使用torchvision预训练权重，冻结backbone，微调分类头
放置路径: models/resnet_backbone.py
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        # 加载预训练ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)

        # 拆分为各个stage，方便hook
        self.conv1 = resnet.conv1        # 3 -> 64, /2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool     # /2
        self.layer1 = resnet.layer1       # 64  -> 64,   /1
        self.layer2 = resnet.layer2       # 64  -> 128,  /2
        self.layer3 = resnet.layer3       # 128 -> 256,  /2
        self.layer4 = resnet.layer4       # 256 -> 512,  /2
        self.avgpool = resnet.avgpool     # adaptive avg pool -> [B, 512, 1, 1]

        self.feat_dim = 512  # ResNet18最终特征维度

        # 各层特征维度（hook用）
        self.layer_dims = [64, 128, 256, 512]

        # 分类头
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # 冻结backbone
        if freeze_backbone:
            for name, param in self.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

    def get_layers(self):
        """返回4个layer模块列表，供hook使用"""
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        """
        返回:
            logits: [B, num_classes]
            features: [B, 512] (全局平均池化后的特征)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)   # [B, 64, 56, 56]
        x = self.layer2(x)   # [B, 128, 28, 28]
        x = self.layer3(x)   # [B, 256, 14, 14]
        x = self.layer4(x)   # [B, 512, 7, 7]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        features = torch.flatten(x, 1)  # [B, 512]
        logits = self.classifier(features)
        return logits, features
