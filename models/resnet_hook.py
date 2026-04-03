"""ResNet中间层Hook
提取每个layer的输出特征图，经全局平均池化后得到向量

与DeiT的区别:
- DeiT: 每层输出 [B, N_tokens, 384]，取CLS token -> [B, 384]，维度相同
- ResNet: 每层输出 [B, C, H, W]，C不同(64/128/256/512)，需GAP -> [B, C]

放置路径: models/resnet_hook.py
"""

import torch
import torch.nn.functional as F


class ResNetHook:
    def __init__(self, model, layer_indices=None):
        """
        Args:
            model: ResNetBackbone实例
            layer_indices: 要hook的层索引列表，如[0,1,2,3]对应layer1-4
                          默认None表示全部4层
        """
        self.features = {}  # {layer_idx: [B, C]}
        self.handles = []

        if layer_indices is None:
            layer_indices = [0, 1, 2, 3]
        self.layer_indices = layer_indices

        layers = model.get_layers()  # [layer1, layer2, layer3, layer4]
        for idx in layer_indices:
            handle = layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output: [B, C, H, W]
            # 全局平均池化 -> [B, C]
            pooled = F.adaptive_avg_pool2d(output, 1)  # [B, C, 1, 1]
            pooled = torch.flatten(pooled, 1)           # [B, C]
            self.features[layer_idx] = pooled.detach()
        return hook_fn

    def get_features(self):
        """
        按层索引顺序返回池化后的特征
        Returns:
            dict: {layer_idx: [B, C]}  C随层不同: 64/128/256/512
        """
        return self.features

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
