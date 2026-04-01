"""
Transformer中间层Hook
提取指定层的CLS token输出
"""

import torch


class TransformerHook:
    def __init__(self, model, layer_indices):
        """
        Args:
            model: DeiTBackbone实例
            layer_indices: 要hook的层索引列表，如[2, 5, 8, 11]
        """
        self.features = {}
        self.handles = []
        self.layer_indices = layer_indices

        blocks = model.get_blocks()
        for idx in layer_indices:
            handle = blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output: [B, N, D]，N=num_tokens, D=embed_dim
            # 取CLS token: output[:, 0, :]
            self.features[layer_idx] = output[:, 0, :].detach()
        return hook_fn

    def get_features(self):
        """
        按层索引顺序返回CLS token特征
        Returns:
            dict: {layer_idx: [B, D]}
        """
        return self.features

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []