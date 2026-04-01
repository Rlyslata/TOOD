# tests/test_config.py
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

# 添加项目根目录到路径



class TestLoader(unittest.TestCase):
    stats = torch.load('checkpoints/traj_stats.pt')

    print("=== num_classes:", stats['num_classes'])
    print("=== n_layers:", stats['n_layers'])
    print("=== typical_idx 形状:", stats['typical_idx'].shape)
    print("=== typical_idx 前3类:\n", stats['typical_idx'][:3])

    # 探结构
    d = stats['idx_distributions']
    print("\n=== 顶层类型:", type(d))

    # 取类0
    c0 = d[0]
    
    print("=== d[0] 类型:", type(c0))
    if isinstance(c0, dict):
        first_key = list(c0.keys())[0]
        v = c0[first_key]
        print(f"=== d[0][{first_key}] 类型: {type(v)}")
        
        if isinstance(v, dict):
            first_sub_key = list(v.keys())[0]
            vv = v[first_sub_key]
            print(f"=== d[0][{first_key}][{first_sub_key}] 类型: {type(vv)}, 值: {vv}")
        elif isinstance(v, torch.Tensor):
            print(f"=== d[0][{first_key}] 形状: {v.shape}, 前10: {v[:10]}")
        else:
            print(f"=== d[0][{first_key}] 值: {v}")

    elif isinstance(c0, torch.Tensor):
        print("=== d[0] 形状:", c0.shape)
        
    for c in range(stats['num_classes']): 
        print(f"d[{c}] = {d[c]}")

if __name__ == '__main__':
    unittest.main()


