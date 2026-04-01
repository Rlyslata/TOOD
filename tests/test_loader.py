# tests/test_config.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from datasets.loader import get_ood_loader

# 添加项目根目录到路径

import config as cfg


class TestLoader(unittest.TestCase):
    """测试配置模块"""
    
    def test_ood_loader(self):
        """测试配置文件是否存在"""
        for ood_name in ["svhn", "textures", "lsun"]:
            ood_loader = get_ood_loader(
                ood_name, cfg.DATA_ROOT, cfg.BATCH_SIZE, cfg.NUM_WORKERS
            )
            if ood_loader is None:
                print(f"未找到{ood_name}数据集，跳过测试")
            else:
                print(f"{ood_name} 已存在")

if __name__ == '__main__':
    unittest.main()