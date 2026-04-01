# tests/test_config.py
import sys
import os
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import config as cfg


class TestConfig(unittest.TestCase):
    """测试配置模块"""
    
    def test_config_exists(self):
        """测试配置文件是否存在"""
        self.assertIsNotNone(cfg)
    
    def test_data_root(self):
        """测试数据根目录配置"""
        self.assertIsNotNone(cfg.DATA_ROOT)
        print(f"Data root: {cfg.DATA_ROOT}")


if __name__ == '__main__':
    unittest.main()