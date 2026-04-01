"""
数据加载模块
ID: CIFAR-10
OOD: SVHN, LSUN-crop, Textures
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_transforms(train=False):
    """DeiT标准预处理"""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_cifar10_loaders(data_root, batch_size, num_workers=4):
    """返回CIFAR-10的训练集和测试集loader"""
    # data_root="~/data" python不认识'~'，因此需要展开它
    data_root = os.path.expanduser(data_root)
    train_set = datasets.CIFAR10(
        root=data_root, train=True, download=True,
        transform=get_transforms(train=True)
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def get_ood_loader(name, data_root, batch_size, num_workers=4):
    """
    加载OOD数据集
    支持: svhn, lsun, textures
    """
    data_root = os.path.expanduser(data_root)

    transform = get_transforms(train=False)

    if name == "svhn":
        dataset = datasets.SVHN(
            root=os.path.join(data_root, "svhn"),
            split="test", download=True,
            transform=transform
        )
    elif name == "textures":
        # DTD (Describable Textures Dataset)
        dataset = datasets.DTD(
            root=os.path.join(data_root, "dtd"),
            split="test", download=True,
            transform=transform
        )
    elif name == "lsun":
        # LSUN-crop，需要手动下载放到data_root/lsun/
        # 如果没有，先跳过
        lsun_path =  os.path.expanduser(os.path.join(data_root, "lsun", "test"))
        if os.path.exists(lsun_path):
            dataset = datasets.ImageFolder(
                root=lsun_path, transform=transform
            )
        else:
            print(f"[WARN] LSUN数据集未找到: {lsun_path}，跳过")
            return None
    else:
        raise ValueError(f"不支持的OOD数据集: {name}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return loader