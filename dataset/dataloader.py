import os
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from .dataprocess import get_train_transforms, get_val_transforms
from .dataprocess import safe_pil_loader  # <-- 新增导入

class CustomImageFolder(ImageFolder):
    """自定义 ImageFolder，用于兼容不同格式的图像"""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform, loader=safe_pil_loader)
        # ImageFolder 会自动根据子文件夹数量确定类别数
        # self.classes 是一个包含所有类别名的列表
        # self.class_to_idx 是一个类别名到索引的映射字典

def create_dataloaders(args):
    """创建训练和验证数据加载器"""
    train_transform = get_train_transforms(args)
    val_transform = get_val_transforms(args)

    # 创建数据集
    train_dataset = CustomImageFolder(root=args.train_data_path, transform=train_transform)

    # 动态设置类别数
    if args.num_classes == -1:
        args.num_classes = len(train_dataset.classes)
        print(f"自动检测到类别数量: {args.num_classes}")

    # 统计类别分布（ImageFolder 提供 targets）
    targets = train_dataset.targets  # list of labels
    class_counts = dict(Counter(targets))
    args.class_counts = class_counts  # 将统计信息保存到 args，供后续构造损失使用
    print(f"[Data] class counts sample: {list(class_counts.items())[:8]}  (total classes {len(class_counts)})")

    # 根据参数决定是否使用 WeightedRandomSampler
    train_loader = None
    if getattr(args, 'use_weighted_sampler', False):
        # 计算每个样本权重： weight = 1.0 / (count[label] ** sampling_power)
        sampling_power = float(getattr(args, 'sampling_power', 0.5))
        weights = [1.0 / (class_counts[label] ** sampling_power) for label in targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
        print(f"[Sampler] 使用 WeightedRandomSampler, sampling_power={sampling_power}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader = None
    # 如果提供了纯净验证集路径，则创建验证数据加载器
    if hasattr(args, 'clean_val_data_path') and args.clean_val_data_path and os.path.exists(args.clean_val_data_path):
        print(f"[重要] 加载纯净验证集: {args.clean_val_data_path}")
        val_dataset = CustomImageFolder(root=args.clean_val_data_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("[注意] 未提供纯净验证集路径，将跳过验证阶段，仅监控训练指标。")

    return train_loader, val_loader, args.num_classes

def robust_iter(dataloader):
    it = iter(dataloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except RuntimeError as e:
            if "[BROKEN_IMAGE]" in str(e):
                continue
            raise