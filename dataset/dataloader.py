import os
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from .dataprocess import get_train_transforms, get_val_transforms
from .dataprocess import safe_pil_loader
from .noise_filter import NoiseFilter
import pickle

class BalancedImageFolder(ImageFolder):
    """支持重采样的 ImageFolder"""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform, loader=safe_pil_loader)
        
        # 统计每个类别的样本数
        self.targets = np.array(self.targets)
        class_counts = Counter(self.targets)
        self.class_counts = class_counts
        
        # 计算重采样权重
        self.weights = self._compute_weights()
        
    def _compute_weights(self):
        # 获取所有类别的样本数
        counts = np.array([self.class_counts[i] for i in range(len(self.class_counts))])
        
        # 计算每个样本的权重：1 / 类别频率
        weights = 1.0 / counts[self.targets]
        
        # 归一化权重，使其和为样本数
        weights = weights * len(self.targets) / weights.sum()
        
        return torch.DoubleTensor(weights)

    @staticmethod
    def filter_noise_samples(root, transform, cache_path=None, eps=0.5, min_samples=5):
        """使用DBSCAN过滤噪声样本"""
        # 如果存在缓存，直接加载
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # 创建噪声过滤器并处理数据集
        noise_filter = NoiseFilter(eps=eps, min_samples=min_samples)
        keep_paths = noise_filter.filter_noise(root, transform)
        
        # 保存结果到缓存
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(keep_paths, f)
        
        return keep_paths

def create_dataloaders(args):
    """创建训练和验证数据加载器"""
    train_transform = get_train_transforms(args)
    val_transform = get_val_transforms(args)
    
    # 添加噪声过滤
    if getattr(args, 'use_noise_filter', False):
        print("开始进行噪声过滤...")
        cache_path = os.path.join(args.output_dir, 'noise_filter_cache.pkl')
        keep_paths = BalancedImageFolder.filter_noise_samples(
            args.train_data_path,
            train_transform,
            cache_path=cache_path,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples
        )
        # 使用过滤后的图像路径创建数据集
        train_dataset = BalancedImageFolder(
            root=args.train_data_path,
            transform=train_transform
        )
        train_dataset.samples = [(p, train_dataset.class_to_idx[os.path.basename(os.path.dirname(p))])
                                for p in keep_paths]
        train_dataset.targets = [s[1] for s in train_dataset.samples]
    else:
        train_dataset = BalancedImageFolder(
            root=args.train_data_path,
            transform=train_transform
        )
    # 动态设置类别数
    if args.num_classes == -1:
        args.num_classes = len(train_dataset.classes)
        print(f"自动检测到类别数量: {args.num_classes}")

    # 打印类别分布信息
    print("\n类别分布统计:")
    class_counts = train_dataset.class_counts
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"最大类别样本数: {max_count}, 最小类别样本数: {min_count}, 不平衡比例: {max_count/min_count:.2f}")

    # 创建带重采样的训练数据加载器
    sampler = WeightedRandomSampler(
        weights=train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=sampler,  # 使用重采样
        num_workers=4,
        pin_memory=True
    )

    # 验证集保持不变，不需要重采样
    val_loader = None
    if hasattr(args, 'clean_val_data_path') and args.clean_val_data_path and os.path.exists(args.clean_val_data_path):
        print(f"[重要] 加载纯净验证集: {args.clean_val_data_path}")
        val_dataset = ImageFolder(
            root=args.clean_val_data_path, 
            transform=val_transform,
            loader=safe_pil_loader
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
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