import os
from torch.utils.data import DataLoader, Dataset
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

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
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