from torchvision import transforms
from PIL import Image
import os
import warnings
import random

class RandomApply:
    """随机应用一个转换"""
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

def get_train_transforms(args):
    """获取增强的训练集数据增强和预处理"""
    return transforms.Compose([
        # 基础调整
        transforms.Resize(args.resize_size),
        transforms.RandomResizedCrop(
            args.img_size,
            scale=(0.8, 1.0),  # 较小的裁剪范围，保留更多细节
            ratio=(0.9, 1.1)   # 较小的长宽比变化
        ),
        
        # 几何增强
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 小角度旋转
        
        # 颜色增强
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        # 随机擦除
        transforms.RandomErasing(p=0.3),
        
        # 随机灰度
        transforms.RandomGrayscale(p=0.1),
        
        # 随机高斯模糊
        RandomApply(
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            p=0.2
        ),
        
        # 标准化
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_val_transforms(args):
    """获取验证/测试集预处理"""
    return transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def safe_pil_loader(path: str) -> Image.Image:
    """
    Load an image robustly:
      - tolerates truncated files
      - converts palette/alpha images to opaque RGB to remove PIL warnings
    On failure, raises RuntimeError with a [BROKEN_IMAGE] tag.
    """
    try: 
        with Image.open(path) as img:
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            return img
    except Exception as e:
        raise RuntimeError(f"[BROKEN_IMAGE] {path}: {repr(e)}")