from torchvision import transforms
from PIL import Image, ImageFilter
import os
import warnings
import random

# ======================
# 自定义增强：随机高斯模糊（兼容 PIL Image）
# ======================
class RandomGaussianBlur:
    """对 PIL Image 应用随机高斯模糊，避免在 ToTensor 前使用 Tensor-only 的 GaussianBlur"""
    def __init__(self, p=0.2, max_radius=1.0):
        """
        Args:
            p (float): 应用模糊的概率
            max_radius (float): 高斯模糊半径上限（PIL 中 radius 越大越模糊，通常 0.1~2.0）
        """
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(0.1, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

# ======================
# 数据预处理函数
# ======================
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
            saturation=0.2
        ),
    
        
        # 随机灰度
        transforms.RandomGrayscale(p=0.1),
        
        # ✅ 安全的高斯模糊（作用于 PIL Image）
        RandomGaussianBlur(p=0.2, max_radius=1.0),
        
        # 转为 Tensor 并标准化
        transforms.ToTensor(),
        # 随机擦除
        transforms.RandomErasing(p=0.3),
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


# ======================
# 安全图像加载器
# ======================
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