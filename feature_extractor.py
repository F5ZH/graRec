# feature_extractor.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 自动选择可用的模型库
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # 测试是否能加载
    HAS_DINO = True
except Exception:
    HAS_DINO = False


class FeatureExtractor:
    def __init__(self, model_name='resnet50', device='cuda'):
        self.device = device
        self.model_name = model_name.lower()
        self.model, self.preprocess = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self):
        if self.model_name.startswith('dinov2'):
            if not HAS_DINO:
                raise ImportError("DINOv2 requires torch>=1.13 and internet access for hub loading.")
            model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            # DINOv2 推荐输入尺寸：224（vits/vitb）或 518（vitl/vitg）
            size = 518 if 'vitl' in self.model_name or 'vitg' in self.model_name else 224
            preprocess = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return model, preprocess

        elif self.model_name.startswith('clip'):
            if not HAS_CLIP:
                raise ImportError("CLIP requires 'pip install git+https://github.com/openai/CLIP.git'")
            model, preprocess = clip.load(self.model_name.replace('clip-', ''), device=self.device)
            # CLIP 的 preprocess 已包含 ToTensor 和 Normalize
            return model.visual, preprocess  # 只用视觉编码器

        else:
            # 使用 timm 加载 ResNet、ViT 等
            if not HAS_TIMM:
                raise ImportError("Please install timm: pip install timm")
            model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,  # 移除分类头，直接输出特征
                global_pool='avg' if 'resnet' in self.model_name else None
            )
            data_config = timm.data.resolve_data_config(model.pretrained_cfg)
            preprocess = timm.data.create_transform(**data_config)
            return model, preprocess

    @torch.no_grad()
    def extract_features_from_paths(self, image_paths, batch_size=32):
        """
        从图像路径列表中提取特征。
        返回: numpy.ndarray, shape (N, D)
        """
        features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Feature extraction", leave=False):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    img = self.preprocess(img)
                    images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    print(f"⚠️ 跳过无效图像 {p}: {e}")
                    continue

            if not images:
                continue

            batch_tensor = torch.stack(images).to(self.device)
            batch_features = self.model(batch_tensor)

            # 确保输出为 2D
            if batch_features.ndim == 3:
                # ViT 类模型输出 (B, N+1, D)，取 [CLS] token
                batch_features = batch_features[:, 0]

            features.append(batch_features.cpu().numpy())

        if not features:
            return None
        return np.vstack(features)


def extract_features_for_folder(image_paths, model_name='dinov2_vits14', batch_size=32, device='cuda'):
    """
    便捷函数：为给定图像路径列表提取特征。
    """
    extractor = FeatureExtractor(model_name=model_name, device=device)
    return extractor.extract_features_from_paths(image_paths, batch_size=batch_size)