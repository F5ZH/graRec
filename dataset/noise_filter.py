import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import timm
from collections import defaultdict

class FeatureExtractor(nn.Module):
    """特征提取器"""
    def __init__(self, model_name='swin_tiny_patch4_window7_224', device='cuda'):
        super().__init__()
        # 加载预训练模型
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def extract_features(self, images):
        """提取图像特征"""
        return self.model(images)

class NoiseFilter:
    def __init__(self, eps=0.5, min_samples=5, device='cuda'):
        self.device = device
        self.feature_extractor = FeatureExtractor(device=device)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        
    def process_class_folder(self, class_path, transform):
        """处理单个类别文件夹中的图像"""
        image_paths = []
        features = []
        
        # 收集所有图像路径
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_paths.append(img_path)
        
        if not image_paths:
            return [], []
            
        # 批量提取特征
        batch_size = 32
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"提取特征: {os.path.basename(class_path)}"):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img).unsqueeze(0)
                    batch_imgs.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Warning: 无法加载图像 {path}: {e}")
                    
            if not batch_imgs:
                continue
                
            # 将批次图像组合并提取特征
            batch_tensor = torch.cat(batch_imgs).to(self.device)
            batch_features = self.feature_extractor.extract_features(batch_tensor)
            features.extend(batch_features.cpu().numpy())
            
        return image_paths, features
        
    def filter_noise(self, dataset_path, transform, save_path=None):
        """
        对整个数据集进行噪声过滤
        返回需要保留的图像路径列表
        """
        print(f"开始对数据集进行噪声过滤: {dataset_path}")
        all_keep_paths = []
        noise_stats = defaultdict(int)
        
        for class_name in tqdm(os.listdir(dataset_path), desc="处理类别"):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # 处理每个类别
            image_paths, features = self.process_class_folder(class_path, transform)
            
            if not features:
                continue
                
            # 标准化特征
            features = StandardScaler().fit_transform(features)
            
            # DBSCAN聚类
            labels = self.dbscan.fit_predict(features)
            
            # 统计并保存非噪声样本
            keep_paths = [path for path, label in zip(image_paths, labels) if label != -1]
            noise_paths = [path for path, label in zip(image_paths, labels) if label == -1]
            
            # 更新统计信息
            noise_stats[class_name] = {
                'total': len(image_paths),
                'kept': len(keep_paths),
                'removed': len(noise_paths)
            }
            
            all_keep_paths.extend(keep_paths)
            
            # 可选：将噪声样本移动到专门的文件夹
            if save_path:
                noise_dir = os.path.join(save_path, 'noise_samples', class_name)
                os.makedirs(noise_dir, exist_ok=True)
                for noise_path in noise_paths:
                    new_path = os.path.join(noise_dir, os.path.basename(noise_path))
                    try:
                        os.rename(noise_path, new_path)
                    except:
                        continue
        
        # 打印统计信息
        print("\n噪声过滤统计:")
        for class_name, stats in noise_stats.items():
            print(f"{class_name}: 总样本={stats['total']}, "
                  f"保留={stats['kept']}, 移除={stats['removed']}, "
                  f"噪声比例={stats['removed']/stats['total']*100:.2f}%")
        
        return all_keep_paths