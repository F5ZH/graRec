import os
import yaml
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from feature_extractor import extract_features_for_folder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

def is_low_quality(img_path, min_size=224, blur_thresh=100.0):
    """简单图像质量过滤"""
    from PIL import Image
    import cv2

    try:
        img = Image.open(img_path).convert('RGB')
        if min(img.size) < min_size:
            return True

        # 模糊检测（Laplacian variance）
        cv_img = np.array(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < blur_thresh:
            return True
    except Exception:
        return True
    return False

def clean_class_folder(class_path: Path, output_class_path: Path, config):
    """对单个类别进行清洗"""
    image_paths = [p for p in class_path.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if len(image_paths) == 0:
        return {'total': 0, 'kept': 0, 'removed': 0}

    # Step 1: 图像质量初筛
    quality_ok_paths = []
    for p in image_paths:
        if not is_low_quality(str(p), 
                              min_size=config['quality']['min_resolution'],
                              blur_thresh=config['quality']['blur_threshold']):
            quality_ok_paths.append(p)
    
    if len(quality_ok_paths) == 0:
        return {'total': len(image_paths), 'kept': 0, 'removed': len(image_paths)}

    # Step 2: 提取特征（使用预训练模型）
    features = extract_features_for_folder(quality_ok_paths, 
                                           model_name=config['feature']['model'],
                                           batch_size=config['feature']['batch_size'],
                                           device=config['device'])
    
    if features is None or len(features) == 0:
        return {'total': len(image_paths), 'kept': 0, 'removed': len(image_paths)}

    # Step 3: 标准化 + L2 归一化
    # features = StandardScaler().fit_transform(features)
    features = np.array(features)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    if len(quality_ok_paths) < 3:
    # 样本太少，无法聚类，全部保留（或按质量过滤后保留）
        keep_paths = quality_ok_paths
    else:

    # Step 4: DBSCAN 聚类去噪
        dbscan = DBSCAN(
            eps=config['dbscan']['eps'],
            min_samples=config['dbscan']['min_samples'],
            metric='euclidean'
        )
        labels = dbscan.fit_predict(features)

    # Step 5: 保留非噪声点
        keep_paths = [p for p, label in zip(quality_ok_paths, labels) if label != -1]

    # Step 6: 复制保留的图像到输出目录
    output_class_path.mkdir(parents=True, exist_ok=True)
    for p in keep_paths:
        shutil.copy2(p, output_class_path / p.name)

    return {
        'total': len(image_paths),
        'kept': len(keep_paths),
        'removed': len(image_paths) - len(keep_paths)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='原始数据集路径')
    parser.add_argument('--output', type=str, required=True, help='清洗后输出路径')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_root = Path(args.input)
    output_root = Path(args.output)
    stats = defaultdict(dict)

    for class_dir in tqdm(list(input_root.iterdir()), desc="Cleaning classes"):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        output_class_dir = output_root / class_name
        stat = clean_class_folder(class_dir, output_class_dir, config)
        stats[class_name] = stat

    # 保存统计报告
    total_original = sum(v['total'] for v in stats.values())
    total_kept = sum(v['kept'] for v in stats.values())
    print(f"\n✅ 清洗完成！\n原始样本: {total_original}\n保留样本: {total_kept}\n保留率: {total_kept/total_original*100:.2f}%")

    with open(output_root / "cleaning_report.yaml", 'w') as f:
        yaml.dump(dict(stats), f, default_flow_style=False)

if __name__ == '__main__':
    main()