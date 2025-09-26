import os
import random
import numpy as np
import torch
import json
import datetime

from models.modelBuild import build_model_from_config


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, optimizer, epoch, path, model_args=None):
    """
    保存模型、优化器状态和模型配置参数。
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前训练轮数
        path: 保存路径
        model_args: 包含模型配置的参数对象 (如 args)，可选但强烈推荐
    """
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if model_args is not None:
        config = {
            'model_name': model_args.model_name,
            'num_classes': model_args.num_classes,
            'pretrained': model_args.pretrained,
            'img_size': getattr(model_args, 'img_size', 224),
        }
        save_dict['model_config'] = config
        print(f"[保存配置] 已保存模型配置: {config}")

    torch.save(save_dict, path)
    print(f"模型已保存至: {path}")


def load_model(model, path, device, build_new_model=False, model_args=None):
    """
    加载模型权重，并可选择根据保存的配置重建模型。
    
    Args:
        model: 模型实例 (如果 build_new_model=False)
        path: 模型权重文件路径
        device: 设备
        build_new_model: 是否根据保存的配置重建一个新模型
        model_args: 当前的参数对象，用于覆盖保存的配置（可选）
    
    Returns:
        model: 加载了权重的模型实例（已移至 device）
    """
    checkpoint = torch.load(path, map_location=device)
    
    if build_new_model:
        if 'model_config' not in checkpoint:
            raise ValueError("Checkpoint 中未找到模型配置，无法重建模型。")
        
        config = checkpoint['model_config']
        print(f"[加载配置] 使用保存的模型配置: {config}")
        
        if model_args is not None:
            config['num_classes'] = model_args.num_classes
            print(f"[加载配置] 使用命令行参数覆盖类别数: {model_args.num_classes}")
        
        model = build_model_from_config(config)
        # 注意：此时 model 在 CPU，下面统一 .to(device)

    # === 关键修复：正确处理权重键名映射（兼容旧 checkpoint）===
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # 情况1: 旧 checkpoint（无 .fc） -> 映射到新结构（有 .fc）
        if key == 'backbone.head.weight':
            new_key = 'backbone.head.fc.weight'
            print(f"[映射] {key} -> {new_key}")
        elif key == 'backbone.head.bias':
            new_key = 'backbone.head.fc.bias'
            print(f"[映射] {key} -> {new_key}")
        # 情况2: 新 checkpoint（有 .fc）或其它层 -> 直接保留
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 加载权重
    model.load_state_dict(new_state_dict, strict=True)
    
    # === 关键修复：确保模型在目标设备上 ===
    model.to(device)
    
    print(f"模型权重已成功加载，并已移至设备: {device}")
    return model


def save_args_json(path, args):
    """
    将 argparse.Namespace 或类似对象保存为可读的 JSON 文件。
    不可序列化的对象会被转换为字符串。
    """
    try:
        raw = vars(args)
    except TypeError:
        raw = getattr(args, '__dict__', dict(args))

    def _convert(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        return str(obj)

    serializable = {k: _convert(v) for k, v in raw.items()}
    serializable['_saved_at'] = datetime.datetime.now().isoformat()

    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"训练参数已保存为 JSON: {path}")