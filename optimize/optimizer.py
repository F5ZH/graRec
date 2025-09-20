import torch
import torch.optim as optim
from typing import Any, Dict, Optional
from torch.optim.optimizer import Optimizer

def get_opt(model_parameters, opt_name: str = 'adamw', lr: float = 5e-5, weight_decay: float = 1e-4, **kwargs) -> Optimizer:
    """
    根据名称获取优化器实例。

    Args:
        model_parameters: 模型的参数，通常是 model.parameters()。
        opt_name (str): 优化器名称。
        lr (float): 学习率。
        weight_decay (float): 权重衰减。
        **kwargs: 传递给优化器构造函数的其他参数。

    Returns:
        torch.optim.Optimizer: 实例化的优化器对象。

    Raises:
        ValueError: 如果指定的优化器名称不存在。
    """

    # 定义优化器映射字典
    opt_dict = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'nadam': optim.NAdam,  # PyTorch 2.0+
        'rmsprop': optim.RMSprop,
    }

    if opt_name not in opt_dict:
        raise ValueError(f"Unsupported optimizer: {opt_name}. "
                         f"Supported optimizers are: {list(opt_dict.keys())}")

    optimizer_class = opt_dict[opt_name]
    optimizer = optimizer_class(model_parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    print(f"[Optimizer] Created: {optimizer.__class__.__name__} | LR: {lr} | WD: {weight_decay} | Extra Args: {kwargs}")
    return optimizer


def get_preset_opt(preset_name: str, model_parameters, base_lr: float = 5e-5, base_wd: float = 1e-4) -> Optimizer:
    """
    根据预设名称获取优化器实例。预设包含了针对特定场景调优的参数。

    Args:
        preset_name (str): 预设名称。
        model_parameters: 模型的参数。
        base_lr (float): 基础学习率，部分预设会在此基础上调整。
        base_wd (float): 基础权重衰减。

    Returns:
        torch.optim.Optimizer: 实例化的优化器对象。
    """

    preset_configs = {
        # 经典组合，稳定可靠
        'default': ('adamw', {'lr': base_lr, 'weight_decay': base_wd}),
        'baseline': ('adamw', {'lr': base_lr, 'weight_decay': base_wd}),

        # 针对大模型/噪声数据，使用稍大的 weight_decay 防止过拟合
        'robust_adamw': ('adamw', {'lr': base_lr, 'weight_decay': 5e-4}),

        # 使用 NAdam，可能在初期收敛更快
        'nadam_fast': ('nadam', {'lr': base_lr * 2, 'weight_decay': base_wd, 'momentum_decay': 0.004}),

        # 使用 SGD + Momentum，经典但需要更精细的LR调度
        'sgd_momentum': ('sgd', {'lr': base_lr * 10, 'weight_decay': base_wd, 'momentum': 0.9, 'nesterov': True}),
    }

    if preset_name not in preset_configs:
        raise ValueError(f"Unsupported optimizer preset: {preset_name}. "
                         f"Supported presets are: {list(preset_configs.keys())}")

    opt_name, kwargs = preset_configs[preset_name]
    # 允许预设中的参数覆盖传入的基础参数
    kwargs.setdefault('lr', base_lr)
    kwargs.setdefault('weight_decay', base_wd)

    return get_opt(model_parameters, opt_name=opt_name, **kwargs)


if __name__ == "__main__":
    import torch.nn as nn

    # 创建一个简单的模型用于测试
    model = nn.Linear(10, 2)
    parameters = model.parameters()

    # 测试预设
    for preset in ['default', 'robust_adamw', 'nadam_fast', 'sgd_momentum']:
        try:
            print(f"\n--- Testing Optimizer Preset: {preset} ---")
            optimizer = get_preset_opt(preset, parameters, base_lr=1e-3)
            print("✅ Success")
        except Exception as e:
            print(f"❌ Failed: {e}")