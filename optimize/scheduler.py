import torch
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from typing import Any, Dict, Optional
from torch.optim.optimizer import Optimizer

def get_scheduler(optimizer: Optimizer, sched_name: str = 'cosine', epochs: int = 30, steps_per_epoch: int = 100, **kwargs):
    """
    根据名称获取学习率调度器实例。

    Args:
        optimizer (Optimizer): 优化器实例。
        sched_name (str): 调度器名称。
        epochs (int): 总训练轮数。
        steps_per_epoch (int): 每个epoch的步数（batch数量），对某些调度器是必需的。
        **kwargs: 传递给调度器构造函数的其他参数。

    Returns:
        Scheduler: 实例化的调度器对象（可以是 _LRScheduler 或 timm.scheduler）。

    Raises:
        ValueError: 如果指定的调度器名称不存在。
    """

    if sched_name == 'cosine_annealing':
        # PyTorch 内置的余弦退火
        T_max = kwargs.get('T_max', epochs)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=kwargs.get('eta_min', 0))
        print(f"[Scheduler] Created: CosineAnnealingLR | T_max: {T_max}")

    elif sched_name == 'one_cycle':
        # OneCycleLR 需要知道总步数
        total_steps = epochs * steps_per_epoch
        max_lr = kwargs.get('max_lr', optimizer.defaults['lr'])
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=kwargs.get('pct_start', 0.2),
            div_factor=kwargs.get('div_factor', 10),
            final_div_factor=kwargs.get('final_div_factor', 1000)
        )
        print(f"[Scheduler] Created: OneCycleLR | Max LR: {max_lr} | Total Steps: {total_steps}")

    elif sched_name == 'reduce_on_plateau':
        # 基于指标的调度器，通常在每个epoch后调用
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 假设我们监控的是准确率，越大越好
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
        print(f"[Scheduler] Created: ReduceLROnPlateau | Factor: {kwargs.get('factor', 0.5)} | Patience: {kwargs.get('patience', 5)}")

    elif sched_name == 'timm_cosine':
        # 使用 timm 的余弦调度器，支持 warmup
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs,
            lr_min=kwargs.get('lr_min', 1e-6),
            warmup_t=kwargs.get('warmup_t', 5),
            warmup_lr_init=kwargs.get('warmup_lr_init', 1e-6),
            warmup_prefix=kwargs.get('warmup_prefix', True)
        )
        print(f"[Scheduler] Created: timm.CosineLRScheduler | Warmup: {kwargs.get('warmup_t', 5)} epochs")

    elif sched_name == 'step_lr':
        # timm 的 Step 调度器
        scheduler = StepLRScheduler(
            optimizer,
            decay_t=kwargs.get('decay_t', 10),
            decay_rate=kwargs.get('decay_rate', 0.5),
            warmup_t=kwargs.get('warmup_t', 3),
            warmup_lr_init=kwargs.get('warmup_lr_init', 1e-6)
        )
        print(f"[Scheduler] Created: timm.StepLRScheduler | Decay every {kwargs.get('decay_t', 10)} epochs")

    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}. "
                         f"Supported schedulers are: cosine_annealing, one_cycle, reduce_on_plateau, timm_cosine, step_lr")

    return scheduler


def get_preset_scheduler(preset_name: str, optimizer: Optimizer, epochs: int, steps_per_epoch: int):
    """
    根据预设名称获取调度器实例。预设包含了与优化器协同工作的推荐配置。

    Args:
        preset_name (str): 预设名称。
        optimizer (Optimizer): 优化器实例。
        epochs (int): 总训练轮数。
        steps_per_epoch (int): 每个epoch的步数。

    Returns:
        Scheduler: 实例化的调度器对象。
    """

    preset_configs = {
        # 默认：简单的余弦退火
        'default': ('cosine_annealing', {'T_max': epochs, 'eta_min': 1e-6}),

        # 基线：使用 timm 的带 warmup 的余弦调度
        'baseline': ('timm_cosine', {'warmup_t': 5, 'lr_min': 1e-6}),

        # 快速收敛：OneCycle 策略，在一个周期内完成学习率升降
        'one_cycle_fast': ('one_cycle', {
            'max_lr': optimizer.defaults['lr'],
            'pct_start': 0.2,  # 20%的时间用于warmup
            'div_factor': 10,  # 初始学习率 = max_lr / 10
            'final_div_factor': 1000  # 最终学习率 = max_lr / 1000
        }),

        # 稳健策略：当指标停滞时降低学习率 (适用于有可靠验证集的情况)
        'reduce_plateau_safe': ('reduce_on_plateau', {
            'factor': 0.5,
            'patience': 3,
            'mode': 'max'  # 监控指标为准确率，越大越好
        }),

        # 分段下降：每10个epoch学习率减半
        'step_lr_10': ('step_lr', {'decay_t': 10, 'decay_rate': 0.5, 'warmup_t': 3}),
    }

    if preset_name not in preset_configs:
        raise ValueError(f"Unsupported scheduler preset: {preset_name}. "
                         f"Supported presets are: {list(preset_configs.keys())}")

    sched_name, kwargs = preset_configs[preset_name]
    return get_scheduler(optimizer, sched_name, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)


if __name__ == "__main__":
    import torch.nn as nn

    # 创建一个简单的模型和优化器用于测试
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    steps_per_epoch = 50  # 假设每个epoch有50个batch
    epochs = 20

    # 测试预设
    for preset in ['default', 'baseline', 'one_cycle_fast', 'step_lr_10']:
        try:
            print(f"\n--- Testing Scheduler Preset: {preset} ---")
            scheduler = get_preset_scheduler(preset, optimizer, epochs, steps_per_epoch)
            print("✅ Success")
        except Exception as e:
            print(f"❌ Failed: {e}")