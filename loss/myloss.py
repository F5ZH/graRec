import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# 1. 自定义损失函数区域 (您可以在此处添加自己的创新损失函数)
# ==============================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失。
    在标准交叉熵基础上对标签进行平滑，可以缓解模型对错误标签的过拟合，在噪声数据下有一定鲁棒性。
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SymmetricCrossEntropy(nn.Module):
    """
    对称交叉熵损失 (Symmetric Cross Entropy, SCE)。
    结合了标准交叉熵和反向交叉熵，旨在平衡噪声标签带来的影响。
    论文: "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
    """
    def __init__(self, alpha=0.1, beta=1.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        # 标准交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        # 反向交叉熵 (Reverse Cross Entropy)
        pred_probs = F.softmax(pred, dim=1)
        rce_loss = -torch.sum(F.one_hot(target, num_classes=pred.size(1)) * torch.log(pred_probs + 1e-7), dim=1)
        # 组合
        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss.mean()

class GeneralizedCrossEntropy(nn.Module):
    """
    广义交叉熵损失 (Generalized Cross Entropy, GCE)。
    通过引入 q 参数，使其在噪声环境下比标准交叉熵更鲁棒。
    论文: "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" (NeurIPS 2018)
    """
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def forward(self, pred, target):
        pred_probs = F.softmax(pred, dim=1)
        pred_for_target = pred_probs[range(pred.size(0)), target]
        loss = (1 - pred_for_target ** self.q) / self.q
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # (N,)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def compute_effective_num_weights(class_counts, beta=0.9999, eps=1e-8):
    """
    class_counts: dict {cls_idx: count}
    返回 torch.Tensor 的 weights，长度 = num_classes，未归一化（但会归一化到 sum = num_classes）
    使用 Effective Number re-weighting: w_i = (1 - beta) / (1 - beta^{n_i})
    """
    num_classes = max(class_counts.keys()) + 1
    counts = torch.ones(num_classes, dtype=torch.float)
    for k, v in class_counts.items():
        counts[k] = float(v)
    if beta < 0 or beta >= 1:
        # fallback to inverse-freq
        weights = 1.0 / (counts + eps)
    else:
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + eps)
    # 归一化，使 sum(weights) == num_classes（保持Loss scale稳定）
    weights = weights / weights.sum() * float(num_classes)
    return weights

# ==============================
# 2. 损失函数工厂方法
# ==============================

def get_loss_fn(loss_name, **kwargs):
    """
    根据名称获取损失函数实例。

    Args:
        loss_name (str): 损失函数名称。
        **kwargs: 传递给损失函数构造器的参数。

    Returns:
        nn.Module: 实例化的损失函数对象。

    Raises:
        ValueError: 如果指定的损失函数名称不存在。
    """

    # 定义一个映射字典，将字符串名称映射到对应的损失函数类或函数
    loss_dict = {
        # PyTorch 内置损失函数
        'cross_entropy': nn.CrossEntropyLoss,
        'ce': nn.CrossEntropyLoss,  # 别名

        # 自定义损失函数
        'label_smoothing_ce': LabelSmoothingCrossEntropy,
        'lsce': LabelSmoothingCrossEntropy,  # 别名
        'symmetric_ce': SymmetricCrossEntropy,
        'sce': SymmetricCrossEntropy,  # 别名
        'generalized_ce': GeneralizedCrossEntropy,
        'gce': GeneralizedCrossEntropy,  # 别名
        'focal': FocalLoss,
    }

    if loss_name not in loss_dict:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                         f"Supported losses are: {list(loss_dict.keys())}")

    # 使用传入的参数实例化损失函数
    return loss_dict[loss_name](**kwargs)

# ==============================
# 3. （可选）便捷的预设配置
# ==============================

def get_preset_loss_fn(preset_name):
    """
    根据预设名称获取带有推荐参数的损失函数。
    这个方法是可选的，方便快速实验常用配置。
    """
    preset_configs = {
        'default': ('cross_entropy', {}),
        'smooth_01': ('label_smoothing_ce', {'smoothing': 0.1}),
        'smooth_02': ('label_smoothing_ce', {'smoothing': 0.2}),
        'sce_balanced': ('symmetric_ce', {'alpha': 0.1, 'beta': 1.0}),
        'gce_q7': ('generalized_ce', {'q': 0.7}),
        'gce_q8': ('generalized_ce', {'q': 0.8}),
        'focal_gamma2': ('focal', {'gamma': 2.0}),
        'focal_gamma3': ('focal', {'gamma': 3.0}),
    }

    if preset_name not in preset_configs:
        raise ValueError(f"Unsupported preset: {preset_name}. "
                         f"Supported presets are: {list(preset_configs.keys())}")

    loss_name, kwargs = preset_configs[preset_name]
    return get_loss_fn(loss_name, **kwargs)


if __name__ == "__main__":
    # 简单的测试代码
    import numpy as np

    # 创建模拟数据
    batch_size, num_classes = 4, 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.tensor(np.random.randint(0, num_classes, batch_size))

    # 测试不同的损失函数
    for name in ['cross_entropy', 'label_smoothing_ce', 'symmetric_ce', 'generalized_ce', 'focal']:
        try:
            criterion = get_loss_fn(name)
            loss = criterion(logits, targets)
            print(f"{name}: {loss.item():.4f}")
        except Exception as e:
            print(f"Error with {name}: {e}")

    # 测试预设
    for preset in ['default', 'smooth_01', 'gce_q7', 'focal_gamma2']:
        try:
            criterion = get_preset_loss_fn(preset)
            loss = criterion(logits, targets)
            print(f"Preset '{preset}': {loss.item():.4f}")
        except Exception as e:
            print(f"Error with preset '{preset}': {e}")