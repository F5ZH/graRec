import torch
import torch.nn as nn
from timm import create_model

class SwinBaseline(nn.Module):
    """以 Swin Transformer 为 Backbone 的 Baseline 模型"""

    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=1000, pretrained=True):
        super(SwinBaseline, self).__init__()
        # 从 timm 库创建预训练的 Swin Transformer 模型
        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # 注：timm 库中的 Swin 模型在创建时已根据 `num_classes` 参数自动调整了最后的分类头。
        # 因此，我们无需手动替换分类层，除非您想进行更复杂的修改。
        # 这里我们直接使用它，符合“不添加其他模块”的要求。

    def forward(self, x):
        return self.backbone(x)

def build_model(args):
    """根据参数构建模型"""
    model = SwinBaseline(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )
    return model