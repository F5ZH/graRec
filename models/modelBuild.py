from models.mymodels import SwinBaseline

def build_model_from_config(config):
    """
    根据配置字典构建模型。
    这是为了配合 load_model 中的 build_new_model 功能。
    """
    return SwinBaseline(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=False  # 加载权重时通常设为 False，因为权重已经包含在 checkpoint 中
    )