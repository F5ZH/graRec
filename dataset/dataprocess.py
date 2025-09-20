from torchvision import transforms

def get_train_transforms(args):
    """获取训练集数据增强和预处理"""
    return transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(args):
    """获取验证/测试集预处理"""
    return transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])