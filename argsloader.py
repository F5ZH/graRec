import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Swin Transformer Baseline for FGVC')

    # 数据路径
    parser.add_argument('--train_data_path', type=str, default='./data/WebFG-400/train', help='训练数据集根目录')
    parser.add_argument('--test_data_path', type=str, default='./data/WebFG-400/test', help='测试数据集根目录')
    parser.add_argument('--clean_val_data_path', type=str, default='', 
                        help='[可选] 纯净验证集根目录。若提供，则用于验证；若不提供，则跳过验证阶段，仅监控训练指标。')
    parser.add_argument('--output_dir', type=str, default='./runs', help='模型和日志输出目录')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='swin_base_patch4_window7_224', 
                        choices=['swin_tiny_patch4_window7_224', 
                                 'swin_small_patch4_window7_224', 
                                 'swin_base_patch4_window7_224',
                                 'swin_large_patch4_window7_224'],
                        help='选择 Swin Transformer 预训练模型')
    parser.add_argument('--num_classes', type=int, default=-1, help='类别数量，-1 表示从数据集自动推断')
    parser.add_argument('--pretrained', type=bool, default=True, help='是否加载 ImageNet 预训练权重')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备 (cuda/cpu)')

    # 数据预处理
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--resize_size', type=int, default=256, help='图像缩放尺寸')

    # 损失函数
    # 在 argsloader.py 的 get_args() 函数中添加：
    parser.add_argument('--loss_fn', type=str, default='default',
                    help='输入预设的损失函数组合，如 "default", "focal", "label_smoothing" 等')

    # 预测参数
    parser.add_argument('--checkpoint_path', type=str, default='./runs/best_model.pth', help='用于预测的模型路径')
    parser.add_argument('--submission_file', type=str, default='./runs/submission.csv', help='提交文件路径')

    args = parser.parse_args()
    return args