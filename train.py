import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_model, ensure_dir, set_seed, save_args_json
from argsloader import get_args
from dataset.dataloader import create_dataloaders, robust_iter 
from models.mymodels import build_model
from loss.myloss import get_preset_loss_fn

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(robust_iter(dataloader), desc=f'Epoch {epoch} [Train]')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({'Loss': total_loss/len(dataloader), 'Acc': 100.*correct/total})

    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({'Loss': total_loss/len(dataloader), 'Acc': 100.*correct/total})

    return total_loss / len(dataloader), 100. * correct / total

def main():
    args = get_args()
    ensure_dir(args.output_dir)
    set_seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    train_loader, val_loader, num_classes = create_dataloaders(args)
    args.num_classes = num_classes  # 更新参数

    # 构建模型
    model = build_model(args)
    model.to(device)

    # 定义损失函数（将 args 传入以便使用 class_counts / reweight）
    criterion = get_preset_loss_fn(args.loss_fn)
    

    best_metric = 0.0  # 用于保存最佳模型的指标，可能是验证集准确率或训练集准确率
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    if args.freeze_backbone_epochs > 0:
        print(f"[策略] 前 {args.freeze_backbone_epochs} 个 epoch 将冻结骨干网络，仅训练分类头。")

    # 冻结骨干网络的所有参数
    for name, param in model.named_parameters():
        if "head" not in name:  # 假设 Swin Transformer 的分类层名为 'head' (timm 库中的命名)
            param.requires_grad = False
        else:
            print(f"  -> 训练参数: {name}")

    # 重新创建优化器，只包含需要更新的参数 (即分类头)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            print(f"[策略] 第 {epoch} 个 epoch，解冻骨干网络，开始全局微调。")

        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        # 重新创建优化器，包含所有参数
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate / 10, weight_decay=args.weight_decay)
        print("优化器已更新，包含所有参数。")
        # 重置学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # 记录训练指标
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # 初始化验证指标
        val_loss, val_acc = None, None

        # 仅在提供了纯净验证集时进行验证
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            current_metric = val_acc  # 以验证集准确率作为保存模型的依据
        else:
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}% (No Val Set)')
            current_metric = train_acc  # 以训练集准确率作为保存模型的依据

        scheduler.step()

        # 保存最佳模型
        # 策略：如果有验证集，按验证集准确率保存；如果没有，按训练集准确率保存
        if current_metric > best_metric:
            best_metric = current_metric
            save_model(model, optimizer, epoch, os.path.join(args.output_dir, 'best_model.pth'))
            save_args_json(os.path.join(args.output_dir, 'train_args.json'), args)
            print(f'New best model saved with metric: {best_metric:.2f}%')

    writer.close()
    print(f'Training completed. Best metric: {best_metric:.2f}%')

if __name__ == '__main__':
    main()