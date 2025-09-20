import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_model, ensure_dir, set_seed
from .argsloader import get_args
from dataset.dataloader import create_dataloaders 
from models.mymodels import build_model
from loss.myloss import get_preset_loss_fn

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
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

    # 定义损失函数和优化器
    criterion = get_preset_loss_fn(args.loss_fn)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = 0.0  # 用于保存最佳模型的指标，可能是验证集准确率或训练集准确率
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    for epoch in range(1, args.epochs + 1):
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
            print(f'New best model saved with metric: {best_metric:.2f}%')

    writer.close()
    print(f'Training completed. Best metric: {best_metric:.2f}%')

if __name__ == '__main__':
    main()