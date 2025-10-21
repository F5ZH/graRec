#!/usr/bin/env python3
"""
Minimal, competition-safe DINO-ViT classifier
- Uses ImageNet-1k self-supervised DINO weights from timm (e.g., vit_base_patch16_224.dino)
- Two modes: train / predict
- ArcFace optional; defaults to standard Linear
- AMP, cosine LR, warmup, label smoothing, RandAugment, Mixup/CutMix
- Saves best.pt (by top-1 acc on an optional val split) and final.pt

Directory assumptions
- train_dir: class-subfolders (ImageFolder)
- test_dir: images in subfolders or flat; we just walk files with typical extensions

CSV output (predict): filename,label(prob)
- label is zero-padded to 3/4 digits according to num_classes digits

This file is self-contained; requires: torch, timm, torchvision
"""
from __future__ import annotations
import argparse, math, os, sys, time, json, csv, random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes*")

try:
    import timm
except ImportError:
    raise RuntimeError("Please install timm: pip install timm>=0.9.7")

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# -------------------- Utils --------------------

def set_seed(seed: int=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class SmoothedCELoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, target):
        n = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

class ArcMarginProduct(nn.Module):
    """ArcFace head.
    logits = s * cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.25, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        x = F.normalize(x)
        W = F.normalize(self.weight)
        cos = F.linear(x, W).clamp(-1, 1)
        sin = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
        phi = cos * self.cos_m - sin * self.sin_m
        if not self.easy_margin:
            phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos)
        logits *= self.s
        return logits

# -------------------- Model --------------------
class ViTDinoClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool=True,
                 arcface: bool=False, scale: float=30.0, margin: float=0.25):
        super().__init__()
        # 先建模型，但不加载预训练
        self.backbone = timm.create_model(
            arch.replace("_dino", ".dino"),  # 确保用到 timm 内部正确tag
            pretrained=False, num_classes=0, global_pool='avg'
        )
        feat_dim = self.backbone.num_features

        if pretrained:
            # 手动加载预训练权重
            from torch.hub import load_state_dict_from_url
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            state_dict = load_state_dict_from_url(url, map_location="cpu")
            # 改 key: norm.* -> fc_norm.*
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith("norm."):
                    new_state["fc_" + k] = v
                else:
                    new_state[k] = v
            missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
            print(">>> Loaded pretrained DINO (fixed keys). Missing:", missing, "Unexpected:", unexpected)

        self.arcface = arcface
        if arcface:
            self.head = ArcMarginProduct(feat_dim, num_classes, s=scale, m=margin)
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        feats = self.backbone(x)  # [B, C]
        if self.arcface:
            assert labels is not None, "ArcFace requires labels in forward() during training"
            logits = self.head(feats, labels)
        else:
            logits = self.head(feats)
        return logits

# -------------------- Data --------------------

def build_transforms(img_size: int, is_train: bool, randaug_N: int=2, randaug_M: int=9):
    if is_train:
        tfms = [
            transforms.Resize(int(img_size*1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        # RandAugment (torchvision>=0.10 has it)
        try:
            from torchvision.transforms import RandAugment
            tfms.append(RandAugment(num_ops=randaug_N, magnitude=randaug_M))
        except Exception:
            pass
        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
    else:
        tfms = [
            transforms.Resize(int(img_size*1.15), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ]
    return transforms.Compose(tfms)

class TestFolder(Dataset):
    def __init__(self, root: str, transform):
        self.root = Path(root)
        self.transform = transform
        self.paths: List[Path] = []
        for p in sorted(self.root.rglob('*')):
            if p.suffix.lower() in IMG_EXTS:
                self.paths.append(p)
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {root}")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return self.transform(img), str(path)

# -------------------- Training --------------------

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res

def create_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndimension() == 1 or n.endswith('bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=lr)

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.t = 0
    def step_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi * t))
        for g in self.opt.param_groups: g['lr'] = lr
        return lr


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    # Data
    train_tf = build_transforms(args.img_size, True, args.randaug_N, args.randaug_M)
    val_tf   = build_transforms(args.img_size, False)

    ds_train = datasets.ImageFolder(args.train_dir, transform=train_tf)
    num_classes = len(ds_train.classes)

    if args.val_dir and os.path.isdir(args.val_dir):
        ds_val = datasets.ImageFolder(args.val_dir, transform=val_tf)
    else:
        ds_val = None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True) if ds_val else None

    # Model
    model = ViTDinoClassifier(
        arch=args.arch, num_classes=num_classes, pretrained=args.pretrained,
        arcface=args.arcface, scale=args.scale, margin=args.margin
    ).to(device)

    # Loss
    if args.arcface:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SmoothedCELoss(args.label_smoothing)

    opt = create_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    sched = WarmupCosine(opt, args.warmup_epochs, args.epochs, args.lr, args.min_lr)

    os.makedirs(args.outdir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        sched_lr = sched.step_epoch(epoch)
        t0 = time.time()
        loss_sum = 0.0
        acc1_sum = 0.0
        n_samples = 0
        for imgs, labels in dl_train:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(imgs, labels if args.arcface else None)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # metrics
            acc1 = accuracy(logits.detach(), labels, topk=(1,))[0]
            bs = imgs.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += acc1 * bs
            n_samples += bs
        train_loss = loss_sum / n_samples
        train_acc1 = acc1_sum / n_samples

        # Val
        if dl_val:
            model.eval()
            va_loss = 0.0; va_acc = 0.0; n = 0
            with torch.no_grad():
                for imgs, labels in dl_val:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs, labels if args.arcface else None)
                    loss = criterion(logits, labels)
                    acc1 = accuracy(logits, labels, topk=(1,))[0]
                    bs = imgs.size(0)
                    va_loss += loss.item() * bs
                    va_acc += acc1 * bs
                    n += bs
            va_loss /= max(1, n); va_acc /= max(1, n)
            is_best = va_acc > best_acc
            if is_best:
                best_acc = va_acc
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'acc1': best_acc, 'classes': ds_train.classes}, os.path.join(args.outdir, 'best.pt'))
            print(f"Epoch {epoch:03d} | lr {sched_lr:.2e} | train {train_loss:.4f}/{train_acc1:.2f} | val {va_loss:.4f}/{va_acc:.2f} | time {time.time()-t0:.1f}s",
                  flush=True)
        else:
            # No val: just log train
            print(f"Epoch {epoch:03d} | lr {sched_lr:.2e} | train {train_loss:.4f}/{train_acc1:.2f} | time {time.time()-t0:.1f}s", flush=True)

    # Save final
    torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'classes': ds_train.classes}, os.path.join(args.outdir, 'final.pt'))


def predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.is_file(), f"checkpoint not found: {ckpt_path}"

    # Infer classes from ckpt or a provided mapping
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = ckpt.get('classes', None)
    if classes is None:
        # Fallback: read from train_dir if provided
        if args.train_dir and os.path.isdir(args.train_dir):
            classes = datasets.ImageFolder(args.train_dir).classes
        else:
            raise ValueError("Classes not found in checkpoint; please provide --train_dir to infer class order.")
    num_classes = len(classes)

    model = ViTDinoClassifier(
        arch=args.arch, num_classes=num_classes, pretrained=False,
        arcface=False
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)

    tf = build_transforms(args.img_size, False)
    ds = TestFolder(args.test_dir, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    digit = max(3, len(str(num_classes-1)))
    os.makedirs(os.path.dirname(args.csv_path) or '.', exist_ok=True)

    with torch.no_grad(), open(args.csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for imgs, paths in dl:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for p, y, c in zip(paths, pred.cpu().tolist(), conf.cpu().tolist()):
                writer.writerow([p, str(y).zfill(digit), f"{c:.6f}"])

    print(f"Wrote predictions -> {args.csv_path}")

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description='DINO-ViT Classifier (minimal, competition-safe)')
    ap.add_argument('--mode', choices=['train','predict'], required=True)
    ap.add_argument('--arch', default='vit_base_patch16_224.dino', help='timm model, e.g. vit_small_patch16_224.dino / vit_base_patch16_224.dino')
    ap.add_argument('--pretrained', action='store_true', help='use timm pretrained (DINO on IN1k)')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--arcface', action='store_true')
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--scale', type=float, default=30.0)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--train_dir', type=str, default=None)
    ap.add_argument('--val_dir', type=str, default=None)
    ap.add_argument('--outdir', type=str, default='./runs/dino_vit')

    ap.add_argument('--test_dir', type=str, default=None)
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--csv_path', type=str, default='./pred_results.csv')

    ap.add_argument('--randaug_N', type=int, default=2)
    ap.add_argument('--randaug_M', type=int, default=9)

    args = ap.parse_args()

    if args.mode == 'train':
        assert args.train_dir and os.path.isdir(args.train_dir), 'train_dir not found'
        assert args.outdir, 'outdir required'
    else:
        assert args.test_dir and os.path.isdir(args.test_dir), 'test_dir not found'
        assert args.checkpoint, '--checkpoint required for predict'

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
