#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file, single-model training & inference pipeline for
"Network-Supervised Fine-Grained Image Recognition" competition.

✅ Constraints:
- Uses ONLY ImageNet-1k pretrained weights (timm/torchvision) ✔
- Single model (no ensemble at inference) ✔
- Noise-robust losses (ELR+/SCE/GCE), long-tail aware ✔
- Fine-grained friendly (WS-DAN: erase + zoom-in) ✔
- Exports required CSVs ✔
"""

# ---------- CUDA memory fragmentation guard (set BEFORE importing torch) ----------
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import math
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import warnings

#visualization
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


# Robust PIL settings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated files to load
Image.MAX_IMAGE_PIXELS = 120_000_000       # disable DecompressionBombError

try:
    import timm  # convnext backbone
except Exception:
    timm = None

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pad4(s: str) -> str:
    return s if len(s) >= 4 else ("0" * (4 - len(s)) + s)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -------------------------
# Robust image loader
# -------------------------
def safe_pil_loader(path: str) -> Image.Image:
    """
    Load an image robustly:
      - tolerates truncated files
      - converts palette/alpha images to opaque RGB to remove PIL warnings
    On failure, raises RuntimeError with a [BROKEN_IMAGE] tag.
    """
    try: 
        with Image.open(path) as img:
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            return img
    except Exception as e:
        raise RuntimeError(f"[BROKEN_IMAGE] {path}: {repr(e)}")

# -------------------------
# Data & Augmentations
# -------------------------
class AlbumentationsWrapper:
    """Optional: use albumentations if installed; otherwise fall back to torchvision transforms."""
    def __init__(self, size: int, mode: str, randaug: Optional[Tuple[int,int]] = None):
        self.size = size
        self.mode = mode
        self.randaug = randaug

        # torchvision transforms
        t_list = [transforms.Resize(int(size*1.15))]
        if mode == 'train':
            # RandAugment if requested
            if randaug is not None and hasattr(transforms, "RandAugment"):
                N, M = randaug
                t_list.append(transforms.RandAugment(num_ops=int(N), magnitude=int(M)))
            t_list += [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            t_list += [transforms.CenterCrop(size)]

        t_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ]
        self.tf = transforms.Compose(t_list)

    def __call__(self, img):
        return self.tf(img)

def make_class_balanced_sampler(dataset: ImageFolder, mode: str = "sqrt"):
    # dataset.targets is list[int] for ImageFolder
    y = np.array(dataset.targets, dtype=np.int64)
    counts = np.bincount(y)
    counts[counts == 0] = 1
    if mode == "sqrt":
        w_class = 1.0 / np.sqrt(counts)
    else:
        w_class = 1.0 / counts
    weights = w_class[y]
    weights = torch.as_tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def build_loaders(train_dir: str,
                  val_dir: Optional[str],
                  img_size: int,
                  batch_size: int,
                  workers: int,
                  randaug: Optional[Tuple[int,int]],
                  mixup: float,
                  cutmix: float,
                  class_balanced: bool):
    train_tf = AlbumentationsWrapper(img_size, 'train', randaug)
    val_tf   = AlbumentationsWrapper(img_size, 'val', None)

    train_ds = ImageFolder(train_dir, transform=train_tf, loader=safe_pil_loader)
    val_ds = ImageFolder(val_dir, transform=val_tf, loader=safe_pil_loader) if val_dir and os.path.isdir(val_dir) else None

    sampler = make_class_balanced_sampler(train_ds) if class_balanced else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=True,
        prefetch_factor=(2 if workers > 0 else None)
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
        prefetch_factor=(2 if workers > 0 else None)
    ) if val_ds else None

    return train_loader, val_loader, train_ds

# -------------------------
# Losses for noisy labels
# -------------------------
class ELRPlus(nn.Module):
    """ELR+ loss (single-network) with bounded memory buffer."""
    def __init__(self, num_classes: int, lmbda: float = 3.0, beta: float = 0.7, eps: float = 1e-8, mem_size: int = 200000):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ = lmbda
        self.beta = beta
        self.eps = eps
        self.mem_size = int(mem_size)
        # allocate buffer to mem_size x C (on device during training)
        self.register_buffer('target', torch.zeros(self.mem_size, num_classes))

    def forward(self, logits, y, idx):
        # logits: (B, C), y: (B,), idx: (B,) any integer indices; we mod by mem_size
        probs = torch.softmax(logits, dim=1)
        idx = idx % self.mem_size
        pt = probs.detach()
        # moving average target
        self.target[idx] = self.beta * self.target[idx] + (1 - self.beta) * pt
        regularizer = (self.target[idx] * probs).sum(dim=1)
        ce = F.cross_entropy(logits, y, reduction='none')
        loss = ce + self.lambda_ * (-torch.log(1 - regularizer + self.eps))
        return loss

class SCELoss(nn.Module):
    """Symmetric Cross Entropy: CE + RCE"""
    def __init__(self, alpha=0.1, beta=1.0, num_classes=1000):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pred = torch.softmax(logits, dim=1)
        y_onehot = F.one_hot(targets, self.num_classes).float()
        rce = -torch.sum(pred.clamp(min=1e-7, max=1.0) * torch.log(y_onehot.clamp(min=1e-4)), dim=1) / pred.size(0)
        return self.alpha * ce + self.beta * rce

class GCELoss(nn.Module):
    """Generalized Cross Entropy"""
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        p = torch.softmax(logits, dim=1)
        p_y = p.gather(1, targets.view(-1,1)).squeeze(1)
        if abs(self.q - 1.0) < 1e-6:
            loss = -torch.log(p_y)
        else:
            loss = (1 - p_y ** self.q) / self.q
        return loss

# ArcFace head (margin-based)
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.25, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1,1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits

# -------------------------
# WS-DAN style attention
# -------------------------
class WSDAN(nn.Module):
    def __init__(self, in_ch: int, K: int = 8):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_ch, K, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor):
        attn = F.relu(self.conv(feat))  # (B, K, H, W)
        B, K, H, W = attn.size()
        attn = attn.reshape(B*K, 1, H, W)
        attn = attn / (attn.amax(dim=(2,3), keepdim=True) + 1e-6)
        attn = attn.view(B, K, H, W)
        return attn

def wsdan_erase(images: torch.Tensor, attn_maps: torch.Tensor, p: float = 0.5, thresh: float = 0.5):
    """
    Erase with WS-DAN attention; upsample attention maps to image size first.
    images:  (B, 3, H, W)
    attn:    (B, K, h, w) in [0,1], usually h,w << H,W
    """
    if p <= 0 or attn_maps is None:
        return images
    B, C, H, W = images.shape

    # ↑↑ 关键：把注意力从 (h,w) 双线性上采样到 (H,W)
    attn_maps = F.interpolate(attn_maps, size=(H, W), mode='bilinear', align_corners=False)
    # 保险起见再压到 [0,1]
    attn_maps = attn_maps.clamp_(0, 1)

    out = images.clone()
    for b in range(B):
        if random.random() < p:
            k = random.randrange(attn_maps.size(1))
            mask = (attn_maps[b, k] > thresh).to(dtype=images.dtype)
            out[b] = out[b] * (1 - mask.unsqueeze(0))  # (1,H,W) 广播到 3 通道
    return out


# -------------------------
# Model wrapper
# -------------------------
class ConvNeXtWS(nn.Module):
    def __init__(self, arch: str, num_classes: int, use_wsdan: bool = True, K: int = 8,
                 arcface: bool = True, m: float=0.25, s: float=30.0, pretrained: bool=False):
        super().__init__()
        assert timm is not None, "Please install timm: pip install timm"
        model_name = {
            'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k',
            'convnext_small': 'convnext_small.fb_in22k_ft_in1k',
            'convnext_base': 'convnext_base.fb_in22k_ft_in1k',
        }.get(arch, 'convnext_base.fb_in22k_ft_in1k')
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=False, num_classes=0)
        self.num_classes = num_classes
        self.use_wsdan = use_wsdan
        self.arcface = arcface

        feat_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if use_wsdan:
            self.wsdan = WSDAN(in_ch=feat_dim, K=K)
        self.dropout = nn.Dropout(0.1)
        self.feat_proj = nn.Linear(feat_dim, 1024)
        if arcface:
            self.margin_head = ArcMarginProduct(1024, num_classes, s=s, m=m)
        else:
            self.fc = nn.Linear(1024, num_classes)

    @torch.no_grad()
    def get_attn(self, x: torch.Tensor):
        feat = self.backbone.forward_features(x)  # (B, C, H, W)
        return self.wsdan(feat) if self.use_wsdan else None

    def forward(self, x, y: Optional[torch.Tensor] = None, return_attn: bool = False):
        feat = self.backbone.forward_features(x)  # (B, C, H, W)
        attn_maps = self.wsdan(feat) if self.use_wsdan else None
        pooled = self.global_pool(feat).flatten(1)
        z = F.relu(self.feat_proj(self.dropout(pooled)))
        if self.arcface:
            if y is not None:
                logits = self.margin_head(z, y)  # 训练：带 margin
            else:
                # 推理：用训练好的 ArcFace 权重做“无 margin”余弦分类
                logits = F.linear(F.normalize(z), F.normalize(self.margin_head.weight)) * self.margin_head.s
        else:
            logits = self.fc(z)
        if return_attn:
            return logits, z, attn_maps
        return logits

# -------------------------
# MixUp / CutMix
# -------------------------
def mixup_cutmix(images, targets, mixup_alpha: float, cutmix_alpha: float, num_classes: int):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, targets, None
    lam = 1.0
    if cutmix_alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        B, C, H, W = images.size()
        cx = np.random.randint(W); cy = np.random.randint(H)
        w = int(W * np.sqrt(1 - lam)); h = int(H * np.sqrt(1 - lam))
        x0 = np.clip(cx - w // 2, 0, W); x1 = np.clip(cx + w // 2, 0, W)
        y0 = np.clip(cy - h // 2, 0, H); y1 = np.clip(cy + h // 2, 0, H)
        idx = torch.randperm(B, device=images.device)
        images[:, :, y0:y1, x0:x1] = images[idx, :, y0:y1, x0:x1]
        lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        images = lam * images + (1 - lam) * images[idx]
    y1 = targets
    y2 = targets[idx]
    y1_onehot = F.one_hot(y1, num_classes=num_classes).float()
    y2_onehot = F.one_hot(y2, num_classes=num_classes).float()
    targets_mixed = lam * y1_onehot + (1 - lam) * y2_onehot
    return images, targets_mixed, (y1, y2, lam)

def criterion_with_soft(logits, soft_targets):
    log_p = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_p).sum(dim=1)
    return loss

# -------------------------
# Robust iterator (skip broken images)
# -------------------------
def robust_iter(dataloader):
    it = iter(dataloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except RuntimeError as e:
            if "[BROKEN_IMAGE]" in str(e):
                continue
            raise

# -------------------------
# Training / Evaluation
# -------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    set_seed(args.seed)

    #设置可视化存储路径
    log_path = args.outdir + '/tb_logs'
    writer = SummaryWriter(log_dir=log_path)
    print(f"📊 TensorBoard logs will be saved to: {log_path}")

    # 存储超参
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    writer.add_hparams(hparams, {})

    # data
    train_loader, val_loader, train_ds = build_loaders(
        args.train_dir, args.val_dir, args.img_size, args.batch_size, args.workers,
        (args.randaug_N, args.randaug_M) if args.randaug_N>0 else None,
        args.mixup, args.cutmix, args.class_balanced)

    num_classes = len(train_ds.classes)
    print(f"Num classes: {num_classes}")

    # save mapping
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {i: c for i, c in enumerate(train_ds.classes)}
    class_to_idx = train_ds.class_to_idx
    save_json({'idx_to_class': idx_to_class, 'class_to_idx': class_to_idx, 'num_classes': num_classes}, outdir / 'classes.json')

    # model
    model = ConvNeXtWS(args.arch, num_classes, use_wsdan=args.use_wsdan, K=args.K,
                       arcface=args.arcface, m=args.margin, s=args.scale, pretrained=args.pretrained)
    model = model.to(device)

    # === NEW: init from a previous checkpoint (weights only) ===
    if getattr(args, "init_from", None):
        ckpt = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        print(f"[InitFrom] Loaded weights from {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # memory format & grad ckpt & TF32
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if hasattr(model.backbone, "set_grad_checkpointing") and args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # param groups with LR multiplier for backbone
    bb_ids = set(id(p) for p in model.backbone.parameters())
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    bb_params   = [p for p in model.parameters() if id(p) in bb_ids]

    optim_groups = [
        {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": bb_params,   "lr": args.lr * args.backbone_lr_mult, "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.AdamW(optim_groups)

    # freeze backbone for first N epochs (if requested)
    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"[Init] Freeze backbone for {args.freeze_backbone_epochs} epoch(s).")

    # scheduler: cosine w/ warmup (per-iteration)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_epochs * len(train_loader)))

    def cosine_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # losses
    elr_mem_size = args.elr_mem if args.elr_mem > 0 else len(train_ds)
    elr_mem = ELRPlus(num_classes, lmbda=args.elr_lambda, mem_size=elr_mem_size).to(device) if args.loss == 'elrplus' else None
    sce = SCELoss(alpha=args.sce_alpha, beta=args.sce_beta, num_classes=num_classes).to(device) if args.loss == 'sce' else None
    gce = GCELoss(q=args.gce_q).to(device) if args.loss == 'gce' else None

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_acc = 0.0
    global_step = 0
    seen_optim_step = False

    micro = max(1, int(args.microbatch))

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # unfreeze when reaching epoch == freeze_backbone_epochs
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("[Unfreeze] Backbone unfrozen.")

        model.train()
        loss_meter, acc_meter = [], []
        # curriculum ratio
        p_keep = args.keep_ratio_final + (args.keep_ratio_start - args.keep_ratio_final) * max(0, (args.curriculum_epochs - max(0, epoch-args.curriculum_start))) / max(1,args.curriculum_epochs)

        for imgs, ys in tqdm(robust_iter(train_loader), desc=f"Train e{epoch}"):
            imgs = imgs.to(device, non_blocking=True)
            ys   = ys.to(device, non_blocking=True)
            if args.channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            # optional MixUp/CutMix
            soft_targets = None
            if args.mixup>0 or args.cutmix>0:
                imgs, soft_targets, mix_meta = mixup_cutmix(imgs, ys, args.mixup, args.cutmix, num_classes)

            # WS-DAN erase via no-grad attention
            use_erase = args.use_wsdan and (epoch >= args.wsdan_warm)
            if use_erase:
                attn = model.get_attn(imgs)
                imgs = wsdan_erase(imgs, attn, p=args.erase_p)

            B = imgs.size(0)
            mb = min(micro, B)
            accum_steps = (B + mb - 1) // mb

            optimizer.zero_grad(set_to_none=True)
            batch_loss_val = 0.0
            batch_corr = 0
            batch_tot  = 0

            for s in range(0, B, mb):
                e = min(B, s + mb)
                imgs_mb = imgs[s:e]
                ys_mb   = ys[s:e]

                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits_mb, feats_mb, _ = model(imgs_mb, ys_mb if model.arcface else None, return_attn=True)

                    if soft_targets is not None:
                        soft_mb = soft_targets[s:e]
                        ce_mb = criterion_with_soft(logits_mb, soft_mb)
                    else:
                        ce_mb = F.cross_entropy(logits_mb, ys_mb, reduction='none', label_smoothing=args.label_smooth)

                    if args.loss == 'elrplus':
                        idxs_mb = torch.arange(global_step + s, global_step + e, device=device)
                        noise_reg_mb = elr_mem(logits_mb, ys_mb, idxs_mb)
                        loss_all_mb = ce_mb + noise_reg_mb
                    elif args.loss == 'sce':
                        loss_all_mb = sce(logits_mb, ys_mb)
                    elif args.loss == 'gce':
                        loss_all_mb = gce(logits_mb, ys_mb)
                    else:
                        loss_all_mb = ce_mb

                    if epoch >= args.curriculum_start:
                        k_mb = int(max(1, p_keep * loss_all_mb.size(0)))
                        _, topk_idx_mb = torch.topk(-loss_all_mb, k_mb)
                        loss_mb = loss_all_mb[topk_idx_mb].mean()
                    else:
                        loss_mb = loss_all_mb.mean()

                    loss_mb = loss_mb / accum_steps

                scaler.scale(loss_mb).backward()

                with torch.no_grad():
                    pred_mb = logits_mb.argmax(1)
                    batch_corr += (pred_mb == ys_mb).sum().item()
                    batch_tot  += ys_mb.numel()
                    batch_loss_val += loss_mb.item() * accum_steps

            scaler.step(optimizer)     # optimizer.step() first
            scaler.update()
            seen_optim_step = True
            scheduler.step()           # then scheduler.step()
            global_step += B

            acc = batch_corr / max(1, batch_tot)
            acc_meter.append(acc)
            loss_meter.append(batch_loss_val)

        print(f"Epoch {epoch}: loss={np.mean(loss_meter):.4f}, acc~={np.mean(acc_meter):.4f}")

        #记录相关值用于可视化
        train_loss_mean = np.mean(loss_meter)
        train_acc_mean = np.mean(acc_meter)
        writer.add_scalar('Loss/Train', train_loss_mean, epoch)
        writer.add_scalar('Accuracy/Train', train_acc_mean, epoch)

        # validate
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            print(f"  VAL acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, outdir / 'best.pt')
        else:
            if (epoch+1) % 5 == 0:
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, outdir / 'last.pt')
        writer.flush()

    torch.save({'model': model.state_dict(), 'epoch': args.epochs-1, 'args': vars(args)}, outdir / 'final.pt')
    print(f"Training finished. Checkpoints saved in {outdir}")

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for imgs, ys in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            logits = model(imgs, None, return_attn=False)
            pred = logits.argmax(1)
            corr += (pred == ys).sum().item()
            tot += ys.numel()
    return corr / max(1, tot)

# -------------------------
# Prediction / CSV export
# -------------------------
class TestImages(Dataset):
    def __init__(self, root: str, size: int):
        self.root = root
        self.size = size
        self.files = []
        for dp, _, fnames in os.walk(root):
            for f in fnames:
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                    self.files.append(os.path.join(dp, f))
        self.files.sort()
        self.tf = transforms.Compose([
            transforms.Resize(int(size*1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = safe_pil_loader(path)
        return self.tf(img), path

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    ck_args = ckpt.get('args', {})
    # classes.json should be located beside checkpoint (or via --classes)
    classes_path = args.classes if args.classes else (Path(args.checkpoint).parent / 'classes.json')
    cls_map = load_json(classes_path)
    idx_to_class = {int(k): v for k, v in cls_map['idx_to_class'].items()}
    num_classes = len(idx_to_class)

    model = ConvNeXtWS(
        arch=ck_args.get('arch', args.arch),
        num_classes=num_classes,
        use_wsdan=ck_args.get('use_wsdan', True),
        K=ck_args.get('K', 8),
        arcface=ck_args.get('arcface', True),
        m=ck_args.get('margin', 0.25),
        s=ck_args.get('scale', 30.0),
        pretrained=False
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)

    ds = TestImages(args.test_dir, args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=(args.workers>0),
                    prefetch_factor=(2 if args.workers>0 else None))

    records = []
    with torch.no_grad():
        for imgs, paths in tqdm(dl, desc='Predict'):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs, None)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1).cpu().numpy()
            for p, path in zip(pred, paths):
                fname = os.path.basename(path)
                cls = idx_to_class[int(p)]
                cls4 = pad4(cls)
                records.append((fname, cls4))

    df = pd.DataFrame(records, columns=['filename','label'])
    df.to_csv(args.csv_path, index=False, header=False)
    print(f"Saved CSV to {args.csv_path}")

# -------------------------
# CLI
# -------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train','predict'], required=True)
    ap.add_argument('--train_dir', type=str, default=None)
    ap.add_argument('--val_dir', type=str, default=None)
    ap.add_argument('--test_dir', type=str, default=None)
    ap.add_argument('--outdir', type=str, default='runs/exp')

    ap.add_argument('--arch', type=str, default='convnext_base', choices=['convnext_tiny','convnext_small','convnext_base'])
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)

    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_epochs', type=int, default=5)
    ap.add_argument('--amp', action='store_true')

    # WS-DAN
    ap.add_argument('--use_wsdan', action='store_true')
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--wsdan_warm', type=int, default=10)
    ap.add_argument('--erase_p', type=float, default=0.5)

    # Losses / noise-robust
    ap.add_argument('--loss', type=str, default='elrplus', choices=['ce','elrplus','sce','gce'])
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--elr_lambda', type=float, default=3.0)
    ap.add_argument('--sce_alpha', type=float, default=0.1)
    ap.add_argument('--sce_beta', type=float, default=1.0)
    ap.add_argument('--gce_q', type=float, default=0.7)
    ap.add_argument('--elr_mem', type=int, default=0, help='ELR+ memory size; 0 => len(train_set)')

    # Curriculum
    ap.add_argument('--curriculum_start', type=int, default=15)
    ap.add_argument('--curriculum_epochs', type=int, default=50)
    ap.add_argument('--keep_ratio_start', type=float, default=0.9)
    ap.add_argument('--keep_ratio_final', type=float, default=0.7)

    # Regularization & randaug/mix
    ap.add_argument('--mixup', type=float, default=0.2)
    ap.add_argument('--cutmix', type=float, default=0.3)
    ap.add_argument('--randaug_N', type=int, default=2)
    ap.add_argument('--randaug_M', type=int, default=10)

    # ArcFace
    ap.add_argument('--arcface', action='store_true')
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--scale', type=float, default=30.0)

    # Sampler
    ap.add_argument('--class_balanced', action='store_true', help='Use class-balanced sampling (sqrt freq)')

    # Pretrain & freezing & LR mult
    ap.add_argument('--pretrained', action='store_true', help='Use timm pretrained IN1k weights')
    ap.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Freeze backbone for N epochs')
    ap.add_argument('--backbone_lr_mult', type=float, default=0.1, help='Backbone lr multiplier vs head')

    # Memory / compute tweaks
    ap.add_argument('--channels_last', dest='channels_last', action='store_true')
    ap.add_argument('--no-channels_last', dest='channels_last', action='store_false')
    ap.set_defaults(channels_last=True)

    ap.add_argument('--grad_ckpt', dest='grad_ckpt', action='store_true')
    ap.add_argument('--no-grad_ckpt', dest='grad_ckpt', action='store_false')
    ap.set_defaults(grad_ckpt=True)

    ap.add_argument('--microbatch', type=int, default=16)

    # === NEW: continue training from a checkpoint (weights only) ===
    ap.add_argument('--init_from', type=str, default=None,
                    help='Load model weights from a checkpoint (weights only) to continue training.')

    # Predict
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--csv_path', type=str, default='pred_results.csv')
    ap.add_argument('--classes', type=str, default=None)

    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        assert args.train_dir and os.path.isdir(args.train_dir), 'train_dir not found'
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.outdir, 'args.json'),'w') as f:
            json.dump(vars(args), f, indent=2)
        train(args)
    elif args.mode == 'predict':
        assert args.test_dir and os.path.isdir(args.test_dir), 'test_dir not found'
        assert args.checkpoint and os.path.isfile(args.checkpoint), 'checkpoint not found'
        predict(args)
    else:
        raise ValueError('Unknown mode')
