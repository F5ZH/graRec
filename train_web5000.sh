#!/bin/bash
set -e  # 遇到报错就退出

# 日志目录
LOG_DIR=./logs
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/train_web5000_$(date +%Y%m%d_%H%M%S).log

echo "训练开始，日志输出到 $LOG_FILE"

# 阶段1: baseline 224
python wsdan_convnext_plus.py \
  --mode train \
  --train_dir ./data/WebiNat-5000/train \
  --outdir ./runs/convnextv2_tiny_web5000_e100_224 \
  --arch convnextv2_tiny --pretrained \
  --img_size 224 --batch_size 64 \
  --epochs 50 --warmup_epochs 3 \
  --lr 1e-4 --weight_decay 0.05 \
  --loss elrplus --elr_lambda 3 \
  --use_wsdan --K 8 --erase_p 0.6 --wsdan_warm 5 \
  --mixup 0.2 --cutmix 0.3 \
  --randaug_N 2 --randaug_M 10 \
  --arcface --margin 0.25 --scale 30 \
  --class_balanced \
  --self_clean --clean_warmup 5 --clean_thresh 0.5 \
  --amp --workers 12 \
  2>&1 | tee -a $LOG_FILE

# 阶段1b: fine-tune 384
python wsdan_convnext_plus.py \
  --mode train \
  --train_dir ./data/WebiNat-5000/train \
  --outdir ./runs/convnextv2_tiny_web5000_ft384 \
  --arch convnextv2_tiny --pretrained \
  --img_size 384 --batch_size 32 \
  --epochs 15 --warmup_epochs 1 \
  --lr 5e-5 --weight_decay 0.05 \
  --loss elrplus --elr_lambda 3 \
  --use_wsdan --K 8 --erase_p 0.6 --wsdan_warm 4 \
  --mixup 0.2 --cutmix 0.3 \
  --randaug_N 2 --randaug_M 10 \
  --arcface --margin 0.25 --scale 30 \
  --class_balanced \
  --self_clean --clean_warmup 5 --clean_thresh 0.5 \
  --amp --workers 12 \
  --init_from ./runs/convnextv2_tiny_web5000_e100_224/final.pt \
  2>&1 | tee -a $LOG_FILE

# 全部完成后自动关机
/usr/bin/shutdown
