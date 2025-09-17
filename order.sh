# 400ConvNeXt训练，课程学习推迟到20轮往后 抗长尾分布未启动
# python train_dual_branch_ConvNeXt.py --mode train \
#   --train_dir data/WebFG-400/train \
#   --outdir runs/dual_branch_v1 \
#   --arch convnext_base \
#   --epochs 20 \
#   --img_size 384 \
#   --batch_size 64 \
#   --loss elrplus --elr_lambda 2.0 \
#   --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
#   --use_wsdan \
#   --arcface --margin 0.15 --scale 20 \
#   --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
#   --workers 8 --amp \
#   --lambda_global 1.0 \
#   --lambda_local 0.5 \
#   --local_crop_size 224 \
#   --lr 5e-4 \
#   --channels_last --grad_ckpt --microbatch 16

# 400Swin训练，课程学习推迟到20轮往后 抗长尾分布未启动
python train_dual_branch_swin.py --mode train \
  --train_dir data/WebFG-400/train \
  --outdir runs/dual_branch_v2 \
  --arch swin_base \
  --epochs 20 \
  --img_size 384 \
  --batch_size 64 \
  --loss elrplus --elr_lambda 2.0 \
  --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
  --use_wsdan \
  --arcface --margin 0.15 --scale 20 \
  --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
  --workers 8 --amp \
  --lambda_global 1.0 \
  --lambda_local 0.5 \
  --local_crop_size 224 \
  --lr 5e-4 \
  --grad_ckpt --microbatch 16; /usr/bin/shutdown

# 预测
# python train_dual_branch.py --mode predict \
#   --test_dir data/WebFG-400/test \
#   --outdir runs/dual_branch_v1 \
#   --arch convnext_base \
#   --img_size 384 \
#   --batch_size 64 \
#   --use_wsdan \
#   --arcface --margin 0.15 --scale 20 \
#   --lambda_global 1.0 \
#   --lambda_local 0.5 \
#   --local_crop_size 224 \
#   --workers 8 \
#   --checkpoint runs/dual_branch_v1/final.pt \
#   --csv_path submission.csv
