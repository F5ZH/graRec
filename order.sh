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
# python train_dual_branch_swin.py --mode train \
#   --train_dir data/WebFG-400/train \
#   --outdir runs/dual_branch_v2 \
#   --arch swin_base \
#   --epochs 20 \
#   --img_size 384 \
#   --batch_size 64 \
#   --loss elrplus --elr_lambda 2.0 \
#   --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
#   --use_wsdan --wsdan_warm 3\
#   --arcface --margin 0.15 --scale 20 \
#   --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
#   --workers 8 --amp \
#   --lambda_global 1.0 \
#   --lambda_local 0.5 \
#   --local_crop_size 224 \
#   --lr 5e-4 \
#   --grad_ckpt --microbatch 16; /usr/bin/shutdown

# 预测
# python train_dual_branch_swin.py --mode predict \
#   --test_dir data/WebFG-400/test \
#   --outdir runs/dual_branch_v2 \
#   --arch swin_base \
#   --img_size 384 \
#   --batch_size 64 \
#   --use_wsdan \
#   --arcface --margin 0.15 --scale 20 \
#   --lambda_global 1.0 \
#   --lambda_local 0.5 \
#   --local_crop_size 224 \
#   --workers 8 \
#   --checkpoint runs/dual_branch_v2/best_fold0.pt \
#   --csv_path submission.csv


python heteroDualBranch.py --mode train \
    --train_dir data/WebFG-400/train \
    --outdir runs/my_experiment \
    --swin_arch swin_base \
    --convnext_arch convnext_base \
    --img_size 384 \
    --local_crop_size 224 \
    --use_wsdan \
    --K 8 \
    --arcface \
    --margin 0.25 \
    --scale 30.0 \
    --batch_size 64 \
    --epochs 100 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --randaug_N 2 \
    --randaug_M 9 \
    --mixup 0.2 \
    --cutmix 0.3 \
    --loss elrplus \
    --label_smooth 0.1 \
    --class_balanced \
    --amp \
    --channels_last \
    --grad_ckpt \
    --microbatch 16 \
    --k_folds 10 \
    --workers 8 \
    --seed 42                           # 随机种子