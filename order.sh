#   400训练测试 无mix增强 无Curriculum 默认启用预训练
#   python wsdan_convnext_elr_single.py \
#   --mode train \
#   --train_dir /root/autodl-tmp/project/data/WebFG-400/train \
#   --arch convnext_base \
#   --epochs 10 --img_size 256 --batch_size 64 \
#   --loss elrplus --elr_lambda 2.0 \
#   --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
#   --use_wsdan --K 4 --erase_p 0.3 --wsdan_warm 2 \
#   --arcface --margin 0.15 --scale 20 \
#   --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
#   --workers 8 --amp \
#   --channels_last --grad_ckpt --microbatch 16 \
#   --outdir /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt


#   继续训练
#   python wsdan_convnext_elr_single.py \
#   --mode train \
#   --train_dir /root/autodl-tmp/project/data/WebFG-400/train \
#   --arch convnext_base \
#   --epochs 60 --img_size 256 --batch_size 64 \
#   --loss elrplus --elr_lambda 2.0 \
#   --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
#   --use_wsdan --K 4 --erase_p 0.3 --wsdan_warm 2 \
#   --arcface --margin 0.15 --scale 20 \
#   --workers 8 --amp \
#   --channels_last --grad_ckpt --microbatch 16 \
#   --outdir /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt \
#   --init_from /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt/final.pt


#   400预测
  # python wsdan_convnext_elr_single.py \
  # --mode predict \
  # --test_dir /root/autodl-tmp/project/data/WebFG-400/test \
  # --checkpoint /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt/final.pt \
  # --csv_path /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt/predictions.csv 


#   5000训练测试 无mix增强 100轮 默认启用预训练
  # python wsdan_convnext_elr_single.py \
  # --mode train \
  # --train_dir /root/autodl-tmp/project/data/WebiNat-5000/train \
  # --arch convnext_base \
  # --epochs 100 --img_size 256 --batch_size 64 \
  # --loss elrplus --elr_lambda 2.0 \
  # --curriculum_epochs 90 \
  # --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
  # --use_wsdan --K 4 --erase_p 0.3 --wsdan_warm 2 \
  # --arcface --margin 0.15 --scale 20 \
  # --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
  # --workers 8 --amp \
  # --channels_last --grad_ckpt --microbatch 16 \
  # --outdir /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt_5000

#   5000训练，耗时更短的方法 
  python wsdan_convnext_elr_single.py \
  --mode train \
  --train_dir /root/autodl-tmp/project/data/WebiNat-5000/train_processed \
  --arch convnext_base \
  --epochs 100 --img_size 256 --batch_size 128 \
  --loss elrplus --elr_lambda 2.0 \
  --curriculum_epochs 90 \
  --mixup 0 --cutmix 0 --randaug_N 2 --randaug_M 10 \
  --use_wsdan --K 4 --erase_p 0.3 --wsdan_warm 2 \
  --arcface --margin 0.15 --scale 20 \
  --pretrained --freeze_backbone_epochs 1 --backbone_lr_mult 0.1 \
  --workers 10 --amp \
  --channels_last --grad_ckpt --microbatch 16 \
  --outdir /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt_5000

#    5000预测
  # python wsdan_convnext_elr_single.py \
  # --mode predict \
  # --test_dir /root/autodl-tmp/project/data/WebiNat-5000/test \
  # --checkpoint /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt_5000/last.pt \
  # --csv_path /root/autodl-tmp/project/runs/main_elr_arc_micro16_pt_5000/pred_results_web5000.csv