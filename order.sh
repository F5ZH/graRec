# python bayes_opt.py

python newcode.py --mode train \
  --train_dir ./data/WebFG-400/train \
  --arch convnextv2_large --pretrained \
  --img_size 384 --batch_size 32 \
  --epochs 40 --warmup_epochs 0 \
  --lr 0.0001 --weight_decay 0.05 \
  --loss elrplus --elr_lambda 3.0 \
  --use_wsdan --K 8 --erase_p 0.6 --wsdan_warm 5 \
  --mixup 0.2 --cutmix 0.3 --randaug_N 2 --randaug_M 10 \
  --arcface --margin 0.25 --scale 30 \
  --class_balanced --self_clean --clean_warmup 5 --clean_thresh 0.4 --clean_min_w 0.3 \
  --class_aware --consistency_lambda 1.0 \
  --amp --workers 8 \
  --outdir ./runs/convnextv2_large_webfg400_in1k \
#   && usr/bin/shutdown


# python wsdan_convnext_plus.py --mode predict \
#   --test_dir ./data/WebFG-400/test \
#   --checkpoint ./runs/convnextv2_large_webfg400_in1k/final.pt \
#   --csv_path ./pred_results_web400.csv \
#   --img_size 384 \
#   --batch_size 32 \
#   --workers 8 \
#   --tta_flip \
#   --amp \
#   --logit_adjust_tau 2.0