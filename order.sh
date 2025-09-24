#训练
python train.py --pretrained \
  --output_dir ./runs/WebFG-400-reweight \
&& \
tensorboard --logdir=runs/WebFG-400-reweight/logs \
&& \
#预测
python predict.py