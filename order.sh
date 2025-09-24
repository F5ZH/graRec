#训练
python train.py --pretrained \
  --output_dir ./runs/WebFG-400-base \
&& \
tensorboard --logdir=runs/WebFG-400-base/logs \
&& \
#预测
python predict.py