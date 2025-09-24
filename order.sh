#训练
python train.py --pretrained \
  --output_dir ./runs/WebFG-400-v1
&&
tensorboard --logdir=runs/WebFG-400-v1/logs 
&&
#预测
python predict.py