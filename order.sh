#训练
python train.py --pretrained &&
tensorboard --logdir=runs/WebFG-400/logs &&
#预测
python predict.py