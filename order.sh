#训练
#python train.py --pretrained \
#  --output_dir ./runs/WebFG-400-base \
#&& \
#tensorboard --logdir=runs/WebFG-400-base/logs \
#&& \
#预测
python predict.py --num_classes 400 \
  --checkpoint ./runs/WebFG-400-base/best_model.pth \
  --submission_file ./runs/WebFG-400-base/pred_reults_400.csv