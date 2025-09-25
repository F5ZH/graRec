#训练
python train.py --pretrained \
  --use_balanced_sampler --resample_alpha 1.0 \
  --output_dir ./runs/WebFG-400-resample \
&& \
tensorboard --logdir=runs/WebFG-400-resample/logs \
&& \
#预测
python predict.py \
  --num_classes 400 \
  --checkpoint ./runs/WebFG-400-resample/best_model.pth \
  --submission_file ./runs/WebFG-400-resample/submission.csv 