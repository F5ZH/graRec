import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from argsloader import get_args
from models.mymodels import build_model
from utils import load_model
from dataset.dataprocess import get_val_transforms

def generate_submission(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = build_model(args)
    model = load_model(model, args.checkpoint_path, device)
    model.eval()
    print(f"Model loaded from {args.checkpoint_path}")

    # 获取图像预处理
    transform = get_val_transforms(args)

    # 获取测试集所有图像文件路径
    image_files = []
    for root, dirs, files in os.walk(args.test_data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images for prediction.")

    # 进行预测
    predictions = []
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Predicting"):
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

            # 预测
            output = model(image)
            _, predicted_class = torch.max(output, 1)
            predicted_label = f"{predicted_class.item():04d}"  # 格式化为四位数字

            # 获取相对路径的文件名
            relative_path = os.path.relpath(img_path, args.test_data_path)
            filename = os.path.basename(relative_path)

            predictions.append((filename, predicted_label))

    # 写入 CSV 文件
    with open(args.submission_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for filename, label in predictions:
            writer.writerow([filename, label])

    print(f"Submission file saved to {args.submission_file}")

if __name__ == '__main__':
    args = get_args()
    generate_submission(args)