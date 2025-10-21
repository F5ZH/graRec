import os
import warnings
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 设置最大像素限制
Image.MAX_IMAGE_PIXELS = 120_000_000

src_dir = "/root/autodl-tmp/project/data/WebFG-400/train"
dst_dir = "/root/autodl-tmp/project/data/WebFG-400/train_processed"

# 支持的扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

def process_single_image(args):
    src_path, dst_path = args
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  

            img = Image.open(src_path)
            # 统一转为 RGB
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert('RGB')
            
            # Resize 并保存
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img.save(dst_path, 'JPEG', quality=95)
            
        return True, None
    except Exception as e:
        return False, f"{src_path}: {e}"

def collect_image_paths():
    """收集所有待处理图像路径"""
    tasks = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, src_dir)
                dst_path = os.path.join(dst_dir, rel_path)
                dst_path = dst_path.rsplit('.', 1)[0] + '.jpg'
                tasks.append((src_path, dst_path))
    return tasks

if __name__ == '__main__':
    
    # 创建目标目录结构
    os.makedirs(dst_dir, exist_ok=True)
    
    # 收集任务
    tasks = collect_image_paths()
    total = len(tasks)
    print(f"Found {total} images. Starting preprocessing...")

    # 使用所有可用 CPU 核心（或指定数量）
    num_workers = min(8, mp.cpu_count())  # 限制最大 worker 数
    failed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if not success:
                print(f"❌ {msg}")
                failed.append(msg)
            if i % 100 == 0 or i == total:
                print(f"Processed {i}/{total}, Failed: {len(failed)}")

    print(f"✅ Preprocessing completed. Success: {total - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailed files:")
        for f in failed[:20]:
            print(f)