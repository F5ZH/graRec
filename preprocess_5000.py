import os
import warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = 120_000_000       # disable DecompressionBombError

src_dir = "/root/autodl-tmp/project/data/WebiNat-5000/train"
dst_dir = "/root/autodl-tmp/project/data/WebiNat-5000/train_processed"

os.makedirs(dst_dir, exist_ok=True)

for root, dirs, files in os.walk(src_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            src_path = os.path.join(root, f)
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            dst_path = dst_path.rsplit('.', 1)[0] + '.jpg'
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    img = Image.open(src_path)
                    # 检查是否有 Truncated 警告
                    is_truncated = any("Truncated" in str(warn.message) for warn in w)
                    if is_truncated:
                        print(f"[WARNING] Truncated image: {src_path}")
                    # 检查是否有 DecompressionBombWarning 警告
                    is_decomp = any("DecompressionBombWarning" in str(warn.message) for warn in w)
                    if is_decomp:
                        print(f"[WARNING] DecompressionBomb image: {src_path}")
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    if img.mode == "RGBA":
                        bg = Image.new("RGB", img.size, (0, 0, 0))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    else:
                        img = img.convert("RGB")
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img.save(dst_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Failed: {src_path}, {e}")