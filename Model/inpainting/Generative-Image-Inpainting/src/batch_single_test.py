import os
import subprocess
from glob import glob

PRETRAIN = "../pretrained/Places2_G1000000.pt"
IMG_DIR = "../../../../mini_data/inpainting/all_150/images"
MASK_DIR = "../../../../mini_data/inpainting/all_150/masks"
OUT_DIR = "../../results/scat"

os.makedirs(OUT_DIR, exist_ok=True)

img_list = sorted(glob(os.path.join(IMG_DIR, "*")))
mask_list = sorted(glob(os.path.join(MASK_DIR, "*")))

assert len(img_list) == len(mask_list), "Image 与 Mask 数量不一致！"

for img, mask in zip(img_list, mask_list):
    print(f"Processing {os.path.basename(img)}")

    cmd = [
        "python", "single_test.py",
        "--pre_train", PRETRAIN,
        "--ipath", img,
        "--mpath", mask,
        "--outputs", OUT_DIR
    ]

    subprocess.run(cmd)

print("All done!")
