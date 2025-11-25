import os
import cv2
import json
import numpy as np
from tqdm import tqdm

import lpips
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torch
from torchvision import transforms


###############################################################
# Load LPIPS
###############################################################
lpips_fn = lpips.LPIPS(net='vgg').cuda()

def to_tensor(img):
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


###############################################################
# Utility Functions
###############################################################
def compute_lpips(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    A = Image.fromarray(a)
    B = Image.fromarray(b)
    return float(lpips_fn(to_tensor(A), to_tensor(B)).item())


def compute_ssim_gray(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    val, _ = ssim(a, b, full=True, data_range=1.0)
    return float(val)


###############################################################
# Inpainting Metrics (mask-aware)
###############################################################
def eval_inpaint_metric(original, inpainted, mask):

    # Resize everything to match original
    if original.shape[:2] != inpainted.shape[:2]:
        inpainted = cv2.resize(inpainted, (original.shape[1], original.shape[0]))
    if original.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    mask = (mask > 128).astype(np.uint8)
    inv_mask = 1 - mask

    # Masked region
    orig_masked = original * mask[:, :, None]
    inpaint_masked = inpainted * mask[:, :, None]

    # Background region
    orig_bg = original * inv_mask[:, :, None]
    inpaint_bg = inpainted * inv_mask[:, :, None]

    # Metrics
    masked_lpips = compute_lpips(orig_masked, inpaint_masked)
    bg_lpips = compute_lpips(orig_bg, inpaint_bg)
    overall_lpips = compute_lpips(original, inpainted)

    masked_ssim = compute_ssim_gray(orig_masked, inpaint_masked)
    bg_ssim = compute_ssim_gray(orig_bg, inpaint_bg)

    # L1 / L2 on masked region
    l1 = float(np.mean(np.abs(orig_masked.astype(np.float32) - inpaint_masked.astype(np.float32))))
    l2 = float(np.mean((orig_masked.astype(np.float32) - inpaint_masked.astype(np.float32))**2))

    return {
        "masked_lpips": masked_lpips,
        "bg_lpips": bg_lpips,
        "overall_lpips": overall_lpips,
        "masked_ssim": masked_ssim,
        "bg_ssim": bg_ssim,
        "l1_mask": l1,
        "l2_mask": l2
    }


###############################################################
# Evaluate one model
###############################################################
def evaluate_model(model_dir, image_dir, mask_dir, output_json):

    results = {}

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")])

    for fname in tqdm(image_files, desc=f"Evaluating {model_dir}"):
        base = fname.split(".")[0]

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        pred_path = os.path.join(model_dir, fname)

        if not (os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(pred_path)):
            print("Skipping:", fname)
            continue

        orig = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        pred = cv2.imread(pred_path)

        metrics = eval_inpaint_metric(orig, pred, mask)
        results[fname] = metrics

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✔ Saved metrics to {output_json}")


###############################################################
# Evaluate all models
###############################################################
def eval_all_models(images_root, masks_root, results_root, output_root="inpaint_metrics"):
    os.makedirs(output_root, exist_ok=True)

    models = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])

    print("Found models:", models)

    for model in models:
        if model == "deepfill":
            continue

        model_dir = os.path.join(results_root, model)
        out_json = os.path.join(output_root, f"{model}.json")

        evaluate_model(
            model_dir=model_dir,
            image_dir=images_root,
            mask_dir=masks_root,
            output_json=out_json
        )


###############################################################
# Entry
###############################################################
if __name__ == "__main__":
    eval_all_models(
        images_root="/mnt/data/charles/MLLM-as-a-Judge/mini_data/inpainting/all_150/images",
        masks_root="/mnt/data/charles/MLLM-as-a-Judge/mini_data/inpainting/all_150/masks",
        results_root="/mnt/data/charles/MLLM-as-a-Judge/Model/inpainting/results",
        output_root="./inpainting"
    )
