import os
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

# =======================================
# Config
# =======================================
DEVICE = "cuda"
OUT_ROOT = "../../../Model/ctrl_gen/results"

TASKS = {
    "canny": {
        "data_dir": "../../../mini_data/ctrl_gen/canny",
        "model": "lllyasviel/control_v11p_sd15_canny",
        "caption_file": "../../../mini_data/ctrl_gen/canny/captions.json"
    },
    "depth": {
        "data_dir": "../../../mini_data/ctrl_gen/depth",
        "model": "lllyasviel/control_v11f1p_sd15_depth",
        "caption_file": "../../../mini_data/ctrl_gen/depth/captions.json"
    },
    "pose": {
        "data_dir": "../../../mini_data/ctrl_gen/pose",
        "model": "lllyasviel/control_v11p_sd15_openpose",
        "caption_file": "../../../mini_data/ctrl_gen/pose/captions.json"
    }
}

# =======================================
# Load base input image
# =======================================
def load_input_image(path):
    return Image.open(path).convert("RGB").resize((512, 512))


# =======================================
# Adaptive Canny (works for ALL scenes)
# =======================================
def load_canny_condition(original_image_path):
    img = cv2.imread(original_image_path)
    img = cv2.resize(img, (512, 512))

    # 1. texture complexity estimation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    # 2. adaptive thresholds
    if variance < 50:
        low, high = 80, 150        # smooth scenes
    elif variance < 150:
        low, high = 100, 200       # medium complexity
    else:
        low, high = 120, 240       # high complexity

    edges = cv2.Canny(img, low, high)

    # 3. adaptive dilation based on density
    edge_density = edges.mean() / 255.0

    if edge_density < 0.10:
        edges = cv2.dilate(edges, None, iterations=1)
    elif edge_density < 0.25:
        edges = cv2.dilate(edges, None, iterations=1)
    else:
        pass  # no dilation for dense scenes

    edges = np.stack([edges] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(edges)

def generate_controlnet_canny(image_path):
    """
    根据原图生成适用于 ControlNet 的 Canny 边缘图。
    该版本在所有类别上（人像 / 动物 / 风景 / 建筑）表现稳定。
    """


    # Load image
    img = cv2.imread(image_path)

    # Resize to SD resolution
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Convert to gray for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----- 1. 图像复杂度检测 -----
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    # ----- 2. 自适应 Canny 阈值 -----
    if variance < 40:          # 场景简单（人像、天空、大面积背景）
        low, high = 70, 140
    elif variance < 120:       # 常见场景（动物、草地、正常照片）
        low, high = 100, 200
    else:                      # 纹理复杂（建筑、森林、密集线条）
        low, high = 130, 260

    edges = cv2.Canny(img, low, high)

    # ----- 3. 边缘稀疏程度检测 -----
    density = edges.mean() / 255.0

    # 稀疏边缘适当加粗（增强结构）
    if density < 0.10:
        edges = cv2.dilate(edges, None, iterations=1)
    elif density < 0.25:
        edges = cv2.dilate(edges, None, iterations=1)
    # 否则太密不做处理

    # ----- 4. 转为 ControlNet 需要的 3 通道格式 -----
    edges_3 = np.stack([edges] * 3, axis=-1).astype(np.uint8)

    return Image.fromarray(edges_3)


# =======================================
# Depth loader  — KEEP EXACTLY AS BEFORE
# =======================================
def load_depth_condition(path):
    cond = Image.open(path).convert("L")
    cond = cond.resize((512, 512), Image.BILINEAR)
    cond = np.expand_dims(np.array(cond), -1).repeat(3, axis=-1)
    return Image.fromarray(cond)


# =======================================
# Pose loader — KEEP EXACTLY AS BEFORE
# =======================================
def load_pose_condition(path):
    cond = Image.open(path)
    return cond.resize((512, 512), Image.NEAREST)


# =======================================
# Dispatch loader
# =======================================
def load_condition(task, cond_path, img_path):
    if task == "canny":
        return generate_controlnet_canny(img_path)
    elif task == "depth":
        return load_depth_condition(cond_path)
    elif task == "pose":
        return load_pose_condition(cond_path)
    else:
        raise ValueError(f"Unknown task: {task}")


# =======================================
# Execute per-task pipeline
# =======================================
def run_task(task_name, task_cfg):
    print(f"\n=========== Running {task_name.upper()} ===========")

    image_dir = os.path.join(task_cfg["data_dir"], "images")
    cond_dir = os.path.join(task_cfg["data_dir"], "conditions")

    out_dir = os.path.join(OUT_ROOT, task_name, "sd15")
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(glob(os.path.join(image_dir, "*")))
    conds  = sorted(glob(os.path.join(cond_dir, "*")))

    assert len(images) == len(conds)

    # Load captions.json
    with open(task_cfg["caption_file"], "r") as f:
        captions = json.load(f)

    # Load models
    print(f"Loading ControlNet: {task_cfg['model']}")
    controlnet = ControlNetModel.from_pretrained(task_cfg["model"], torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    for img_path, cond_path in tqdm(zip(images, conds), total=len(images)):
        fname = os.path.basename(img_path)

        # Load corresponding prompt
        if fname not in captions:
            prompt = "a high-quality photograph"
        else:
            prompt = captions[fname]

        print(f"\nProcessing {fname} with prompt: {prompt}")

        # Input image & condition
        image = load_input_image(img_path)
        condition = load_condition(task_name, cond_path, img_path)

        # Control strength
        control_scale = 0.6 if task_name == "canny" else 1.0

        # Run SD + ControlNet
        out = pipe(
            prompt=prompt,
            image=image,
            control_image=condition,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=control_scale,
        ).images[0]

        save_path = os.path.join(out_dir, fname.replace(".jpg", "_sd15.png"))
        out.save(save_path)

    print(f"Saved results at: {out_dir}")


# =======================================
# Main
# =======================================
if __name__ == "__main__":
    for task_name, cfg in TASKS.items():
        run_task(task_name, cfg)

    print("\n====== All SD15 ControlNet tasks DONE ✓ ======")
