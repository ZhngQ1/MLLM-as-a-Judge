import os
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
DEVICE = "cuda"
OUT_ROOT = "../../../Model/ctrl_gen/results"
os.makedirs(OUT_ROOT, exist_ok=True)

PROMPT = "a high-quality realistic photograph"

TASKS = {
    # "canny": {
    #     "controlnet": "diffusers/controlnet-canny-sdxl-1.0",
    #     "src": "../../../mini_data/ctrl_gen/canny",
    # },
    "depth": {
        "controlnet": "diffusers/controlnet-depth-sdxl-1.0",
        "src": "../../../mini_data/ctrl_gen/depth",
    },
}

# ---------------------------
# Helper
# ---------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def run_task(task_name, cfg):
    print(f"\n========== Running {task_name.upper()} ==========")

    # Prepare paths
    src_dir = cfg["src"]
    cn_name = cfg["controlnet"]

    out_dir = os.path.join(OUT_ROOT, task_name, "sdxl")
    os.makedirs(out_dir, exist_ok=True)

    # Load ControlNet
    print(f"Loading ControlNet: {cn_name}")
    controlnet = ControlNetModel.from_pretrained(
        cn_name,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)

    # pipe.enable_model_cpu_offload()

    imgs = sorted([
        f for f in os.listdir(os.path.join(src_dir, "conditions"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    for img_name in tqdm(imgs):
        in_path = os.path.join(src_dir, "conditions", img_name)
        out_path = os.path.join(out_dir, img_name.replace(".jpg", ".png"))

        try:
            ctrl_img = load_image(in_path)

            result = pipe(
                prompt=PROMPT,
                image=ctrl_img,
                num_inference_steps=30,
                controlnet_conditioning_scale=1.0,
            ).images[0]

            result.save(out_path)
        except Exception as e:
            print(f"❌ Error on {img_name}: {e}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    for name, cfg in TASKS.items():
        run_task(name, cfg)

    print("\n Done! Results saved to:", OUT_ROOT)