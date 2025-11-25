import os
import json
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

def auto_strength(instruction):
    ins = instruction.lower()

    # 小幅编辑：颜色、轻微外观调整
    if any(k in ins for k in ["color", "recolor", "change color", "brighten", "darken"]):
        return 0.3

    # 局部添加物体
    if any(k in ins for k in ["add ", "put ", "place ", "insert "]):
        return 0.45

    # 改变服装、发型、表情
    if any(k in ins for k in ["wear", "dress", "shirt", "hair", "expression", "smile"]):
        return 0.4

    # 改变大型结构 / 风格
    if any(k in ins for k in ["style", "winter", "snow", "rain", "scene", "background"]):
        return 0.55

    # 强力编辑
    if any(k in ins for k in ["make the person", "turn the", "transform", "convert"]):
        return 0.65

    # 默认
    return 0.5

def run_edit(input_img, instruction, output_img):
    img = Image.open(input_img).convert("RGB")

    # 如果需要动态调整 strength，可以取消下面一行的注释，
    # 但这个function只是比较简单的关键词匹配，效果有限
    # 如果需要更复杂的理解，可以使用 CLIP 或 LLaVan 进行分析
    # strength = auto_strength(instruction)

    out = pipe(
        prompt=instruction,
        image=img,
        strength=0.4,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]

    out.save(output_img)
    print("Saved →", output_img)


# === JSON-based batch processing ===
session = json.load(open("../../../mini_data/editing/magicbrush/edit_sessions.json"))

img_root = "../../../mini_data/editing/magicbrush/images"
save_root = "../../../Model/editing/results/sdxl_i2i"

os.makedirs(save_root, exist_ok=True)

for sid, steps in session.items():
    step = steps[0]  # only the first step (one-shot editing)

    in_path = os.path.join(img_root, sid, step["input"])
    inst = step["instruction"]

    out_dir = os.path.join(save_root, sid)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{sid}-sdxl.png")

    run_edit(in_path, inst, out_path)
