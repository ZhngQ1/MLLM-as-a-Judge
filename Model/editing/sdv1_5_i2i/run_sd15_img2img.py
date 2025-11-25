from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os
import json

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def edit_image(input_img_path, instruction, output_img_path,
               strength=0.55, guidance_scale=7.5, steps=40):

    init_image = Image.open(input_img_path).convert("RGB")

    out = pipe(
        prompt=instruction,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]

    out.save(output_img_path)
    print("Saved →", output_img_path)

# your json file
session = json.load(open("../../../mini_data/editing/magicbrush/edit_sessions.json"))

base_dir = "../../../mini_data/editing/magicbrush/images"
save_dir = "sd15_edit_results"
os.makedirs(save_dir, exist_ok=True)

for sid, steps in session.items():
    print("Processing:", sid)

    sdir = os.path.join(base_dir, sid)
    odir = os.path.join(save_dir, sid)
    os.makedirs(odir, exist_ok=True)

    step = steps[0]    # only use first editing
    input_img = os.path.join(sdir, step["input"])
    instruction = step["instruction"]

    output_path = os.path.join(odir, f"{sid}-sd15.png")

    edit_image(input_img, instruction, output_path)