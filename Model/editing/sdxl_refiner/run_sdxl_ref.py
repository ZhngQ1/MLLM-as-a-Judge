import os
import json
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import DiffusionPipeline
from PIL import Image
import torch

device = "cuda"

# SDXL Base
pipe_base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# SDXL Refiner
pipe_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)


def sdxl_edit(input_path, instruction, output_path,
              strength=0.4, guidance_scale=7.5,
              base_steps=40, refiner_steps=20):

    init_image = Image.open(input_path).convert("RGB")

    # 1) Stage 1: SDXL Base
    base_result = pipe_base(
        prompt=instruction,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=base_steps,
        output_type="latent"  # VERY IMPORTANT → give to Refiner
    )

    latent = base_result.images[0]

    # 2) Stage 2: SDXL Refiner
    refined = pipe_refiner(
        prompt=instruction,
        image=latent,
        num_inference_steps=refiner_steps
    ).images[0]

    refined.save(output_path)
    print("Saved →", output_path)


session = json.load(open("../../../mini_data/editing/magicbrush/edit_sessions.json"))
img_root = "../../../mini_data/editing/magicbrush/images"
save_root = "../../../Model/editing/results/sdxl_refiner"
os.makedirs(save_root, exist_ok=True)

for sid, steps in session.items():
    print("Processing:", sid)

    step = steps[0]  # one-shot edit
    instruction = step["instruction"]
    input_path = os.path.join(img_root, sid, step["input"])

    out_dir = os.path.join(save_root, sid)
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"{sid}-sdxl-refiner.png")

    sdxl_edit(
        input_path=input_path,
        instruction=instruction,
        output_path=output_path,
        strength=0.4  # can adjust dynamically
    )
