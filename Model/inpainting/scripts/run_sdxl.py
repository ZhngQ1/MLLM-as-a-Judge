# scripts/run_sdxl.py
import argparse
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionXLInpaintPipeline

def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask) // 255
    mask = Image.fromarray((mask*255).astype(np.uint8))
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16
    ).to("cuda")

    image = Image.open(args.img).convert("RGB")
    mask = load_mask(args.mask)

    res = pipe(prompt="", image=image, mask_image=mask).images[0]
    res.save(args.out)

if __name__ == "__main__":
    main()
