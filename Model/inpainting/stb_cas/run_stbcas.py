import os
import torch
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


#########################################
# Load Stable Cascade Prior + Decoder
#########################################
prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    variant="bf16",
    torch_dtype=torch.float16,
)

prior.enable_model_cpu_offload()
decoder.enable_model_cpu_offload()


#########################################
# Inpaint a single image (no prompt)
#########################################
def stable_cascade_inpaint_no_prompt(image_path, mask_path, height=1024, width=1024):

    # Load image + mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask_np = np.array(mask)
    mask_np = (mask_np > 128).astype(np.float32)

    #########################################
    # (1) PRIOR → embeddings (latent 24×24)
    #########################################
    prior_out = prior(
        prompt="",  
        height=height,
        width=width,
        negative_prompt="",
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=20
    )

    embeddings = prior_out.image_embeddings.to(torch.float16)  # (1, 24, 24, 24)

    #########################################
    # (2) Downsample mask to latent size
    #########################################
    latent_h, latent_w = embeddings.shape[-2], embeddings.shape[-1]

    mask_small = cv2.resize(mask_np, (latent_w, latent_h))   # 24×24
    mask_small = torch.tensor(mask_small, device=embeddings.device, dtype=torch.float16)
    mask_small = mask_small.unsqueeze(0).unsqueeze(0)  # (1,1,24,24)

    #########################################
    # (3) Apply mask to embeddings
    #########################################
    masked_embeddings = embeddings * (1 - mask_small)

    #########################################
    # (4) Decode modified embeddings
    #########################################
    result = decoder(
        image_embeddings=masked_embeddings,
        prompt="",
        negative_prompt="",
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10
    ).images[0]

    return result



#########################################
# Batch processing
#########################################
def batch_inpaint(image_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # assume filenames match: 0001.png and 0001_mask.png
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")])

    for fname in tqdm(images, desc="Stable Cascade Inpainting"):

        img_path = os.path.join(image_dir, fname)
        
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print(f"❌ Missing mask for {fname}, skip")
            continue

        out_img = stable_cascade_inpaint_no_prompt(img_path, mask_path)
        out_img.save(os.path.join(out_dir, fname))

    print(f"\n✔ Done! Output saved to: {out_dir}")


#########################################
# Entry point
#########################################
if __name__ == "__main__":
    batch_inpaint(
        image_dir="/mnt/data/charles/MLLM-as-a-Judge/mini_data/inpainting/all_150/images",
        mask_dir="/mnt/data/charles/MLLM-as-a-Judge/mini_data/inpainting/all_150/masks",
        out_dir="/mnt/data/charles/MLLM-as-a-Judge/Model/inpainting/results/stb_cas"
    )
