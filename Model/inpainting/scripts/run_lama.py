import torch
import numpy as np
from PIL import Image
import yaml
import sys
import os

# --- Fix Python import path for LaMa ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "lama"))
sys.path.append(LAMA_DIR)

print("Current dir:", CURRENT_DIR)
print("LaMa dir:", LAMA_DIR)

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device


def load_lama_model(device="cuda"):
    # Load LaMa train config
    config_path = os.path.join(LAMA_DIR, "big-lama", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Train config not found: {config_path}")

    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    ckpt_path = os.path.join(LAMA_DIR, "big-lama", "models", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Your version: load_checkpoint(train_config, path)
    model = load_checkpoint(
        train_config=train_config,
        path=ckpt_path,
        map_location=device,
        strict=False
    )

    model = model.to(device).eval()
    return model


def preprocess(img, mask):
    img = np.array(img.convert("RGB"))
    mask = np.array(mask.convert("L"))
    mask = (mask > 0).astype(np.uint8)

    img_t = torch.tensor(img.transpose(2, 0, 1)).float() / 255.
    mask_t = torch.tensor(mask).unsqueeze(0).float()

    return img_t.unsqueeze(0), mask_t.unsqueeze(0)


def inpaint(model, img, mask, device="cuda"):
    batch = {"image": img.to(device), "mask": mask.to(device)}
    batch = move_to_device(batch, device)

    with torch.no_grad():
        out = model(batch)["inpainted"][0]

    out = out.permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    img = Image.open(args.img)
    mask = Image.open(args.mask)

    model = load_lama_model(device)
    img_t, mask_t = preprocess(img, mask)
    result = inpaint(model, img_t, mask_t, device)

    result.save(args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
