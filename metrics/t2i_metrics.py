import os
import json
import torch
import clip
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


#############################################
# Load models only once
#############################################

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# LPIPS
lpips_fn = lpips.LPIPS(net='vgg').to(device)

# Aesthetic predictor
aes_extractor = AutoFeatureExtractor.from_pretrained("cafeai/cafe_aesthetic")
aes_model = AutoModelForImageClassification.from_pretrained("cafeai/cafe_aesthetic").to(device)


#############################################
# Utility functions
#############################################

def load_image(path):
    return Image.open(path).convert("RGB")


def compute_clip_score(image, prompt):
    img = clip_preprocess(image).unsqueeze(0).to(device)
    txt = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img)
        txt_feat = clip_model.encode_text(txt)

    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    return float((img_feat @ txt_feat.T).item())


def compute_lpips_baseline(image):
    blank = Image.new("RGB", image.size)
    return float(lpips_fn(lpips.im2tensor(np.array(image)).to(device),
                          lpips.im2tensor(np.array(blank)).to(device)).item())


def compute_aesthetic_score(image):
    inputs = aes_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = aes_model(**inputs).logits

    # logits shape: [1,2]  → [low, high]
    probs = torch.softmax(logits, dim=1)

    # return "high aesthetic" probability
    return float(probs[0, 1].item())



#############################################
# Evaluate a SINGLE MODEL folder
#############################################

def evaluate_single_model(model_dir, prompts_json, output_json):
    prompts = json.load(open(prompts_json, "r"))

    results = {}

    print(f"\n=== Evaluating model: {model_dir} ===")

    for fname, prompt in tqdm(prompts.items(), desc=f"{model_dir}"):
        png_name = fname.replace(".jpg", ".png")
        img_path = os.path.join(model_dir, png_name)

        if not os.path.exists(img_path):
            continue

        image = load_image(img_path)

        clipscore = compute_clip_score(image, prompt)
        aesthetics = compute_aesthetic_score(image)
        lp = compute_lpips_baseline(image)

        results[png_name] = {
            "prompt": prompt,
            "clipscore": clipscore,
            "aesthetic_score": aesthetics,
            "lpips_vs_blank": lp,
        }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✔ Saved metrics to {output_json}")


#############################################
# Evaluate ALL MODELS in a parent folder
#############################################

def evaluate_all_models(parent_dir, prompts_json, out_dir="metrics"):
    os.makedirs(out_dir, exist_ok=True)

    models = sorted([
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])

    print(f"\nFound models: {models}")

    for model_name in models:
        model_dir = os.path.join(parent_dir, model_name)
        output_json = os.path.join(out_dir, f"{model_name}.json")

        evaluate_single_model(
            model_dir=model_dir,
            prompts_json=prompts_json,
            output_json=output_json,
        )


#############################################
# Script entry point
#############################################

if __name__ == "__main__":
    evaluate_all_models(
        parent_dir="/mnt/data/charles/MLLM-as-a-Judge/Model/t2i/results",  
        prompts_json="/mnt/data/charles/MLLM-as-a-Judge/mini_data/t2i/selected_100_with_captions.json",
        out_dir="./t2i"
    )
