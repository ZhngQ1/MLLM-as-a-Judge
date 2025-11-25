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


###########################################
# Load LPIPS
###########################################
lpips_fn = lpips.LPIPS(net='vgg').cuda()

def to_tensor(img):
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


###########################################
# Utilities
###########################################
def compute_lpips(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    A = Image.fromarray(a)
    B = Image.fromarray(b)
    return float(lpips_fn(to_tensor(A), to_tensor(B)).item())


def compute_ssim_gray(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float32)/255.0
    b = b.astype(np.float32)/255.0
    val, _ = ssim(a, b, full=True, data_range=1.0)
    return float(val)


###########################################
# Core metric (no mask)
###########################################
def eval_edit_nomask(original, edited):

    # --- Fix: resize edited to match original ---
    if original.shape[:2] != edited.shape[:2]:
        edited = cv2.resize(edited, (original.shape[1], original.shape[0]))

    return {
        "ssim": compute_ssim_gray(original, edited),
        "lpips": compute_lpips(original, edited)
    }


###########################################
# Evaluate model results
###########################################
def evaluate_model(model_dir, image_root, edit_sessions, output_json):

    results = {}

    for session_id, steps in tqdm(edit_sessions.items(), desc=f"Evaluating {model_dir}"):

        # Only evaluate first editing step
        first = steps[0]

        input_name = first["input"]
        output_name = first["output"]
        instruction = first.get("instruction", "")

        # original input image
        folder = os.path.join(image_root, session_id)
        input_path = os.path.join(folder, input_name)
        if not os.path.exists(input_path):
            print("Missing:", input_path)
            continue
        original = cv2.imread(input_path)

        # model's output location
        model_output_path = os.path.join(model_dir, session_id, output_name)
        if not os.path.exists(model_output_path):
            print("Model output missing:", model_output_path)
            continue

        edited = cv2.imread(model_output_path)

        metrics = eval_edit_nomask(original, edited)

        results[session_id] = {
            "input": input_name,
            "output": output_name,
            "instruction": instruction,
            "metrics": metrics
        }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✔ Saved metrics to {output_json}")


###########################################
# Run all models
###########################################
def eval_all_models(
    image_root,
    results_root,
    edit_session_path,
    output_root="editing_metrics"
):
    os.makedirs(output_root, exist_ok=True)

    edit_sessions = json.load(open(edit_session_path))

    models = [
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
    ]

    print("Found models:", models)

    for model in models:
        model_dir = os.path.join(results_root, model)
        out_json = os.path.join(output_root, f"{model}.json")

        evaluate_model(
            model_dir=model_dir,
            image_root=image_root,
            edit_sessions=edit_sessions,
            output_json=out_json
        )


###########################################
# Entry point
###########################################
if __name__ == "__main__":
    eval_all_models(
        image_root="/mnt/data/charles/MLLM-as-a-Judge/mini_data/editing/magicbrush/images",
        results_root="/mnt/data/charles/MLLM-as-a-Judge/Model/editing/results",
        edit_session_path="/mnt/data/charles/MLLM-as-a-Judge/mini_data/editing/magicbrush/edit_sessions.json",
        output_root="./editing"
    )
