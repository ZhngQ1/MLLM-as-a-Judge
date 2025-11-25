# orchestrator/run_all_models.py
import os
import argparse
import subprocess
from tqdm import tqdm

DATA_ROOT = "../../mini_data/inpainting/all_150"
IMG_DIR = f"{DATA_ROOT}/images"
MASK_DIR = f"{DATA_ROOT}/masks"

MODELS = {
    "sdxl":  ("sdxl",  "scripts/run_sdxl.py"),
    "lama":  ("lama_env",  "scripts/run_lama.py"),
    "repaint": ("repaint_env", "scripts/run_repaint.py"),
    "deepfill": ("deepfill_env", "scripts/run_deepfill.py"),
}

OUT_ROOT = "results/inpainting"
os.makedirs(OUT_ROOT, exist_ok=True)


def select_images(k=2):
    """Pick k cub_,  k celeba_, k places_ images from IMG_DIR."""
    files = sorted(os.listdir(IMG_DIR))

    def pick(prefix, k=2):
        candidates = [f for f in files if f.startswith(prefix)]
        return candidates[:k]

    selected = []
    selected += pick("cub", k)
    selected += pick("celeba", k)
    selected += pick("places", k)

    print("\nSelected images:", selected)
    return selected

def run_model(env, script, img, mask, out):
    cmd = f"conda run -n {env} python {script} --img {img} --mask {mask} --out {out}"
    subprocess.run(cmd, shell=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="which models to run: e.g. --models sdxl lama repaint"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.models == ["all"]:
        selected_models = MODELS.keys()
    else:
        for m in args.models:
            if m not in MODELS:
                raise ValueError(f"Unknown model: {m}")
        selected_models = args.models

    print("\nRunning models:", list(selected_models))

    # Only pick 3*k images (k per dataset)
    images = select_images(k=50)

    # Run each selected model
    for model_name in selected_models:
        env, script = MODELS[model_name]

        print(f"\n=== Running {model_name} ===")
        model_out = f"{OUT_ROOT}/{model_name}"
        os.makedirs(model_out, exist_ok=True)

        for name in tqdm(images, desc=f"{model_name}"):
            img_path = f"{IMG_DIR}/{name}"
            mask_path = f"{MASK_DIR}/{name}"
            out_path = f"{model_out}/{name}"

            run_model(env, script, img_path, mask_path, out_path)



if __name__ == "__main__":
    main()
