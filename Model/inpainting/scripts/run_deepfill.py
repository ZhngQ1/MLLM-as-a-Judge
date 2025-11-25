# scripts/run_deepfill.py
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cmd = [
        "python", "deepfill/test.py",
        "--image", args.img,
        "--mask", args.mask,
        "--output", args.out
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
