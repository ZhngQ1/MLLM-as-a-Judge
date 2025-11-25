import cv2
import os
import json
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

#####################################
# LPIPS load
#####################################
lpips_fn = lpips.LPIPS(net="vgg").cuda()

def to_tensor(img):
    return transforms.ToTensor()(img).unsqueeze(0).cuda()

#####################################
# SSIM
#####################################
def compute_ssim(a, b):
    # Convert to grayscale if needed
    if len(a.shape) == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if len(b.shape) == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # Ensure same size
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    # Convert to float32 in [0,1]
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0

    # ----- ❗ FIX: specify data_range=1.0 -----
    s, _ = ssim(a, b, full=True, data_range=1.0)
    return s



#####################################
# LPIPS
#####################################
def compute_lpips(a, b):
    a = Image.fromarray(a)
    b = Image.fromarray(b)
    return lpips_fn(to_tensor(a), to_tensor(b)).item()

#####################################
# 1. Canny Consistency
#####################################
def eval_canny(gt_edge_img, gen_img):

    # ensure gt is grayscale
    if len(gt_edge_img.shape) == 3:
        gt_edge_img = cv2.cvtColor(gt_edge_img, cv2.COLOR_BGR2GRAY)

    # get gen edges
    gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    gen_edge = cv2.Canny(gen_gray, 100, 200)

    # ---- resize to match ----
    if gt_edge_img.shape != gen_edge.shape:
        gen_edge = cv2.resize(gen_edge, (gt_edge_img.shape[1], gt_edge_img.shape[0]))

    # ---- SSIM (guaranteed same H,W and both grayscale) ----
    ssim_val = compute_ssim(gt_edge_img, gen_edge)

    # ---- LPIPS: requires 3 channels ----
    e1 = cv2.cvtColor(gt_edge_img, cv2.COLOR_GRAY2BGR)
    e2 = cv2.cvtColor(gen_edge, cv2.COLOR_GRAY2BGR)

    # resize for LPIPS if needed
    if e1.shape != e2.shape:
        e2 = cv2.resize(e2, (e1.shape[1], e1.shape[0]))

    lp = compute_lpips(e1, e2)

    return {
        "canny_ssim": ssim_val,
        "canny_lpips": lp
    }

#####################################
# 2. Depth Consistency (MiDaS)
#####################################
def load_midas():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval().cuda()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

midas_model, midas_tf = load_midas()

def predict_depth(img_bgr):
    # Convert BGR to RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to MiDaS expected size
    img = cv2.resize(img, (256, 256))

    # HWC -> CHW
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # (3,256,256)

    # Add batch dim
    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()  # (1,3,256,256)

    with torch.no_grad():
        pred = midas_model(img_tensor)  # (1,1,256,256) or (1,256,256)

    # Remove batch dim
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    depth = pred.squeeze().cpu().numpy()

    # depth may be 2D or 1D if shape wrong
    if depth.ndim != 2:
        print("WARNING: MiDaS returned wrong shape:", depth.shape)
        return None

    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth

def load_gt_depth(path):
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if gt is None:
        return None

    # If 16-bit depth
    if gt.dtype == np.uint16:
        gt = gt.astype(np.float32)
        # typical NYU depth range: up to ~10 meters
        gt = gt / gt.max()

    # If already 8-bit grayscale
    elif gt.dtype == np.uint8:
        gt = gt.astype(np.float32) / 255.0

    return gt

def silog(gt, pred):
    """
    Scale-invariant log error (SILog), widely used in depth estimation papers.
    """
    eps = 1e-6
    mask = (gt > eps)

    gt_v = gt[mask]
    pr_v = pred[mask]

    if len(gt_v) < 10:
        return None

    d = np.log(pr_v + eps) - np.log(gt_v + eps)
    return float(np.sqrt((d**2).mean() - (d.mean()**2)))

def eval_depth(gt_depth_img_path, gen_img):

    # ---- 1. Load GT depth ----
    gt = load_gt_depth(gt_depth_img_path)
    if gt is None:
        return {"depth_error": "GT depth not found"}

    # ---- 2. Predict depth using MiDaS ----
    pred = predict_depth(gen_img)
    if pred is None or pred.ndim != 2:
        return {"depth_error": "pred depth invalid"}

    # ---- 3. Resize GT to pred size ----
    H, W = pred.shape
    gt_resized = cv2.resize(gt, (W, H), interpolation=cv2.INTER_LINEAR)

    # ---- 4. Compute SILog (best) ----
    silog_err = silog(gt_resized, pred)

    # ---- 5. Compute SSIM ----
    ssim_val, _ = ssim(
        gt_resized.astype(np.float32),
        pred.astype(np.float32),
        full=True,
        data_range=1.0
    )

    return {
        "depth_ssim": float(ssim_val),
        "depth_silog": float(silog_err) if silog_err is not None else None
    }

#####################################
# Pose Consistency with YOLOv8 Pose
#####################################
pose_model = YOLO('yolov8n-pose.pt')

def skeleton_to_mask(img, threshold=30):
    """
    将彩色 skeleton 渲染图转成二值 mask（骨架=255，背景=0）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    return mask


def keypoints_to_skeleton_image(img_shape, keypoints):
    """
    根据 YOLOv8 pose 的关键点，绘制 skeleton，生成 mask。
    img_shape: 原图 shape (H, W, C)
    keypoints: Nx2 array
    """
    H, W = img_shape[:2]
    skeleton = np.zeros((H, W), dtype=np.uint8)

    # COCO keypoint pairs（YOLOv8 pose 的标准拓扑）
    pairs = [
        (5,7),(7,9),
        (6,8),(8,10),
        (11,13),(13,15),
        (12,14),(14,16),
        (5,6),(11,12),(5,11),(6,12)
    ]

    for (i, j) in pairs:
        if i < len(keypoints) and j < len(keypoints):
            p1 = keypoints[i]
            p2 = keypoints[j]
            cv2.line(skeleton,
                     (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     255,
                     3)

    return skeleton


def pose_mask_metrics(gt_mask, pred_mask):

    # 两者 resize 成一致尺寸
    if gt_mask.shape != pred_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
    union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()

    iou = intersection / (union + 1e-6)

    gt_float = gt_mask.astype(np.float32) / 255.0
    pr_float = pred_mask.astype(np.float32) / 255.0

    ssim_val, _ = ssim(gt_float, pr_float, full=True, data_range=1.0)

    return {
        "pose_iou": float(iou),
        "pose_ssim": float(ssim_val)
    }

def eval_pose(gt_pose_img, gen_img):
    """
    计算 pose consistency：
      - GT skeleton 图 → mask
      - 生成图 → YOLO keypoints → skeleton mask
      - IoU + SSIM
    """

    # ---- 1. GT skeleton → mask ----
    gt_mask = skeleton_to_mask(gt_pose_img)

    # ---- 2. YOLO keypoints from generated image ----
    result = pose_model(gen_img)[0]

    if len(result.keypoints) == 0:
        # 生成图没有人物，pose 无法对齐
        return {"pose_iou": 0.0, "pose_ssim": 0.0}

    keypoints = result.keypoints.xy[0].cpu().numpy()

    # ---- 3. keypoints → mask ----
    pred_mask = keypoints_to_skeleton_image(gen_img.shape, keypoints)

    # ---- 4. compute IoU + SSIM ----
    return pose_mask_metrics(gt_mask, pred_mask)


#####################################
# Top-level API
#####################################
def evaluate_control(gt_path, gen_path, mode):
    gt = cv2.imread(gt_path)
    gen = cv2.imread(gen_path)

    if gt is None:
        return {"error": f"Cannot read control image {gt_path}"}
    if gen is None:
        return {"error": f"Cannot read generated image {gen_path}"}

    if mode == "canny":
        return eval_canny(gt, gen)

    elif mode == "pose":
        return eval_pose(gt, gen)

    elif mode == "depth":
        # IMPORTANT FIX: pass path, not image
        return eval_depth(gt_path, gen)

    else:
        return {"error": "unknown mode"}


def save_result_json(output_path, gt_img_path, gen_img_path, metrics_dict):
    """
    保存评估结果到 JSON 文件。
    包含:
        - ground truth 控制图路径
        - 生成图路径
        - 具体 metric 结果
    """

    result = {
        "gt_control_image": gt_img_path,
        "generated_image": gen_img_path,
        "metrics": metrics_dict
    }

    with open(output_path, "a") as f:
        json.dump(result, f, indent=4)
        f.write(",")  # 方便后续处理为列表
        f.write("\n")

    print(f"[✓] JSON saved to {output_path}")


if __name__ == "__main__":

    csig = ["depth", "pose"]  # canny / depth / pose
    model = ["sd15", "sdxl"]  # sd15 / sdxl

    for cs in csig:
        for m in model:
            print(f"Evaluating {cs} consistency for model {m}...")
            CONDITIONS_PATH = f"../mini_data/ctrl_gen/{cs}/conditions/"
            OUTPUTS_PATH = f"../Model/ctrl_gen/results/{cs}/{m}/"

            if len(os.listdir(OUTPUTS_PATH)) == 0:
                continue

            for fname in os.listdir(OUTPUTS_PATH):
                print(f"Processing {fname}...")
                if not fname.endswith(".png"):
                    continue

                ctrl_path = os.path.join(CONDITIONS_PATH, fname)
                
                if cs == "canny" or cs == "pose":
                    ctrl_path = os.path.join(CONDITIONS_PATH, fname.replace(".png", ".jpg"))
                elif cs == "depth":
                    ctrl_path = os.path.join(CONDITIONS_PATH, fname.replace("_rgb", "_depth"))

                gen_path =  os.path.join(OUTPUTS_PATH, fname)
                
                out = evaluate_control(ctrl_path, gen_path, cs)
                print(out)

                output_path = f"./ctrl_gen/{cs}_metrics.json"
                
                save_result_json(
                    output_path=output_path,
                    gt_img_path=ctrl_path,
                    gen_img_path=gen_path,
                    metrics_dict=out
                )