import matplotlib.pyplot as plt
from PIL import Image
import os
import json

def show_images_with_labels_and_instruction(
    img_paths,
    labels,
    instruction,
    save_path=None,
    figsize=(18, 6),
    title_fontsize=12,
    instruction_fontsize=14,
    row=1,
    col=None
):
    """
    img_paths: 图片路径列表
    labels: 每张图下方的标签（与 img_paths 对应）
    instruction: 显示在整个图像下方的一句话
    save_path: 保存路径（自动创建目录）
    figsize: 图像整体大小
    """

    assert len(img_paths) == len(labels), "img_paths 和 labels 长度必须一致"

    images = [Image.open(p) for p in img_paths]
    n = len(images)
    if col is None:
        col = n
    

    plt.figure(figsize=figsize)

    # 显示每张图和对应label
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(row, col, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(label, fontsize=title_fontsize)

    # 在下方添加 instruction（全局居中）
    plt.figtext(
        0.5,          # X 坐标 (0~1)，0.5 为居中
        0.01,         # Y 坐标，越小越靠下
        f"Instruction: {instruction}",
        ha="center", fontsize=instruction_fontsize
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部instruction留空间

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved → {save_path}")

    plt.show()

def show(which, num=2):
    """
    显示指定类型的结果对比图
    which: "ctrl_gen", "editing", "inpainting"
    num: 显示的样本数量
    """
    if which == "ctrl_gen":
        ###############################################
        # show ctrl generation results
        ###############################################
        count = 0
        img_lists = os.listdir(f"mini_data/t2i/selected_100_images")
        show_num = num

        for img in img_lists:
            if count >= show_num:
                break
            count += 1

            img_path = [
                f"mini_data/t2i/selected_100_images/{img}",
                f"Model/ctrl_gen/results/glide/{img.replace('.jpg', '.png')}",
                f"Model/ctrl_gen/results/sd15/{img.replace('.jpg', '.png')}",
                f"Model/ctrl_gen/results/sdxl/{img.replace('.jpg', '.png')}",
                f"Model/t2i/results/stb_cas/{img.replace('.jpg', '.png')}"
            ]

            instruction = ""
            labels = [
                f"Input Image",
                f"GLIDE Result",
                f"SDv15 Result",
                f"SDXL Result",
                f"STB CAS Result"
            ]
            
            save_path = f"Model/ctrl_gen/comp_results/{img.replace('.jpg', '.png')}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            show_images_with_labels_and_instruction(
                img_path,
                labels=labels,
                instruction=instruction,
                save_path=save_path,
                figsize=(20, 5)
            )
    elif which == "editing":

        ###############################################
        # show editing results
        ###############################################
        show_num = num
        count = 0
        with open("mini_data/editing/magicbrush/edit_sessions_modified.json", "r") as f:
            edit_sessions = json.load(f)

        subfolder_lists = os.listdir(f"mini_data/editing/magicbrush/images")

        for subf in subfolder_lists:
            if count >= show_num:
                break
            count += 1

            img_path = [
                f"mini_data/editing/magicbrush/images/{subf}/{subf}-input.png",
                f"mini_data/editing/magicbrush/images/{subf}/{subf}-output1.png",
                f"Model/editing/results/flux/{subf}/{subf}-output1.png",
                f"Model/editing/results/diffedit/{subf}/{subf}-output1.png",
                f"Model/editing/results/ip2p/{subf}/{subf}-output1.png",
                f"Model/editing/results/sd15/{subf}/{subf}-output1.png",
                f"Model/editing/results/sdxl/{subf}/{subf}-output1.png",
                f"Model/editing/results/sdxl_refiner/{subf}/{subf}-output1.png",
            ]

            instruction = edit_sessions[subf][0]["instruction"]
            labels = [
                f"Input Image",
                f"Output Image",
                f"Flux Result",
                f"DiffEdit Result",
                f"IP2P Result",
                f"SDv15 Result",
                f"SDXL Result",
                f"SDXL Refiner Result"
            ]

            save_path = f"Model/editing/comp_results/{subf}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            show_images_with_labels_and_instruction(
                img_path,
                labels=labels,
                instruction=instruction,
                save_path=save_path,
                figsize=(20, 5)
            )

    elif which == "inpainting":
        ###############################################
        # show inpainting results
        ###############################################
        show_num = num
        count = {
            "celeba": 0,
            "places": 0,
            "cub": 0
        }

        img_lists = os.listdir(f"mini_data/inpainting/all_150/images")
        for img in img_lists:
            if "celeba" in img:
                count["celeba"] = count.get("celeba", 0) + 1
                if count["celeba"] > num / 3:
                    continue
            elif "places" in img:
                count["places"] = count.get("places", 0) + 1
                if count["places"] > num / 3:
                    continue
            elif "cub" in img:
                count["cub"] = count.get("cub", 0) + 1
                if count["cub"] > num / 3:
                    continue

            img_path = [
                f"mini_data/inpainting/all_150/images/{img}",
                f"mini_data/inpainting/all_150/masks/{img}",
                f"Model/inpainting/results/flux_fill/{img}",
                f"Model/inpainting/results/lama/{img}",
                f"Model/inpainting/results/repaint/{img}",
                f"Model/inpainting/results/sdxl/{img}",
                f"Model/inpainting/results/stb_cas/{img}"
            ]
            instruction = ""
            labels = [
                f"Input Image",
                f"Input Mask",
                f"Flux Fill Result",
                f"Lama Result",
                f"Repaint Result",
                f"SDXL Result",
                f"STB CAS Result"
            ]
            save_path = f"Model/inpainting/comp_results/{img}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            show_images_with_labels_and_instruction(
                img_path,
                labels=labels,
                instruction=instruction,
                save_path=save_path,
                figsize=(20, 5)
            )

    else: # t2i
        ###############################################
        # show text to image results
        ###############################################
        show_num = num
        count = 0

        subf_lists = os.listdir(f"MLLM-as-a-Judge/T2I Data")
        for subf in subf_lists:
            if count >= show_num:
                break
            count += 1

            img_path = []
            num_img_max = 8
            for img in os.listdir(f"MLLM-as-a-Judge/T2I Data/{subf}"):
                if not img.endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path.append(f"MLLM-as-a-Judge/T2I Data/{subf}/{img}")
                print(img_path)
                if len(img_path) >= num_img_max:
                    break

                with open(f"MLLM-as-a-Judge/T2I Data/{subf}/prompt.txt", "r") as f:
                    instruction = f.read().strip()

                labels = ["Image " + str(i+1) for i in range(num_img_max)]
                print(labels)
            
            save_path = f"Results/T2I/{subf}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            show_images_with_labels_and_instruction(
                img_path,
                labels=labels,
                instruction=instruction,
                save_path=save_path,
                figsize=(20, 5),
                row=2,
                col=4
            )        

if __name__ == "__main__":
    which = input("Which results to show? (ctrl_gen / editing / inpainting / t2i): ").strip()
    num = int(input("How many samples to show?: ").strip())
    show(which, num)