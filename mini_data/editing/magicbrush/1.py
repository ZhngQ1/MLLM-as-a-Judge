import json

EDIT_JSON = "edit_sessions.json"
CAPTION_JSON = "captions.json"
OUT_JSON = "edit_sessions_modified.json"

def replace_first_step_instruction():
    # 加载两个 JSON
    with open(EDIT_JSON, "r") as f:
        edit_data = json.load(f)

    with open(CAPTION_JSON, "r") as f:
        caption_data = json.load(f)

    # 遍历 edit_sessions
    for sid, steps in edit_data.items():
        if not isinstance(steps, list) or len(steps) == 0:
            print(f"[WARN] {sid} has no steps")
            continue

        if sid not in caption_data:
            print(f"[WARN] No caption for {sid}; skip")
            continue

        new_prompt = caption_data[sid]

        # 只修改第一步
        old_prompt = steps[0].get("instruction", "")
        steps[0]["instruction"] = new_prompt

        print(f"[{sid}] step[0]: '{old_prompt}' → '{new_prompt}'")

    # 保存结果
    with open(OUT_JSON, "w") as f:
        json.dump(edit_data, f, indent=4)

    print(f"\nSaved modified JSON to {OUT_JSON}")

if __name__ == "__main__":
    replace_first_step_instruction()
