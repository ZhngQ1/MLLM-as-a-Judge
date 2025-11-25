import os
import json
from tqdm import tqdm
from openai import OpenAI

# ==============================================
# CONFIG
# ==============================================
EDIT_JSON = "./mini_data/editing/magicbrush/edit_sessions.json"
OUT_JSON = "edit_sessions_enhanced.json"
MODEL = "gpt-4o-mini"
API_KEY = "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"   # 建议用环境变量

client = OpenAI(api_key=API_KEY)

def enhance_instruction(original_instruction, region_hint=None):
    """
    强化 MagicBrush editing instruction，使其更精确、局部化、适配 diffusion editing。
    输出风格类似：
    'Edit only the clothing on the woman standing at the right side of the image...'
    """

    system_prompt = """
You are an editing-instruction enhancement engine for image editing models (Flux, InstructPix2Pix, SDXL, DiffEdit).
Rewrite user editing prompts with these rules:

1. Output must be strictly ≤ 40 English words.
2. The instruction must be highly localized and precise.
3. Always specify *exactly* what region should be modified.
4. Always list what must *not* change (background, pose, body, scene).
5. Never hallucinate new people or new objects unless the user explicitly requests them.
6. Preserve the original meaning but make the edit description clearer, more explicit, and more model-friendly.
7. Use the following structure:

"Edit only the <specific region>. 
Do not change <body/pose/background>. 
<Describe precise edit action>."

8. Output should match this style example:
"Edit only the clothing on the woman standing at the right side of the image. 
Do not change her body, pose, or background. 
Replace only her shirt with a realistic party gown."
"""

    if region_hint is None:
        region_hint = "the relevant region"

    user_prompt = f"""
Original instruction: {original_instruction}
Region hint if available: {region_hint}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=100,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ==============================================
# Main: enhance all first-step instructions
# ==============================================
def run_enhancement():
    with open(EDIT_JSON, "r") as f:
        edit_data = json.load(f)

    out_data = edit_data.copy()

    for sid, steps in tqdm(out_data.items()):
        if not steps:
            continue

        orig = steps[0].get("instruction", "")
        try:
            enhanced = enhance_instruction(orig, sid)
        except Exception as e:
            enhanced = f"ERROR: {e}"

        out_data[sid][0]["instruction"] = enhanced
        print(f"[{sid}] {orig}  →  {enhanced}")

    with open(OUT_JSON, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"\nSaved enhanced instructions to {OUT_JSON}")


if __name__ == "__main__":
    run_enhancement()

