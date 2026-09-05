#!/home/jingyue/miniconda3/envs/internimage/bin/python
"""Generate Qwen 2.5-VL 7B prompts for v5 FSC SAM data. GPU 1.
复用 qwen_mar20_prompts.py 的完全相同逻辑,仅 prompt 改为舰船版 + 数据源改为 v5_fsc_sam_data。
"""
import os, sys, json, torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

QWEN_PATH = "/home/jingyue/models/Qwen2.5-VL-7B-Instruct"
SAM_DATA = "/home/jingyue/EarthSynth/v5_fsc_sam_data"
DEVICE = "cuda:0"

QWEN_PROMPT = (
    "You are a military remote sensing analyst. Describe this satellite image for an AI image generator. "
    "Your description must be VISUALLY PRECISE so the AI can recreate the scene.\n\n"
    "1) SHIP DETAILS: Describe each ship visually WITHOUT naming specific classes. "
    "Use generic descriptions: vehicle size (small/medium/large), wheels or tracked, "
    "launcher configuration (raised launch rails, canister tubes, flatbed), "
    "cab layout, color, camouflage patterns, and whether parked alone or in formation. "
    "DO NOT guess class names. Instead say 'a medium vehicle with raised launch rails' or "
    "'a large vehicle with multiple canister tubes'. "
    "Mention arrangement (in a line, dispersed, in revetments).\n\n"
    "2) BACKGROUND: Describe the military depot or launch site environment realistically: "
    "concrete pad, dirt tracks, revetments, surrounding terrain, vegetation, buildings.\n\n"
    "Write TWO to THREE detailed English sentences. Start with: A satellite image of"
)

print("Loading Qwen2.5-VL-7B...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_PATH, torch_dtype=torch.bfloat16, device_map={"": DEVICE})
processor = AutoProcessor.from_pretrained(QWEN_PATH)
print("Loaded.")

for split in ["train", "val"]:
    meta_path = os.path.join(SAM_DATA, split, "metadata.jsonl")
    with open(meta_path) as f:
        entries = [json.loads(l) for l in f]

    prompts = {}
    for entry in tqdm(entries, desc=f"Qwen {split}"):
        fname = entry["file_name"]
        img_path = os.path.join(SAM_DATA, split, fname)
        old_prompt = entry.get("text", "")

        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": QWEN_PROMPT}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[Image.open(img_path)], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
            ids = ids[:, inputs["input_ids"].shape[1]:]
            prompt = processor.decode(ids[0], skip_special_tokens=True).strip()
            if not prompt.lower().startswith("a satellite image"):
                prompt = "A satellite image of " + prompt
        except Exception as e:
            prompt = old_prompt

        prompts[fname] = {"old": old_prompt, "qwen": prompt}
        # Update metadata
        entry["text"] = prompt
        entry["old_text"] = old_prompt

    # Save updated metadata
    with open(meta_path + ".bak", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    # Replace original
    os.rename(meta_path + ".bak", meta_path)

    # Save prompts map
    with open(os.path.join(SAM_DATA, split, "qwen_prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)

    # Print samples
    print(f"\n{split} samples:")
    for fname, p in list(prompts.items())[:3]:
        print(f"  OLD: {p['old'][:100]}...")
        print(f"  NEW: {p['qwen'][:120]}...")
        print()

print("Done!")
