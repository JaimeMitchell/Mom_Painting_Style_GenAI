#!/usr/bin/env python3
# local_lora_inspect.py

import os
import glob
import torch
from safetensors.torch import safe_open
from diffusers import StableDiffusionPipeline

# --- CONFIGURATION ---
BASE_CHECKPOINT = "models/sd-1.5-base.ckpt"  # path to a known working checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PROMPT = "a photograph of a mountain landscape, high resolution, 4k"
OUTPUT_DIR = "test_outputs"
SKIP_IMAGE_TEST = False  # Set True to inspect metadata only

# --- FUNCTIONS ---
def inspect_safetensor(path):
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata()
        return meta
    except Exception as e:
        print("  [!] Could not parse metadata:", e)
        return None

def quick_generate(prompt, model_path, lora_path=None, output_path=None):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    if lora_path:
        print("  Loading LoRA:", lora_path)
        pipe.unet.load_attn_procs(lora_path)
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
    return image

def is_lora_file(path):
    name = os.path.basename(path).lower()
    return "lora" in name or "adapter" in name

def main(folder):
    # Recursive glob
    for ext in ("*.safetensors", "*.ckpt"):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(ext.replace("*","")):
                    filepath = os.path.join(root, file)
                    print("\n=== Inspecting:", filepath)
                    meta = None
                    if filepath.endswith(".safetensors"):
                        meta = inspect_safetensor(filepath)
                    if meta:
                        print("  Metadata:", meta)
                    else:
                        print("  No metadata found or parse failed.")

                    if SKIP_IMAGE_TEST:
                        continue

                    try:
                        if is_lora_file(filepath):
                            print("  ==> Testing as LoRA on base checkpoint")
                            out = quick_generate(
                                TEST_PROMPT,
                                BASE_CHECKPOINT,
                                lora_path=filepath,
                                output_path=os.path.join(OUTPUT_DIR, os.path.relpath(filepath, folder) + ".png")
                            )
                        else:
                            print("  ==> Testing as base checkpoint (full model)")
                            out = quick_generate(
                                TEST_PROMPT,
                                filepath,
                                output_path=os.path.join(OUTPUT_DIR, os.path.relpath(filepath, folder) + ".png")
                            )
                        print("  ✔ Success — output saved" if out else "  ✘ Generation failed")
                    except Exception as e:
                        print("  [!] Generation error:", e)

# --- ENTRY POINT ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python local_lora_inspect.py <models_folder>")
        exit(1)
    folder = sys.argv[1]
    main(folder)

