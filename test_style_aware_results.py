#!/usr/bin/env python3
"""
Test the Style-Aware mom LoRA Results
This loads the completed style-aware training and generates test images
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

# ========================
# LOAD STYLE-AWARE MODEL
# ========================
print("🎨 Testing Style-Aware mom LoRA...")

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_style_aware"
CONCEPT_NAME = "mom_art"

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
    print("✅ Using M1/M2 Metal Performance Shaders (MPS)")
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

# Load pipeline with style-aware LoRA
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False
)

# Load the style-aware LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

if DEVICE == "mps":
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

pipe.to(DEVICE)

print("✅ Style-aware LoRA loaded successfully!")
print("💎 Using BEST model from style-aware training")

# ========================
# GENERATE STYLE-SPECIFIC IMAGES
# ========================
style_test_prompts = [
    "warm red garden landscape with bright light, mom style",
    "mom art, bright light paintings with warm tones",
    "garden scene with warm red-orange palette and luminous lighting",
    "mom art, soft warm colors and golden light",
    "warm colored garden flowers with bright illumination"
]

print(f"\n🧪 Generating {len(style_test_prompts)} style-specific test images...")

for i, prompt in enumerate(style_test_prompts):
    print(f"\n   Testing {i+1}/5: {prompt}")
    
    try:
        # Generate with style-aware model
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            result = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        # Save the result
        filename = f"style_aware_test_{i+1}.png"
        result.save(filename)
        print(f"   ✅ Saved: {filename}")
        
    except Exception as e:
        print(f"   ❌ Error generating: {e}")

print(f"\n🎉 STYLE-AWARE TESTING COMPLETED!")
print(f"📁 Check the generated images:")
for i in range(1, len(style_test_prompts) + 1):
    filename = f"style_aware_test_{i}.png"
    if os.path.exists(filename):
        print(f"   ✅ {filename}")

print(f"\n🎯 COMPARISON:")
print(f"   - Original generic results: enhanced_test_*.png")
print(f"   - Style-aware results: style_aware_test_*.png")
print(f"   - These should look MORE like Mom's actual warm, bright style!")

print(f"\n💡 TO USE THE WEB INTERFACE:")
print(f"   python style_aware_gradio_interface.py")
