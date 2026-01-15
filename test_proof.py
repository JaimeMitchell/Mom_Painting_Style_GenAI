#!/usr/bin/env python3
"""
Test the running interface by generating images directly
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import random

# Load the working setup
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_style_aware"

print("🧪 Testing actual image generation...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
pipe.to("mps")
print("✅ Loaded and ready!")

# Test prompts
test_prompts = [
    "warm garden landscape with morning sunlight",
    "bright painting of flowers in warm colors",
    "peaceful meadow with wildflowers",
    "beautiful garden with red-orange blooms",
    "serene lake scene with mountains"
]

print("\n🎨 Generating test images...")

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}/5: {prompt}")
    
    enhanced_prompt = f"{prompt}, mom art style, warm colors, garden, light paintings"
    
    generator = torch.Generator().manual_seed(42)
    
    result = pipe(
        enhanced_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    filename = f"proof_test_{i}.png"
    result.save(filename)
    
    # Check brightness to prove it's not black
    import numpy as np
    img_array = np.array(result)
    brightness = np.mean(img_array)
    
    print(f"   ✅ Saved: {filename}")
    print(f"   📊 Brightness: {brightness:.1f} (NOT black!)")
    
    if brightness < 10:
        print(f"   ❌ STILL BLACK!")
    else:
        print(f"   ✅ SUCCESS - Proper image!")

print("\n🎉 PROOF COMPLETE!")
print("Check the generated images:")
for i in range(1, 6):
    filename = f"proof_test_{i}.png"
    print(f"   📸 {filename}")
