#!/usr/bin/env python3
"""
Simple test to verify the Mom LoRA model works
Tests: Model loading, LoRA loading, and image generation
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

print("=" * 60)
print("🎨 TESTING MOM LORA SETUP")
print("=" * 60)

# Configuration
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_kohya_style_aware"
OUTPUT_DIR = "./test_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print("\n1️⃣ Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    print("   ✅ Base model loaded successfully")
    
    print("\n2️⃣ Loading LoRA weights...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    print("   ✅ LoRA weights loaded successfully")
    
    print("\n3️⃣ Setting device...")
    if torch.backends.mps.is_available():
        pipe = pipe.to("mps")
        device = "mps"
        print("   ✅ Using M1/M2 Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
        device = "cuda"
        print("   ✅ Using CUDA GPU")
    else:
        pipe = pipe.to("cpu")
        device = "cpu"
        print("   ✅ Using CPU")
    
    print("\n4️⃣ Generating test image...")
    prompt = "mom_art, warm garden landscape with bright light, soft brushwork"
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
    
    print("   ✅ Image generated successfully")
    
    print("\n5️⃣ Saving test image...")
    output_path = os.path.join(OUTPUT_DIR, "test_output.png")
    image.save(output_path)
    print(f"   ✅ Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✨ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n📝 Prompt used:", prompt)
    print("🎨 Model device:", device)
    print("💾 Output saved to:", output_path)
    print("\n🚀 The Mom LoRA model is working correctly!")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ TEST FAILED")
    print("=" * 60)
    print(f"\n❌ Error: {str(e)}")
    print("\nTroubleshooting:")
    print("  1. Check if lora_output_kohya_style_aware/ exists")
    print("  2. Verify adapter_model.safetensors is in the folder")
    print("  3. Ensure you have enough VRAM/RAM")
    print("  4. Try with smaller image size (256x256)")
    import traceback
    traceback.print_exc()
