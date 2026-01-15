#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC SCRIPT - Find the REAL cause of black images
This script will diagnose EVERYTHING step by step
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import numpy as np

print("🔍 COMPREHENSIVE DIAGNOSTIC - Finding REAL Black Image Cause")
print("="*70)

# 1. CHECK ALL LoRA PATHS
print("\n1️⃣ CHECKING ALL LoRA PATHS...")
lora_paths = [
    "./lora_output_style_aware",
    "./lora_output_kohya_style_aware", 
    "./lora_output_improved",
    "./lora_output",
    "./lora_output_improved/best_lora",
    "./lora_output_style_aware/best_lora"
]

for path in lora_paths:
    if os.path.exists(path):
        safetensors = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(safetensors):
            size_mb = os.path.getsize(safetensors) / (1024*1024)
            print(f"   ✅ {path}: {size_mb:.1f}MB")
        else:
            print(f"   ❌ {path}: No adapter_model.safetensors")
    else:
        print(f"   ❌ {path}: Directory doesn't exist")

# 2. TEST EACH LoRA SEPARATELY
print("\n2️⃣ TESTING EACH LoRA SEPARATELY...")

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
test_prompt = "warm garden landscape with morning sunlight"

# Load base model once
print("📦 Loading base model...")
base_pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

if torch.backends.mps.is_available():
    base_pipe.to("mps")
    print("✅ Base model loaded on MPS")
else:
    base_pipe.to("cuda")
    print("✅ Base model loaded on CUDA")

def test_lora(lora_path, name):
    """Test a specific LoRA"""
    print(f"\n🧪 Testing {name} ({lora_path})...")
    
    try:
        # Fresh copy of base model
        test_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load LoRA
        test_pipe.unet = PeftModel.from_pretrained(test_pipe.unet, lora_path)
        
        if torch.backends.mps.is_available():
            test_pipe.to("mps")
        
        # Generate test image
        print(f"   Generating: {test_prompt}")
        result = test_pipe(
            test_prompt,
            num_inference_steps=20,
            guidance_scale=7.0
        ).images[0]
        
        # Check if image is valid
        if result.size == (512, 512) and result.mode == 'RGB':
            # Check pixel values
            img_array = np.array(result)
            mean_brightness = np.mean(img_array)
            
            filename = f"test_{name.replace('/', '_').replace('.', '_')}.png"
            result.save(filename)
            
            if mean_brightness < 10:
                print(f"   ❌ {name}: BLACK IMAGE (brightness: {mean_brightness:.1f})")
                return False
            elif mean_brightness < 50:
                print(f"   ⚠️ {name}: DARK IMAGE (brightness: {mean_brightness:.1f})")
                return True
            else:
                print(f"   ✅ {name}: GOOD IMAGE (brightness: {mean_brightness:.1f})")
                return True
        else:
            print(f"   ❌ {name}: Invalid image dimensions/mode")
            return False
            
    except Exception as e:
        print(f"   ❌ {name}: ERROR - {e}")
        return False

# Test each LoRA
working_loras = []
for path in lora_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        if test_lora(path, path):
            working_loras.append(path)

print(f"\n📊 SUMMARY:")
print(f"   Working LoRAs: {len(working_loras)}")
for lora in working_loras:
    print(f"   ✅ {lora}")

# 3. TEST BASE MODEL WITHOUT LoRA
print("\n3️⃣ TESTING BASE MODEL (NO LoRA)...")
try:
    result = base_pipe(
        test_prompt,
        num_inference_steps=20,
        guidance_scale=7.0
    ).images[0]
    
    img_array = np.array(result)
    mean_brightness = np.mean(img_array)
    
    result.save("test_base_model.png")
    
    if mean_brightness < 10:
        print(f"   ❌ BASE MODEL: BLACK IMAGE (brightness: {mean_brightness:.1f})")
    elif mean_brightness < 50:
        print(f"   ⚠️ BASE MODEL: DARK IMAGE (brightness: {mean_brightness:.1f})")
    else:
        print(f"   ✅ BASE MODEL: GOOD IMAGE (brightness: {mean_brightness:.1f})")
        
except Exception as e:
    print(f"   ❌ BASE MODEL: ERROR - {e}")

# 4. CHECK MPS/DTYPE ISSUES
print("\n4️⃣ CHECKING MPS/DTYPE ISSUES...")
if torch.backends.mps.is_available():
    print("   ✅ MPS available")
    print("   🔧 Testing MPS float16...")
    try:
        mps_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16"
        )
        mps_pipe.to("mps")
        
        result = mps_pipe(
            test_prompt,
            num_inference_steps=10,
            guidance_scale=7.0
        ).images[0]
        
        img_array = np.array(result)
        mean_brightness = np.mean(img_array)
        
        result.save("test_mps_float16.png")
        
        if mean_brightness < 10:
            print(f"   ❌ MPS float16: BLACK IMAGE (brightness: {mean_brightness:.1f})")
        else:
            print(f"   ✅ MPS float16: WORKS (brightness: {mean_brightness:.1f})")
            
    except Exception as e:
        print(f"   ❌ MPS float16: ERROR - {e}")
else:
    print("   ❌ MPS not available")

# 5. FINAL DIAGNOSIS
print("\n" + "="*70)
print("🎯 FINAL DIAGNOSIS:")
if len(working_loras) == 0:
    print("   ❌ NO WORKING LoRAs FOUND!")
    print("   💡 SOLUTION: Retrain LoRA or use base model only")
elif len(working_loras) == 1:
    print(f"   ✅ ONE WORKING LoRA: {working_loras[0]}")
    print(f"   💡 SOLUTION: Use only this LoRA in interfaces")
else:
    print(f"   ✅ MULTIPLE WORKING LoRAs: {len(working_loras)}")
    print(f"   💡 SOLUTION: Choose the best one and remove others")

print("\n📁 Check generated test images:")
for file in os.listdir("."):
    if file.startswith("test_") and file.endswith(".png"):
        print(f"   📸 {file}")

print("\n💡 NEXT STEPS:")
print("   1. Identify which test image looks best")
print("   2. Use that LoRA path in your interface")
print("   3. Remove/disable other LoRA paths")
print("="*70)
