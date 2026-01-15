#!/usr/bin/env python3
"""
Test Suite for Mom Style LoRA Interfaces
Tests each interface to see which ones work
"""

import subprocess
import sys
import os

interfaces = {
    "final_working_interface.py": {
        "model": "lora_output_style_aware",
        "description": "Final working interface (older model)"
    },
    "kohya_ss_style_gradio_interface.py": {
        "model": "lora_output_kohya_style_aware", 
        "description": "Kohya-SS interface (newer model, full features)"
    },
    "mom_professional_interface.py": {
        "model": "lora_output_kohya_style_aware",
        "description": "Professional interface (newer model, clean)"
    },
    "kohya_simple_test.py": {
        "model": "lora_output_kohya_style_aware",
        "description": "Simple test (newer model, minimal)"
    }
}

print("=" * 80)
print("🎨 MOM STYLE LORA - INTERFACE TEST SUITE")
print("=" * 80)
print("\nTesting which interfaces work...\n")

working = []
broken = []

for interface_file, info in interfaces.items():
    model_path = info["model"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ {interface_file}")
        print(f"   └─ Model not found: {model_path}")
        broken.append((interface_file, f"Missing model: {model_path}"))
        continue
    
    # Check if file exists
    if not os.path.exists(interface_file):
        print(f"❌ {interface_file}")
        print(f"   └─ File not found")
        broken.append((interface_file, "File not found"))
        continue
    
    print(f"✅ {interface_file}")
    print(f"   └─ Model: {model_path}")
    print(f"   └─ {info['description']}")
    working.append((interface_file, info['description']))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n✅ Working Interfaces ({len(working)}):")
for interface, desc in working:
    print(f"   • {interface}")
    print(f"     {desc}")

if broken:
    print(f"\n❌ Broken Interfaces ({len(broken)}):")
    for interface, reason in broken:
        print(f"   • {interface}")
        print(f"     {reason}")

print("\n" + "=" * 80)
print("TO TEST AN INTERFACE, RUN:")
print("=" * 80)
for interface, _ in working:
    print(f"\n   python {interface}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
if working:
    best = working[0]  # First one is usually best
    print(f"\nStart with: python {best[0]}")
    print(f"Description: {best[1]}")
