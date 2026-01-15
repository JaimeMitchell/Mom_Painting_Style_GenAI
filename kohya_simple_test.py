#!/usr/bin/env python3
"""
Simple Kohya-SS Test Interface
Quick test to see if Kohya-SS training captured Mom's style
"""

import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import json

# Load model
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_kohya_style_aware"

print("🎨 Loading Kohya-SS Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

try:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    print("✅ Kohya-SS LoRA loaded!")
except Exception as e:
    print(f"❌ Error loading LoRA: {e}")
    exit(1)

if torch.backends.mps.is_available():
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("mps")
    print("✅ Using M1/M2 Metal Performance Shaders (MPS)")
else:
    pipe.to("cuda")

def generate_mom_style(prompt, steps=30, guidance=7.5):
    """Generate image with Kohya-SS model"""
    enhanced_prompt = f"mom_art, {prompt}, warm red color palette, light paintings, garden focus, bright warm tones, soft brushwork, consistent style"
    
    result = pipe(
        enhanced_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    
    return result

# Simple interface
demo = gr.Interface(
    fn=generate_mom_style,
    inputs=[
        gr.Textbox(label="🎨 Describe image in Mom's style", placeholder="warm garden landscape with morning light..."),
        gr.Slider(10, 50, value=30, label="🎯 Steps"),
        gr.Slider(1, 15, value=7.5, label="🎭 Guidance")
    ],
    outputs=gr.Image(label="🎨 Generated with Kohya-SS Model"),
    title="🎨 Kohya-SS mom Style Test",
    description="Test if Kohya-SS training captured Mom's actual painting style"
)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎉 Kohya-SS Style Test Interface Ready!")
    print("="*50)
    print("Test prompts:")
    print("- warm garden landscape with morning light")
    print("- bright painting with soft warm colors") 
    print("- garden scene with red-orange palette")
    print("="*50)
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
