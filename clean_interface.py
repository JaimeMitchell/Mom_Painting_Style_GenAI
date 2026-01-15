#!/usr/bin/env python3
"""
CLEAN WORKING INTERFACE - No complex code, just working LoRA
"""

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import random

# Load base model
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_style_aware"

print("🎨 Loading Rosanna Mitchell LoRA...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

# Load working LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

if torch.backends.mps.is_available():
    pipe.to("mps")
    print("✅ Using MPS")
else:
    pipe.to("cuda")
    print("✅ Using CUDA")

print("✅ LoRA loaded!")

def generate(prompt, steps=25, guidance=7.5, seed=None):
    """Generate image"""
    enhanced_prompt = f"{prompt}, rosanna mitchell art style, warm colors, garden"
    
    if seed:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(random.randint(0, 999999))
    
    result = pipe(
        enhanced_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]
    
    return result

# Simple interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Describe image", lines=3),
        gr.Slider(10, 50, value=25, label="Steps"),
        gr.Slider(1, 15, value=7.5, label="Guidance"),
        gr.Number(label="Seed")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="🎨 Rosanna Mitchell Generator",
    description="✅ Uses working LoRA - no black images!"
)

if __name__ == "__main__":
    print("\n🎉 CLEAN INTERFACE READY!")
    demo.launch(server_name="0.0.0.0", server_port=7860)
