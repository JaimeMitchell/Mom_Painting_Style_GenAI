#!/usr/bin/env python3

import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import random

BASE_MODEL_ID = "./stable-diffusion-v1-5"

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True
)

# Load LoRA adapter from lora_output_style_aware directory
adapter_dir = "lora_output_style_aware"  # Where the safetensors files are stored

try:
    # Load LoRA adapter from root directory
    pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter_dir)
    # CRITICAL: Ensure LoRA is enabled during inference
    pipe.unet.eval()  # Set to eval mode
    print("✅ Kaggle LoRA loaded and ACTIVE")
    
    # Verify LoRA is actually loaded
    if hasattr(pipe.unet, 'peft_config'):
        print(f"✅ PEFT config present: {pipe.unet.peft_config}")
    if hasattr(pipe.unet, 'active_adapters'):
        active = pipe.unet.active_adapters
        print(f"✅ Active adapters: {active}")
    
except Exception as e:
    print(f"❌ CRITICAL: LoRA load FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)  # Exit if LoRA fails to load

if torch.backends.mps.is_available():
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("mps")
else:
    pipe.to("cuda")

def generate_image(prompt, steps=30, guidance_scale=7.5, seed=None, width=512):
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(random.randint(0, 999999))
    
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=width,
        generator=generator
    ).images[0]
    
    return result

examples = [
    ["serene lake scene with mountains in mixed tones, mixed media", 30, 7.5, None, 512],
    ["beautiful garden with red-orange and purple blooms and uplifting mood", 30, 7.5, None, 512]
]

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Describe your image",
            placeholder="warm garden landscape with morning sunlight...",
            lines=3
        ),
        gr.Slider(10, 50, value=30, step=5, label="Steps"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance"),
        gr.Number(value=None, label="Seed (blank for random)"),
        gr.Dropdown([512, 768], value=512, label="Size")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Kaggle Model Tester",
    examples=examples
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        show_error=True,
        share=False
    )
