#!/usr/bin/env python3

import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import random

BASE_MODEL_ID = "./stable-diffusion-v1-5"
LORA_PATHS = [
    "./lora_output_kohya_style_aware",
    "./lora_output_style_aware"
]

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

lora_loaded = False
for lora_path in LORA_PATHS:
    if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            lora_loaded = True
            break
        except:
            continue

if torch.backends.mps.is_available():
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("mps")
else:
    pipe.to("cuda")

def generate_image(prompt, steps=30, guidance_scale=7.5, seed=None, width=512, height=512):
    # Add mom_art trigger token that the LoRA was trained on
    if not prompt.lower().startswith("mom_art"):
        prompt = f"mom_art, {prompt}"
    
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(random.randint(0, 999999))
    
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    return result

examples = [
    ["mom_art, warm garden landscape with morning sunlight", 30, 7.5, None, 512],
    ["mom_art, bright painting of flowers in warm colors", 30, 7.5, None, 512],
    ["mom_art, peaceful meadow with wildflowers and soft brushwork", 30, 7.5, None, 512],
    ["mom_art, serene lake scene with mountains in warm tones", 30, 7.5, None, 512],
    ["mom_art, beautiful garden with red-orange blooms and uplifting mood", 30, 7.5, None, 512]
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
    title="mom's Art Generator",
    examples=examples
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
