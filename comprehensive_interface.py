#!/usr/bin/env python3
"""
COMPREHENSIVE mom INTERFACE - All Controls You Want
Using the proven working LoRA with full feature set
"""

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import random

# Load the WORKING setup
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_output_style_aware"

print("🎨 Loading Comprehensive mom Interface...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

if torch.backends.mps.is_available():
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.to("mps")
    print("✅ Using M1/M2 Metal Performance Shaders (MPS)")
else:
    pipe.to("cuda")

print("✅ Working LoRA loaded!")

def generate_comprehensive(
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    seed,
    width,
    height,
    style_intensity,
    num_images
):
    """Generate with ALL the controls"""
    
    # Enhanced prompt based on style intensity
    if style_intensity <= 0.3:
        style_elements = "mom art, warm colors"
    elif style_intensity <= 0.6:
        style_elements = "mom art, warm red palette, garden"
    else:
        style_elements = "mom art, warm red color palette, light paintings, garden focus, bright warm tones, soft brushwork"
    
    enhanced_prompt = f"{prompt}, {style_elements}"
    
    # Set seed
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(random.randint(0, 999999))
    
    # Generate
    results = []
    for i in range(num_images):
        result = pipe(
            enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        results.append(result)
    
    if num_images == 1:
        return results[0]
    else:
        return results

def random_seed():
    return random.randint(0, 999999)

# Comprehensive interface with ALL controls
with gr.Blocks(title="🎨 mom - Full Control Interface") as interface:
    
    gr.HTML("""
    <div style="text-align: center; background: linear-gradient(135deg, #8B4513, #D2691E); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.5em;">🎨 mom Generator</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em;">✅ Working LoRA • 🔧 Full Controls • 🎨 Professional Results</p>
    </div>
    """)
    
    with gr.Row():
        # Main Controls
        with gr.Column(scale=2):
            # Prompt input
            prompt = gr.Textbox(
                label="🎨 Describe Your Image",
                placeholder="warm garden landscape with morning sunlight and red-orange blooms...",
                lines=4
            )
            
            # Negative prompt
            negative_prompt = gr.Textbox(
                label="🚫 Negative Prompt",
                value="blurry, dark, low quality, poor lighting, cool colors, harsh shadows, negative mood",
                lines=3
            )
            
            # Size controls
            with gr.Row():
                width = gr.Dropdown([256, 384, 512, 640, 768, 896, 1024, 1152], value=512, label="📐 Width")
                height = gr.Dropdown([256, 384, 512, 640, 768, 896, 1024, 1152], value=512, label="📐 Height")
            
            # Generation parameters
            with gr.Row():
                steps = gr.Slider(10, 100, value=30, step=5, label="🎯 Steps")
                guidance_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="🎭 Guidance")
            
            # Style and output controls
            with gr.Row():
                style_intensity = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="🎨 Style Intensity")
                num_images = gr.Slider(1, 4, value=1, step=1, label="🖼️ Number of Images")
            
            # Seed controls
            with gr.Row():
                seed = gr.Number(value=None, label="🎲 Seed")
                generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                random_seed_btn = gr.Button("🎲 Random", size="sm")
            
        # Side Panel
        with gr.Column(scale=1):
            # Model info
            gr.HTML("""
            <div style="background: #f5f5dc; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h3>🎯 mom Style</h3>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Warm red color palette</li>
                    <li>Light paintings</li>
                    <li>Garden landscapes</li>
                    <li>Bright warm tones</li>
                    <li>Soft brushwork</li>
                    <li>Uplifting mood</li>
                </ul>
            </div>
            """)
            
            # Controls info
            gr.HTML("""
            <div style="background: #e6f3ff; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4>📊 Controls Guide</h4>
                <p style="margin: 0; font-size: 0.9em;">
                    <strong>Size:</strong> Control image dimensions<br>
                    <strong>Steps:</strong> More = better quality<br>
                    <strong>CFG:</strong> Prompt adherence (7.5 ideal)<br>
                    <strong>Style:</strong> Mom style intensity<br>
                    <strong>Seed:</strong> Reproducible results
                </p>
            </div>
            """)
            
            # Examples
            gr.HTML("""
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px;">
                <h4>💡 Example Prompts</h4>
                <ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
                    <li>warm garden landscape with morning light</li>
                    <li>bright painting of flowers in warm colors</li>
                    <li>peaceful meadow with wildflowers</li>
                    <li>serene lake scene with mountains</li>
                    <li>beautiful garden with red-orange blooms</li>
                </ul>
            </div>
            """)
    
    # Output section
    with gr.Row():
        output_images = gr.Gallery(label="🎨 Generated Images", columns=2, height=400)
    
    status_output = gr.Markdown(label="📊 Generation Status")
    
    # Event handlers
    generate_btn.click(
        fn=generate_comprehensive,
        inputs=[prompt, negative_prompt, steps, guidance_scale, seed, width, height, style_intensity, num_images],
        outputs=[output_images]
    )
    
    random_seed_btn.click(
        fn=random_seed,
        inputs=[],
        outputs=[seed]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE INTERFACE READY!")
    print("="*60)
    print("✅ Using proven working LoRA")
    print("✅ All controls you requested:")
    print("   • Width/Height controls")
    print("   • Steps & Guidance sliders")
    print("   • Style intensity control")
    print("   • Multiple image generation")
    print("   • Negative prompts")
    print("   • Seed control")
    print("   • Professional interface")
    print("="*60)
    
    interface.launch(server_name="0.0.0.0", server_port=7860)
