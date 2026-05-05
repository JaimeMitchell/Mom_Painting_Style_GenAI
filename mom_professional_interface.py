#!/usr/bin/env python3
import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import random

# --- ALIGN THESE WITH YOUR TRAINING SCRIPT ---
BASE_MODEL_ID = "./stable-diffusion-v1-5"
LORA_BASE_DIR = "./lora_mom_art_final"  # Now points to your latest successful training

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 1. Load Base Model Once
# Using float16 for faster inference on Mac/GPU; change to float32 if on older CPU
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True
).to(device)

def get_available_checkpoints():
    if not os.path.exists(LORA_BASE_DIR): 
        return []
    # Only pick directories that contain a LoRA configuration (to avoid picking random folders)
    checkpoints = [
        d for d in os.listdir(LORA_BASE_DIR) 
        if os.path.isdir(os.path.join(LORA_BASE_DIR, d)) and "checkpoint" in d
    ]
    # Include the final base LoRA folder if it exists
    if os.path.exists(os.path.join(LORA_BASE_DIR, "adapter_model.safetensors")):
        checkpoints.append("Final_Model")
        
    return sorted(checkpoints)

def load_selected_checkpoint(checkpoint_name):
    # CRITICAL FIX: Handle None or empty selection
    if not checkpoint_name or checkpoint_name == "No checkpoints found":
        print("Skipping load: No valid checkpoint selected.")
        return False
    
    # Path logic
    if checkpoint_name == "Final_Model":
        checkpoint_path = LORA_BASE_DIR
    else:
        checkpoint_path = os.path.join(LORA_BASE_DIR, checkpoint_name)
    
    print(f"🔄 Loading LoRA weights from: {checkpoint_path}")
    
    # Handle LoRA switching
    try:
        # If PeftModel was already applied, we unwrap it back to base UNet first
        if hasattr(pipe, "unet") and isinstance(pipe.unet, PeftModel):
            pipe.unet = pipe.unet.unload()
            
        pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_path)
        return True
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        return False

def normalize_prompt(prompt):
    """Auto-correct keywords to match exact training captions and add mom_art token"""
    keyword_map = {
        # Mediums (all case variations)
        "acrylic on canvas": "Acrylic on Canvas",
        "acrylic collage with handmade paper": "Acrylic Collage with Handmade Paper",
        "ecoprint on watercolor paper": "Ecoprint on Watercolor Paper",
        "ink on yupo paper": "Ink on Yupo Paper",
        "acrylic pour on canvas": "Acrylic Pour on Canvas",
        "soft pastel on paper": "Soft Pastel on Paper",
        "oil and pastel on canvas": "Oil and Pastel on Canvas",
        "mixed media collage on paper": "Mixed Media Collage on Paper",
        "acrylic & ink collage": "Acrylic & Ink Collage",
        "ink on watercolor paper": "Ink on Watercolor Paper",
        "ink collage on yupo paper": "Ink Collage on Yupo Paper",
        "watercolor on paper": "watercolor on paper",
    }
    
    normalized = prompt.lower()
    for key, value in keyword_map.items():
        normalized = normalized.replace(key, value)
    
    # Auto-prepend "mom_art," if not already present
    if not normalized.startswith("mom_art"):
        normalized = f"mom_art, {normalized}"
    
    return normalized

def generate_image(prompt, checkpoint_name, steps=30, guidance=7.5, seed=None, size=512):
    # Normalize prompt keywords to exact training format
    normalized_prompt = normalize_prompt(prompt)
    
    success = load_selected_checkpoint(checkpoint_name)
    if not success and checkpoint_name:
        raise gr.Error(f"Could not load checkpoint: {checkpoint_name}")
    
    fixed_seed = random.randint(0, 999999) if not seed else int(seed)
    generator = torch.Generator(device=pipe.device).manual_seed(fixed_seed)
    
    image = pipe(
        normalized_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=int(size),
        height=int(size),
        generator=generator
    ).images[0]
    return image

# --- GRADIO INTERFACE ---
with gr.Blocks() as demo: # Moved theme to launch() per Gradio 6.0 warning
    gr.Markdown("# 🎨 mom_art Style Explorer")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", value="abstract landscape, vibrant colors", lines=3)
            
            # Initialize choices safely
            initial_choices = get_available_checkpoints()
            checkpoint_dropdown = gr.Dropdown(
                choices=initial_choices if initial_choices else ["No checkpoints found"], 
                value=initial_choices[0] if initial_choices else "No checkpoints found",
                label="Select Epoch (Checkpoint)"
            )
            refresh_btn = gr.Button("🔄 Refresh List", size="sm")

            generate_btn = gr.Button("🚀 Generate Image", variant="primary")
            output_img = gr.Image(label="Result")

        with gr.Column(scale=1):
            gr.Markdown("""### 💡 How to Prompt
            **The Formula:**
            `[Subject], [Medium Keyword], [Color Details]`
            
            **Medium Keywords to try:**
            *   `Acrylic Pour on Canvas`
            *   `Ecoprint on Watercolor Paper`
            *   `Ink on Yupo Paper`
            *   `Acrylic Collage with Handmade Paper`
            *   `Soft Pastel on Paper`
            """)
            
            with gr.Accordion("Advanced Settings", open=False):
                steps_val = gr.Slider(10, 50, value=30, step=1, label="Steps")
                guidance_val = gr.Slider(1, 15, value=7.5, step=0.1, label="Guidance (CFG)")
                seed_val = gr.Textbox(label="Seed (Leave blank for random)")
                size_val = gr.Dropdown([512, 768], value=512, label="Resolution")

    # --- ACTIONS ---
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, checkpoint_dropdown, steps_val, guidance_val, seed_val, size_val],
        outputs=output_img
    )
    
    def refresh():
        choices = get_available_checkpoints()
        return gr.update(choices=choices if choices else ["No checkpoints found"])

    refresh_btn.click(fn=refresh, outputs=checkpoint_dropdown)

if __name__ == "__main__":
    # Fix the theme warning by passing it to launch
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True, theme=gr.themes.Soft())