#!/usr/bin/env python3
"""
LOCAL MODEL CONFIGURATION FOR LoRA TRAINING
Use your existing ComfyUI models instead of redownloading

SETUP INSTRUCTIONS:
1. Copy this file to 'config_local_models.py'
2. Update COMFYUI_MODELS_DIR with your actual ComfyUI path
3. Update any other paths as needed
"""

import os

# ========================
# LOCAL MODEL PATHS
# ========================

# Your ComfyUI Models Directory - UPDATE THIS!
COMFYUI_MODELS_DIR = "/path/to/your/ComfyUI/models"  # e.g., "/Users/yourname/Documents/comfy/ComfyUI/models"

# Stable Diffusion 1.5 (for basic LoRA training)
SD_1_5_PATH = os.path.join(COMFYUI_MODELS_DIR, "checkpoints", "v1-5-pruned-emaonly-fp16.safetensors")

# WAN 2.2 Video Models (for video LoRA training)
WAN_PATHS = {
    "i2v_high_noise": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
    "i2v_low_noise": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
    "s2v": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "wan2.2_s2v_14B_bf16.safetensors"),
}

# WAN 2.2 Quantized Models (GGUF format)
WAN_GGUF_PATHS = {
    "i2v_high_noise": os.path.join(COMFYUI_MODELS_DIR, "unet", "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf"),
    "i2v_low_noise": os.path.join(COMFYUI_MODELS_DIR, "unet", "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"),
    "s2v_q4": os.path.join(COMFYUI_MODELS_DIR, "unet", "Wan2.2-S2V-14B-Q4_K_M.gguf"),
    "s2v_q8": os.path.join(COMFYUI_MODELS_DIR, "unet", "Wan2.2-S2V-14B-Q8_0.gguf"),
}

# FLUX Models (Alternative option)
FLUX_PATHS = {
    "dev_q4": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "FLUX1", "flux1-dev-Q4_1.gguf"),
    "dev_q8": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "FLUX1", "flux1-dev-Q8_0.gguf"),
    "schnell_q4": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "FLUX1", "flux1-schnell-Q4_1.gguf"),
    "schnell_q8": os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", "FLUX1", "flux1-schnell-Q8_0.gguf"),
}

# Text Encoders
TEXT_ENCODER_PATHS = {
    "t5xxl": os.path.join(COMFYUI_MODELS_DIR, "text_encoders", "t5xxl_fp16.safetensors"),
    "mistral": os.path.join(COMFYUI_MODELS_DIR, "text_encoders", "mistral_3_small_flux2_bf16.safetensors"),
    "t5_gguf": os.path.join(COMFYUI_MODELS_DIR, "text_encoders", "t5", "t5-v1_1-xxl-encoder-Q8_0.gguf"),
}

# VAE Models
VAE_PATHS = {
    "flux": os.path.join(COMFYUI_MODELS_DIR, "vae", "flux2-vae.safetensors"),
    "wan": os.path.join(COMFYUI_MODELS_DIR, "vae", "wan_2.1_vae.safetensors"),
}

# Existing LoRA Models (for reference/fine-tuning)
EXISTING_LORAS = {
    "wan_i2v_high": os.path.join(COMFYUI_MODELS_DIR, "loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"),
    "wan_i2v_low": os.path.join(COMFYUI_MODELS_DIR, "loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
    "wan_t2v": os.path.join(COMFYUI_MODELS_DIR, "loras", "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"),
    "ltxv_detailer": os.path.join(COMFYUI_MODELS_DIR, "loras", "ltxv-098-ic-lora-detailer-comfyui.safetensors"),
}

# ========================
# MODEL CONFIGURATIONS
# ========================

# Configuration for different training scenarios
MODEL_CONFIGS = {
    # For basic image LoRA training (like your current setup)
    "sd1_5_image": {
        "base_model_path": SD_1_5_PATH,
        "model_type": "stable_diffusion",
        "description": "Stable Diffusion 1.5 for image LoRA training"
    },
    
    # For video LoRA training using WAN 2.2
    "wan2_2_video": {
        "base_model_path": WAN_PATHS["s2v"],
        "model_type": "wan_video",
        "description": "WAN 2.2 for video LoRA training",
        "text_encoder": TEXT_ENCODER_PATHS["t5xxl"],
        "vae": VAE_PATHS["wan"]
    },
    
    # For quantized video training (faster, less VRAM)
    "wan2_2_quantized": {
        "base_model_path": WAN_GGUF_PATHS["s2v_q8"],
        "model_type": "wan_video_quantized",
        "description": "WAN 2.2 quantized for efficient video LoRA training",
        "text_encoder": TEXT_ENCODER_PATHS["t5_gguf"],
        "vae": VAE_PATHS["wan"]
    },
    
    # Alternative: FLUX model
    "flux": {
        "base_model_path": FLUX_PATHS["dev_q8"],
        "model_type": "flux",
        "description": "FLUX dev model for high-quality LoRA training",
        "text_encoder": TEXT_ENCODER_PATHS["mistral"],
        "vae": VAE_PATHS["flux"]
    }
}

def check_model_exists(model_path):
    """Check if a model file exists"""
    return os.path.exists(model_path)

def get_available_models():
    """Get list of available models in your ComfyUI folder"""
    available = {}
    
    for config_name, config in MODEL_CONFIGS.items():
        if check_model_exists(config["base_model_path"]):
            available[config_name] = config
            print(f"✅ {config_name}: {config['description']}")
        else:
            print(f"❌ {config_name}: Model not found at {config['base_model_path']}")
    
    return available

def get_model_path(config_name):
    """Get the path for a specific model configuration"""
    if config_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[config_name]
    return None

if __name__ == "__main__":
    print("🔍 Checking available local models...")
    available_models = get_available_models()
    
    print(f"\n📁 ComfyUI Models Directory: {COMFYUI_MODELS_DIR}")
    print(f"🎯 Found {len(available_models)} usable models for LoRA training")
