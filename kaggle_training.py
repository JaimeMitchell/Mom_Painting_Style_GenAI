"""
Kaggle-optimized LoRA training for mom's painting style.
Upload this + Paintings folder to Kaggle, then run in a notebook.
"""

import os
import torch
from pathlib import Path
from PIL import Image
import random
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Kaggle paths
INPUT_DIR = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
PAINTINGS_DIR = INPUT_DIR / "paintings" if (INPUT_DIR / "paintings").exists() else Path("./Paintings")

# Create output directory
OUTPUT_DIR = WORKING_DIR / "lora_output_kaggle_style_aware"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Paintings directory: {PAINTINGS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Model configuration
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LEARNING_RATE = 8e-5
NUM_EPOCHS = 15
BATCH_SIZE = 1
LORA_RANK = 20
LORA_ALPHA = 40

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use float32 on CPU, float16 on GPU
torch_dtype = torch.float32 if device == "cpu" else torch.float16

class PaintingDataset(Dataset):
    def __init__(self, paintings_dir, transform=None):
        self.paintings_dir = Path(paintings_dir)
        self.image_files = list(self.paintings_dir.glob("*.jpg")) + list(self.paintings_dir.glob("*.png"))
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"pixel_values": image}

# Load model
print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype)
pipe.to(device)

# Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["to_k", "to_v", "to_q"],
    lora_dropout=0.1,
    bias="none",
)

pipe.unet = get_peft_model(pipe.unet, lora_config)

# Freeze VAE and text encoder
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# Optimizer
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LEARNING_RATE)

# Dataset
print("Loading training data...")
dataset = PaintingDataset(PAINTINGS_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training on {len(dataset)} paintings for {NUM_EPOCHS} epochs")

# Training loop
for epoch in range(NUM_EPOCHS):
    pipeline_losses = []
    
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            # Cast input to the same dtype as VAE
            pixel_values = batch["pixel_values"].to(device).to(torch_dtype)
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, len(pipe.scheduler), (latents.shape[0],), device=device)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(
                pipe.tokenizer(["a painting in the style of mom"], 
                              max_length=77, padding=True, 
                              return_tensors="pt").input_ids.to(device)
            )[0]
        
        model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred, noise, reduction="mean")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pipeline_losses.append(loss.item())
        
        if batch_idx % 5 == 0:
            avg_loss = sum(pipeline_losses[-5:]) / min(5, len(pipeline_losses))
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}")

# Save LoRA
print("Saving LoRA model...")
pipe.unet.save_pretrained(str(OUTPUT_DIR))
print(f"✓ LoRA saved to {OUTPUT_DIR}")

# Save config
import json
config = {
    "model": BASE_MODEL,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_images": len(dataset),
}
with open(OUTPUT_DIR / "training_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✓ Training complete!")
print(f"✓ Download the lora_output_kaggle_style_aware folder from /kaggle/working")
