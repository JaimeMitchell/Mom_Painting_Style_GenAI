#!/usr/bin/env python3
"""
Kohya-SS Style-Aware Training for mom_art
Optimized for high-diversity datasets (38 images across multiple mediums).
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    UNet2DConditionModel, 
    AutoencoderKL, 
    DDPMScheduler
)
from peft import LoraConfig, get_peft_model
import json
from tqdm.auto import tqdm

# ========================
# PROFESSIONAL PER-IMAGE CAPTIONS
# ========================
IMAGE_CAPTIONS = {
    "Brendas_Garden.jpg": "mom_art, Acrylic on Canvas,lush garden with layered cool and warm tones, vibrant multicolors, expressive brushwork, sunlit flowers with dark backgrounds",
    "Vitos_Garden.jpg": "mom_art, Acrylic on Canvas, lush garden with layered cool and warm tones, vibrant multicolors, sunlit flowers , dynamic brushwork, uplifting mood",
    "Hopes_Garden.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, vibrant garden scene with warm red, orange, pink and cool greens, purples, layered collage elements, dripping paint, flowers contrast against white background",
    "In_The_Light.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract landscape with cool greys, blues and greens, small spash of warm orange-red",
    "Leslies_Garden.jpg": "mom_art, Acrylic on Canvas, lush garden with layered cool and warm toned abstract landscape, vibrant multicolors, sunlit flowers , bright background, dynamic brushwork",
    "Stony_Creek_Trail_1.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract creek landscape, soft teal-blue water, warm brown, gray green stones and creekbed, misty, forest green foliage in hazy background",
    "Stony_Creek_Trail_2.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract creek landscape, warm brown landscape with hints of yellow-green moss and soft teal-blue and purple-blue water, gray green moss stones and creekbed, hazy background",
    "Stony_Creek_Trail_3.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract creek landscape, soft teal, blue and gray water and sky, warm brown, gray green stones and creekbed, misty, dark forest-green hazy background",
    "Sunflower.jpg": "mom_art, Ecoprint on Watercolor Paper, close-up sunflowers, bold yellow petals,blue middle textured, contrast againt white background",
    "Tide_Pool_1.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, organic forms of tide pool, cool blues and greens, delicate brushwork",
    "SunflowerandCorepsis.jpg": "mom_art, Ecoprint on Watercolor Paper, close-up sunflowers and coreopsis, bold yellow and red petals, with cool blues and greens, contrast against white background",
    "The_Three_Sisters.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract of corn, pumpkin and sunflower, vibrant multicolor, multi layered collage elements, contrast against white background",
    "Thinking_of_Charlie.jpg": "mom_art, Acrylic Collage with Handmade Paper on Boards, abstract mountain winter landscape with layered cool and warm tones, vibrant multicolors, sunlit flowers, dynamic brushwork, fine and corse tree forms, various moody textures",
    "balance-2.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract underwater landscape with layered cool tones blues, greens,against white background, organic forms",
    "flatcreek.jpg": "mom_art,  Ink on Watercolor Paper, abstract creek landscape with layered in vibrant cool and warm tones, organic forms",
    "joes_garden.jpg": "mom_art, Ink Collage on Yupo Paper, layered cool tones of blue, green, purple with subtle highlights of pink and orange, emotional brushwork, bright background",
    "poppies_on_trail.jpg": "mom_art, Acrylic & Ink Collage on Wood Cradle, vibrant red-orange poppies with layered cool greens and blues, dynamic brushwork, contrast against blue sky and purple mountains in far distant background",
    "willowcreek.jpg": "mom_art,Soft Pastel on Paper, abstract creek landscape with layered cool blues and greens, warm brown, gray green stones and creekbed, misty, forest green and purple mountain hazy background",
    "treeo-1.jpg": "mom_art,  Mixed Media Collage on Paper, abstract tree landscape with layered cool blues and greens, multitoned and colored trees, gray green mossy stone and earth, misty forest green hazy background",
    "treeo-2.jpg": "mom_art,  Mixed Media Collage on Paper, abstract tree landscape with layered cool blues and greens, multitoned and colored trees, gray green mossy stone and earth, misty forest green hazy background",
    "sleeping_indian.jpg": "mom_art, Oil and Pastel on Canvas, triptych of sleeping indian mountain landscape, layered cool and warm colors and tones. vibrant multicolors, sunlit, abstract, loose and soft brushwork",
    "westwoods.jpg": "mom_art, Soft Pastel on Paper, moody landscape with warm foreground of trees in sunlight, layered with  cool blues and purple darkbackground of forest, strong contrast between light and dark, expressive strong strokes.",
    "wilson_dike.jpg": "mom_art, Acrylic Collage with Handmade Paper on Board, abstract water landscape with layered cool and warm tones, vibrant multicolors, abstract reflections in water, expressive strokes, contrasting tones and colors",
    "dream-field.jpg": "mom_art, watercolor on paper, abstract floral, vibrant multicolors, sunlit, dynamic brushwork, contrast against white background",
    "acrylicpour-1.jpg": "mom_art, Acrylic Pour on Canvas, abstract, bold vibrant multicolors, dynamic organic forms",
    "acrylicpour-2.jpg": "mom_art, Acrylic Pour on Canvas, abstract, bold vibrant multicolors, dynamic organic forms",
    "acrylicpour-3.jpg": "mom_art, Acrylic Pour on Canvas, abstract, bold vibrant multicolors, dynamic organic forms",
    "ecoprint-1.jpg": "mom_art, Ecoprint on Watercolor Paper, abstract floral, vibrant multicolors, sunlit, dynamic brushwork, contrast against white background",
    "ecoprint-2.jpg": "mom_art, Ecoprint on Watercolor Paper, abstract floral, vibrant multicolors, sunlit, dynamic brushwork, contrast against white background",
    "ecoprint-3.jpg": "mom_art, Ecoprint on Watercolor Paper, abstract floral, vibrant multicolors, sunlit, dynamic brushwork, contrast against white background",
    "ecoprint-4.jpg": "mom_art, Ecoprint on Watercolor Paper, abstract floral, vibrant multicolors, sunlit, dynamic brushwork, contrast against white background",
    "inklandscape-1.jpg": "mom_art, Ink on Yupo Paper, abstract landscape with layered bright multicolors, dynamic brushwork, contrast against white background",
    "inklandscape-2.jpg": "mom_art, Ink on Yupo Paper, abstract landscape with layered bright multicolors, dynamic brushwork, contrast against white background",
    "inklandscape-3.jpg": "mom_art, Ink on Yupo Paper, abstract landscape with layered bright multicolors, dynamic brushwork, contrast against white background",
    "inklandscape-4.jpg": "mom_art, Ink on Yupo Paper, abstract landscape with layered bright multicolors, dynamic brushwork, contrast against white background"
}

FALLBACK_CAPTION = "mom_art, expressive abstract landscape, vibrant colors, artistic texture"

# ========================
# DATASET CLASS
# ========================
class KohyaStyleDataset(Dataset):
    def __init__(self, image_dir, vae, device):
        self.files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.vae = vae
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = os.path.basename(file_path)
        caption = IMAGE_CAPTIONS.get(file_name, FALLBACK_CAPTION)
        
        image = Image.open(file_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = latents.squeeze(0)
            
        return {"pixel_values": latents, "caption": caption}

# ========================
# TRAINING ENGINE
# ========================
def main():
    # Setup
    BASE_MODEL_ID = "./stable-diffusion-v1-5"
    OUTPUT_DIR = "./lora_mom_art_final"
    EPOCHS = 150 # Higher epochs to handle diverse styles
    LORA_RANK = 128 # High rank captures distinct mediums (pours vs prints)
    LORA_ALPHA = 128
    LEARNING_RATE = 5e-5
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"🔄 Preparing environment on {DEVICE}...")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet").to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL_ID).to(DEVICE)
    text_encoder, tokenizer = pipe.text_encoder, pipe.tokenizer

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none"
    )
    lora_model = get_peft_model(unet, lora_config)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)

    dataset = KohyaStyleDataset("./Paintings", vae, DEVICE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"🚀 Starting High-Breadth Training (38 images, {EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        lora_model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            latents = batch["pixel_values"].to(DEVICE)
            caption = batch["caption"][0]
            
            # Tokenize & Encode Text
            inputs = tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(DEVICE)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            # Diffusion Logic
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=DEVICE).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Backprop
            noise_pred = lora_model(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save frequent checkpoints to find the "sweet spot"
        if (epoch + 1) % 25 == 0:
            lora_model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}"))

    lora_model.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training Finished. Final LoRA saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()