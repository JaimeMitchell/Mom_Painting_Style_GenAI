# Mom's Painting Style GenAI - Architecture Walkthrough

## System Overview

This is a **local ML application** for training and using custom LoRA (Low-Rank Adaptation) models to generate images in a specific artistic style. It's not a traditional web application with databases—it's a self-contained system for style-aware image generation.

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Gradio Web UI)                 │
│  - Text input for image description                         │
│  - Sliders for generation parameters (steps, guidance)      │
│  - Real-time image generation and display                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND (Python ML Pipeline)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Stable Diffusion v1.5 (Base Model)                 │   │
│  │  - Text encoding                                    │   │
│  │  - Denoising diffusion process                      │   │
│  │  - Image decoding                                   │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │                                       │
│  ┌──────────────────▼──────────────────────────────────┐   │
│  │  LoRA Adapter (Trained Model)                       │   │
│  │  - Style-specific weights                           │   │
│  │  - Applied to UNet during generation                │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          DATA LAYER (Local Filesystem)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ./Paintings/          (Training data - 38 images)   │  │
│  │  ./stable-diffusion-v1-5/  (Base model weights)      │  │
│  │  ./lora_output_kohya_style_aware/  (Trained LoRA)    │  │
│  │  ./lora_output_style_aware/  (Previous LoRA)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. FRONTEND LAYER - Gradio Interface

**File**: `final_working_interface.py`

**What it does**:
- Provides a simple web UI for image generation
- Converts user input into API-ready data
- Displays generated images in real-time

**User Interface Elements**:
```
┌──────────────────────────────────────────────┐
│  mom's Art Generator                         │
├──────────────────────────────────────────────┤
│  [Text Input: "describe what you want..."]   │
│  [Steps: 10-50 slider, default 25]          │
│  [Guidance Scale: 1-15 slider, default 7.5] │
│  [Seed: optional number for reproducibility]│
│  [Generate Button]                           │
├──────────────────────────────────────────────┤
│  [Output Image - Generated in real-time]     │
└──────────────────────────────────────────────┘
```

**Technology**: Gradio 6.14.0 (fast Python → web UI)

---

### 2. BACKEND LAYER - ML Pipeline

The backend consists of three integrated components:

#### A. Model Loading & Initialization
```python
# Load base Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5"
)

# Load trained LoRA adapter on top
pipe.unet = PeftModel.from_pretrained(
    pipe.unet, 
    "./lora_output_kohya_style_aware"
)
```

**What happens**:
1. Base Stable Diffusion loads (3.44 GB model)
2. LoRA weights are dynamically fused into the UNet
3. Memory efficient - LoRA adds only ~50MB of custom weights
4. Model runs on M1 Metal Performance Shaders (MPS) for GPU acceleration

#### B. Image Generation Function
```python
def generate_image(prompt, steps=25, guidance=7.5, seed=None):
    # Seed ensures reproducibility
    if seed:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(random.randint(0, 999999))
    
    # Generate with style
    result = pipe(
        prompt,
        num_inference_steps=steps,      # More steps = better quality
        guidance_scale=guidance,         # Higher = follow prompt more strictly
        width=512,
        height=512,
        generator=generator
    ).images[0]
    
    return result
```

**Data Flow - How an Image is Generated**:
```
User Prompt (text)
    ↓
Tokenize (convert text to embeddings)
    ↓
Encode prompt with CLIP text encoder
    ↓
Start with random noise (512×512)
    ↓
[DENOISING LOOP - 25 steps]
    For each step:
    - UNet predicts noise to remove
    - LoRA adapter influences prediction with style
    - Remove predicted noise from image
    - Use guidance scale to weight prompt influence
    ↓
VAE Decoder (convert latent space to pixels)
    ↓
Output Image (512×512)
```

---

### 3. DATA LAYER - File System Storage

Unlike traditional web apps with databases, this system uses the **local file system** as its data store:

#### Training Data
```
./Paintings/
├── painting_001.jpg    (Training example)
├── painting_002.jpg    (Training example)
├── ... (38 total images)
└── painting_038.jpg    (Training example)
```

**Role**: Source material for style learning

#### Model Artifacts
```
./stable-diffusion-v1-5/    (4.27 GB)
├── text_encoder/
│   └── model.safetensors
├── unet/
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json

./lora_output_kohya_style_aware/
├── adapter_config.json      (LoRA architecture definition)
├── adapter_model.safetensors (Trained style weights - ~50MB)
└── training_logs/
    └── loss_history.json
```

**Role**: 
- Base model: Foundation for all generation
- LoRA: Custom weights trained on 38 paintings

---

## KEY FEATURE: LoRA Training System

### What is LoRA?
**Low-Rank Adaptation** = Train only a small set of additional weights (~0.1% of model) instead of all 1.2B parameters.

### How the Training Pipeline Works

**File**: `kohya_ss_style_aware_complete.py`

#### Step 1: Style Analysis
```
Input: 38 paintings from ./Paintings/
    ↓
[analyze_mom_style.py analyzes each image]
    - Average RGB color: (165.9, 165.4, 151.6)
    - Average brightness: 161.0
    - Warm red palette detected
    - Garden/landscape focus identified
    ↓
Output: Style profile (used for caption generation)
```

#### Step 2: Dataset Preparation
```
For each painting:
    ↓
[Read image, resize to 512×512]
    ↓
[Encode through VAE to latent space]
    ↓
[Assign style-specific caption]
    Examples:
    - "mom_art, warm red color palette, light paintings"
    - "mom_art, garden and landscape focus"
    - "mom_art, soft brushwork, consistent style"
    ↓
Output: Training dataset (38 samples × 18 captions = 684 training examples)
```

#### Step 3: LoRA Training Loop
```
For each epoch (15 total):
    For each training example:
        1. Load encoded image (latent)
        2. Add random noise (simulating diffusion)
        3. Forward through Stable Diffusion + LoRA
        4. Calculate loss (difference between predicted and actual noise)
        5. Backprop through LoRA weights ONLY (not base model)
        6. Update LoRA with learning rate 8e-5
    ↓
    Save checkpoint if loss improved
    ↓
Output: Trained LoRA checkpoint (./lora_output_kohya_style_aware/)
```

#### Key Hyperparameters:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| LORA_RANK | 20 | Complexity of style learning |
| LORA_ALPHA | 40 | Influence strength of LoRA |
| LEARNING_RATE | 8e-5 | How much weights change each step |
| EPOCHS | 15 | How many times to iterate through data |
| BATCH_SIZE | 1 | Images processed at once (memory efficient) |

---

## Data Flow: Complete Architecture

### Training Flow
```
┌─────────────┐
│  Paintings  │  (38 images)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  analyze_mom_style.py               │
│  - Analyze colors, brightness       │
│  - Detect style characteristics     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  kohya_ss_style_aware_complete.py   │
│  ┌─────────────────────────────┐   │
│  │ 1. Load base model          │   │
│  │ 2. Prepare dataset          │   │
│  │ 3. Add LoRA to UNet         │   │
│  │ 4. Train for 15 epochs      │   │
│  │ 5. Save best checkpoint     │   │
│  └─────────────────────────────┘   │
└──────┬──────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  ./lora_output_kohya_style_aware/        │
│  ├── adapter_config.json                 │
│  └── adapter_model.safetensors (~50MB)   │
└──────────────────────────────────────────┘
```

### Inference (Generation) Flow
```
┌──────────────────┐
│  User Input      │
│  "warm garden"   │
│  Steps: 25       │
│  Guidance: 7.5   │
│  Seed: 42        │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  final_working_interface.py          │
│  - Parse inputs                      │
│  - Validate parameters               │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Stable Diffusion Pipeline           │
│  ┌──────────────────────────────┐   │
│  │ 1. Text → CLIP embeddings    │   │
│  │ 2. Random noise (512×512)    │   │
│  │ 3. Denoising loop (25 steps) │   │
│  │    - UNet predicts noise     │   │
│  │    - LoRA influences output  │   │
│  │    - Remove noise            │   │
│  │    - Apply guidance          │   │
│  │ 4. VAE decode latents        │   │
│  │ 5. Output RGB image          │   │
│  └──────────────────────────────┘   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Generated Image             │
│  (512×512, PNG)              │
│  Sent to Gradio UI           │
└──────────────────────────────┘
```

---

## How Components Communicate

### Frontend → Backend
```
User fills Gradio form
    ↓
JavaScript submits HTTP POST
    ↓
Gradio server receives request
    ↓
Calls Python generate_image(prompt, steps, guidance, seed)
    ↓
Backend processes
    ↓
Returns PIL Image object
    ↓
Gradio converts to PNG
    ↓
Sends back to browser for display
    ↓
User sees result instantly
```

### Backend → File System
```
ML Pipeline needs:
    ├─ Base model weights? Load from ./stable-diffusion-v1-5/
    ├─ LoRA? Load from ./lora_output_kohya_style_aware/
    └─ Training data? Read from ./Paintings/

During training:
    └─ Save checkpoints to ./lora_output_kohya_style_aware/
```

---

## State Management

### What is Stateful?
- **Model Pipeline** - Loaded once at startup, stays in GPU memory
- **LoRA Weights** - Fused into UNet, persistent during session

### What is Stateless?
- **Each Generation** - Independent request, doesn't affect next one
- **User Input** - No session tracking, no history saved

### Reproducibility
- **Seed Parameter** - Ensures same prompt + seed = same image
- **Deterministic** - Same inputs always produce same output

---

## Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web Framework | Gradio | 6.14.0 | Frontend interface |
| ML Framework | PyTorch | 2.11.0 | Neural network compute |
| Stable Diffusion | Diffusers | 0.38.0 | Diffusion model implementation |
| LoRA | PEFT | 0.19.1 | Parameter-efficient fine-tuning |
| Image Processing | Pillow | 12.2.0 | Image I/O |
| Distributed Training | Accelerate | 1.13.0 | Multi-GPU support (future) |
| Hardware | Apple M1 | - | MPS GPU acceleration |

---

## Key Files & Responsibilities

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `final_working_interface.py` | Web UI & inference | User text prompt | Generated image |
| `analyze_mom_style.py` | Style analysis | 38 paintings | Style metrics (RGB, brightness) |
| `kohya_ss_style_aware_complete.py` | Training | Paintings + style data | Trained LoRA checkpoint |
| `kohya_ss_style_aware_training.py` | Advanced training | Same + distributed setup | Trained LoRA checkpoint |

---

## Memory & Performance

### Model Sizes
- Base Stable Diffusion: 4.27 GB
- LoRA Adapter: ~50 MB (0.001× base model)
- VRAM during generation: ~2.5 GB (M1 GPU buffer)

### Generation Speed
- Steps: 10-50 (default 25)
- Time per generation: 30-60 seconds on M1
- Bottleneck: Denoising iterations, not LoRA overhead

### Training Performance
- 38 paintings × 18 captions = 684 examples
- 15 epochs = 10,260 training steps
- ~2 seconds per step = ~5 hours total training time

---

## Future Architecture Improvements

1. **Database Layer** (Optional)
   - Store generation history
   - Track training metrics
   - Version control for LoRA checkpoints

2. **API Server** (Optional)
   - REST API for external integrations
   - Batch processing queue
   - Multi-user support

3. **Distributed Training**
   - Multi-GPU training via Accelerate
   - Larger datasets support

4. **Model Optimization**
   - Quantization (reduce VRAM)
   - Export to ONNX (cross-platform)
   - Real-time LoRA switching

---

## Summary

**This system = Style Transfer + Image Generation**

- **Training Phase**: Learn style from paintings → Generate LoRA weights
- **Inference Phase**: User prompt + LoRA → Generated image in target style
- **No database**: Everything is file-based and self-contained
- **Single-user**: Designed for local machine (not production web app)
- **Reproducible**: Seed-based deterministic generation
