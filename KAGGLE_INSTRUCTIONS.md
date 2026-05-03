# Running Training on Kaggle (FREE GPU - 10x FASTER)

## Why Kaggle?
- **Free GPU**: Tesla T4 or better
- **10x faster** than M1 MacBook for this training
- **No setup needed**: Just upload and run

## Steps:

### 1. **Upload to Kaggle**
   - Go to https://kaggle.com and sign in
   - Create new dataset
   - Upload folder: `Paintings/` (the 38 images)
   - Name it: `paintings`
   
### 2. **Create Kaggle Notebook**
   - New Notebook (Python)
   - Add your uploaded dataset as input
   - Add this repo as input if available, or copy the code below

### 3. **Paste + Run This Code in Kaggle Notebook**

```python
# Install dependencies (first cell)
!pip install -q diffusers peft transformers torch torchvision accelerate safetensors pillow

# Run training (second cell) - copy entire content of kaggle_training.py here
```

### 4. **Download Result**
   - After training completes, download `lora_output_kaggle_style_aware` folder
   - Extract to your local machine at: `./lora_output_kaggle_style_aware/`
   - Restart the interface - it will auto-load the new model

## File Structure on Kaggle:

```
/kaggle/input/
  ├── paintings/          (your 38 images)
  └── [this-repo]/        (optional - kaggle_training.py code)

/kaggle/working/
  └── lora_output_kaggle_style_aware/  (output - download this)
```

## Expected Training Time on Kaggle
- **T4 GPU**: ~45 minutes - 1.5 hours (15 epochs, 38 images)
- **V100 GPU**: ~20-30 minutes
- Compare to M1 local: ~10 hours ❌

## After Training
1. Download `lora_output_kaggle_style_aware` folder
2. Put it in your Project_Mom directory
3. Kill the local interface: `pkill -f final_working`
4. Restart: `source venv/bin/activate && python final_working_interface.py`
5. Test the improved style ✓

## Kaggle Notebook Template (Ready to Copy-Paste)

```python
# Cell 1: Install
!pip install -q diffusers peft transformers torch torchvision accelerate safetensors pillow

# Cell 2: Training (copy kaggle_training.py content)
import os
import torch
from pathlib import Path
...
[PASTE FULL kaggle_training.py CONTENT HERE]
```

That's it. Kaggle handles GPU, you just wait for email when done.
