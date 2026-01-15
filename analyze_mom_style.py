#!/usr/bin/env python3
"""
Analyze mom's paintings to understand the actual style
This will help create better training data and approaches
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Analyze the actual paintings
paintings_dir = "./Ma_React_App/Paintings"
analysis_results = {}

print("🔍 ANALYZING mom'S ACTUAL STYLE...")

def analyze_painting_style(image_path):
    """Analyze individual painting characteristics"""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Basic statistics
    mean_color = np.mean(img_array, axis=(0, 1))
    std_color = np.std(img_array, axis=(0, 1))
    
    # Color characteristics
    r_mean, g_mean, b_mean = mean_color
    brightness = np.mean(img_array)
    
    # Dominant colors (simplified)
    dominant_colors = []
    for i in range(0, 256, 32):  # Sample every 32 levels
        mask = np.all((img_array >= i) & (img_array < i+32), axis=2)
        if np.sum(mask) > 0:
            dominant_colors.append((i, np.sum(mask)))
    
    return {
        'mean_color': mean_color,
        'brightness': brightness,
        'dominant_colors': dominant_colors[:5],  # Top 5
        'size': img.size
    }

# Analyze all paintings
painting_files = [f for f in os.listdir(paintings_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"📊 Analyzing {len(painting_files)} paintings...")

for painting in painting_files:
    path = os.path.join(paintings_dir, painting)
    try:
        analysis = analyze_painting_style(path)
        analysis_results[painting] = analysis
        print(f"✅ {painting}: Brightness={analysis['brightness']:.1f}, RGB=({analysis['mean_color'][0]:.1f}, {analysis['mean_color'][1]:.1f}, {analysis['mean_color'][2]:.1f})")
    except Exception as e:
        print(f"❌ Error analyzing {painting}: {e}")

# Generate style summary
print("\n🎨 mom STYLE ANALYSIS:")
print("=" * 50)

if analysis_results:
    all_brightness = [r['brightness'] for r in analysis_results.values()]
    all_colors = [r['mean_color'] for r in analysis_results.values()]
    
    avg_brightness = np.mean(all_brightness)
    avg_color = np.mean(all_colors, axis=0)
    
    print(f"Average Brightness: {avg_brightness:.1f}")
    print(f"Average RGB: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")
    
    # Color palette analysis
    red_dominant = avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]
    green_dominant = avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]
    blue_dominant = avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]
    
    if red_dominant:
        print("🎨 Style: Warm/red-leaning color palette")
    elif green_dominant:
        print("🎨 Style: Green-leaning color palette") 
    elif blue_dominant:
        print("🎨 Style: Cool/blue-leaning color palette")
    else:
        print("🎨 Style: Balanced color palette")
    
    if avg_brightness > 150:
        print("💡 Brightness: Light/brighter paintings")
    elif avg_brightness > 100:
        print("💡 Brightness: Medium brightness")
    else:
        print("💡 Brightness: Darker paintings")

print(f"\n📋 FINDINGS:")
print(f"- {len(analysis_results)} paintings analyzed")
print(f"- This data should inform better training approaches")
print(f"- Need style-specific captions, not generic ones")
