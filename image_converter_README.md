# Image Converter - Detailed Documentation

## Overview
This script provides a flexible image preprocessing tool for batch converting images with options for resizing and color mode conversion. It's designed to prepare images for statistical analysis by standardizing their format and dimensions.

---

## Line-by-Line Code Explanation

### Module Documentation (Lines 1-22)
```python
"""
Image Converter Script
...
"""
```
**What it does:** Provides comprehensive documentation for the script usage, arguments, and behavior.

**Theory:** Good documentation is essential for reproducibility in scientific computing. This docstring follows Python PEP 257 conventions.

---

### Import Statements (Lines 24-28)
```python
import argparse
import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
```

**Line-by-line:**
- **Line 24:** `argparse` - Parses command-line arguments (user inputs)
- **Line 25:** `os` - Provides operating system functionality (file operations)
- **Line 26:** `sys` - System-specific parameters (exit codes)
- **Line 27:** `Path` from `pathlib` - Modern, object-oriented file path handling
- **Line 28:** `Image, ImageOps` from `PIL` - Python Imaging Library for image manipulation

**Theory:** These are standard Python libraries for CLI applications and image processing. PIL (Pillow) provides efficient image I/O and transformations.

---

### Function: convert_image_to_bw (Lines 31-33)
```python
def convert_image_to_bw(image):
    """Convert image to black and white (grayscale)."""
    return image.convert('L')
```

**What it does:** Converts a color image to grayscale (black and white).

**Theory:** 
- Grayscale conversion reduces dimensionality from 3 channels (RGB) to 1 channel (luminance)
- The 'L' mode in PIL represents 8-bit pixels (0-255 intensity values)
- This simplifies statistical analysis by reducing feature space from 3D to 1D per pixel

**Mathematical basis:** Grayscale = 0.299×R + 0.587×G + 0.114×B (weighted average, perception-based)

---

### Function: resize_image (Lines 36-103)
```python
def resize_image(image, target_width, target_height, mode='fit', keep_color=False):
```

**Purpose:** Resizes images using different strategies to meet target dimensions.

#### Mode: 'stretch' (Lines 53-55)
```python
if mode == 'stretch':
    return image.resize((target_width, target_height), Image.LANCZOS)
```

**What it does:** Ignores aspect ratio and forces image to exact dimensions.

**Theory:** 
- LANCZOS is a high-quality resampling filter (windowed sinc function)
- It provides better quality than bilinear or nearest-neighbor interpolation
- Mathematical basis: Uses sinc function with a=3 window for interpolation

**When to use:** When exact dimensions are critical and distortion is acceptable.

---

#### Mode: 'fit' (Lines 57-60)
```python
elif mode == 'fit':
    image.thumbnail((target_width, target_height), Image.LANCZOS)
    return image
```

**What it does:** Maintains aspect ratio, scales down to fit within dimensions.

**Theory:**
- Thumbnail operation: scales = min(target_width/width, target_height/height)
- Preserves aspect ratio by using the smaller scaling factor
- Never enlarges the image (only shrinks)

**When to use:** Preserving aspect ratio is critical, and smaller dimensions are acceptable.

---

#### Mode: 'contain' (Lines 62-75)
```python
elif mode == 'contain':
    image.thumbnail((target_width, target_height), Image.LANCZOS)
    image_mode = 'RGB' if keep_color else 'L'
    bg_color = (255, 255, 255) if keep_color else 255
    new_image = Image.new(image_mode, (target_width, target_height), color=bg_color)
    x = (target_width - image.width) // 2
    y = (target_height - image.height) // 2
    new_image.paste(image, (x, y))
    return new_image
```

**What it does:** Fits image within dimensions and adds white padding (letterboxing/pillarboxing).

**Theory:**
- First scales image to fit (like 'fit' mode)
- Creates canvas with exact target dimensions
- Centers the scaled image on white background
- Centering formula: offset = (target_dim - image_dim) / 2

**When to use:** Need exact output dimensions while preserving aspect ratio and context.

---

#### Mode: 'cover' (Lines 77-98)
```python
elif mode == 'cover':
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height
    
    if img_ratio > target_ratio:
        # Image is wider, scale by height
        scale_factor = target_height / image.height
        new_width = int(image.width * scale_factor)
        image = image.resize((new_width, target_height), Image.LANCZOS)
        left = (new_width - target_width) // 2
        image = image.crop((left, 0, left + target_width, target_height))
    else:
        # Image is taller, scale by width
        scale_factor = target_width / image.width
        new_height = int(image.height * scale_factor)
        image = image.resize((target_width, new_height), Image.LANCZOS)
        top = (new_height - target_height) // 2
        image = image.crop((0, top, target_width, top + target_height))
    
    return image
```

**What it does:** Scales and crops to fill exact dimensions while maintaining aspect ratio.

**Theory:**
1. **Aspect ratio comparison:** Determines which dimension to prioritize
   - If image is wider than target: scale to height, crop width
   - If image is taller than target: scale to width, crop height
2. **Scaling factor:** Ensures one dimension matches exactly
3. **Center cropping:** Removes excess from center to maintain focal point

**Mathematical basis:**
- Aspect ratio = width / height
- Scale factor = target_dim / source_dim (for matched dimension)
- Crop offset = (scaled_dim - target_dim) / 2

**When to use:** Need exact dimensions, aspect ratio preservation, and can afford to lose edge content.

---

### Function: process_images (Lines 106-184)
```python
def process_images(folder_path, target_width=None, target_height=None, 
                   mode='fit', keep_color=False, prefix='converted'):
```

**Purpose:** Batch processes all images in a folder with specified transformations.

#### Supported Formats (Lines 118-119)
```python
supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
```

**Theory:** These are common raster image formats. Set data structure provides O(1) lookup time.

---

#### File Filtering (Lines 121-124)
```python
image_files = [f for f in folder_path.iterdir() 
               if f.is_file() and f.suffix.lower() in supported_formats
               and not f.stem.startswith(f"{prefix}_")]
```

**What it does:** 
- Iterates through directory
- Filters for supported image formats
- Excludes already-processed files (those with prefix)

**Theory:** List comprehension provides Pythonic filtering. The prefix check prevents reprocessing.

---

#### Image Processing Loop (Lines 147-169)
```python
for image_file in image_files:
    try:
        with Image.open(image_file) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            if target_width and target_height:
                resized_img = resize_image(img, target_width, target_height, mode, keep_color)
            else:
                resized_img = img
            
            if not keep_color:
                final_img = convert_image_to_bw(resized_img)
            else:
                final_img = resized_img
            
            output_path = folder_path / f"{prefix}_{image_file.stem}.png"
            final_img.save(output_path, 'PNG')
```

**Processing pipeline:**
1. **Open image** with context manager (automatic cleanup)
2. **Mode normalization:** Convert RGBA, CMYK, etc. to RGB
3. **Resize** (if dimensions specified)
4. **Convert to B&W** (if color not preserved)
5. **Save** with prefix to avoid overwriting originals

**Theory:**
- Context manager (`with`) ensures proper resource cleanup
- Pipeline architecture allows modular transformations
- PNG format is lossless, suitable for further processing

---

### Function: main (Lines 187-275)
```python
def main():
    parser = argparse.ArgumentParser(...)
```

**Purpose:** Handles command-line interface, validates inputs, and orchestrates processing.

#### Argument Parsing (Lines 187-247)
Creates CLI with arguments:
- `--folder`: Required, specifies input directory
- `--width`, `--height`: Optional dimensions
- `--mode`: Resize strategy (fit, contain, stretch, cover)
- `--color`: Flag to preserve color
- `--prefix`: Output filename prefix

**Theory:** `argparse` provides automatic help generation, type validation, and user-friendly error messages.

---

#### Input Validation (Lines 250-261)
```python
if args.width is not None and args.width <= 0:
    print("Error: Width must be a positive integer")
    sys.exit(1)

if (args.width is None) != (args.height is None):
    print("Error: Both width and height must be specified together, or neither")
    sys.exit(1)
```

**What it does:** 
- Ensures dimensions are positive
- Ensures both dimensions specified together (XOR check)

**Theory:** Input validation prevents runtime errors and provides clear user feedback. Exit code 1 indicates error to shell.

---

#### Path Handling (Lines 263-273)
```python
script_dir = Path(__file__).parent
folder_path = script_dir / args.folder

if not folder_path.exists():
    print(f"Error: Folder '{args.folder}' not found at {folder_path}")
    sys.exit(1)
```

**What it does:** 
- Constructs absolute path from script location
- Validates folder exists and is a directory

**Theory:** Using `Path` objects provides platform-independent path handling (works on Windows, Linux, macOS).

---

## Statistical and Image Processing Theory

### Why Image Preprocessing Matters

1. **Standardization:** ML/statistical models require consistent input dimensions
2. **Dimensionality reduction:** Grayscale reduces features by 67% (3 channels → 1)
3. **Computational efficiency:** Smaller images = faster processing
4. **Noise reduction:** Resampling can act as low-pass filter

### Resampling Theory

**LANCZOS Resampling:**
- Based on sinc function: sinc(x) = sin(πx)/(πx)
- Window size: a=3 (uses 3 lobes on each side)
- Provides excellent quality but computationally expensive
- Minimizes aliasing and ringing artifacts

**Quality ranking:** LANCZOS > BICUBIC > BILINEAR > NEAREST

### Grayscale Conversion

**Luminance formula (ITU-R BT.601):**
```
Y = 0.299×R + 0.587×G + 0.114×B
```

**Why these weights?**
- Human eyes are most sensitive to green (57%)
- Least sensitive to blue (11%)
- Perceptually uniform grayscale

---

## Usage Examples

### Basic conversion (B&W, original size)
```bash
python image_converter.py --folder dataset/raw
```

### Resize to 178x218 (project standard)
```bash
python image_converter.py --folder dataset/raw --width 178 --height 218 --mode contain
```

### Keep color, resize with cropping
```bash
python image_converter.py --folder dataset/raw --width 512 --height 512 --mode cover --color
```

### Custom prefix
```bash
python image_converter.py --folder dataset/raw --width 178 --height 218 --prefix processed
```

---

## Best Practices

1. **Always backup originals** - Script creates new files, but be careful
2. **Use 'contain' mode** - Preserves aspect ratio with exact dimensions
3. **Choose appropriate format:**
   - PNG: Lossless, larger files (for further processing)
   - JPEG: Lossy, smaller files (for final storage)
4. **Batch process consistently** - Use same parameters for all images in a dataset

---

## Common Issues

**Issue:** "No images found"
- **Solution:** Check folder path, ensure images have supported extensions

**Issue:** Memory error with large images
- **Solution:** Process images in smaller batches, use lower resolution

**Issue:** Quality loss after resize
- **Solution:** Ensure using LANCZOS resampling (default in this script)

---

## Output Specifications

- **Format:** PNG (lossless)
- **Bit depth:** 8-bit grayscale (L mode) or 24-bit RGB
- **Naming:** `{prefix}_{original_name}.png`
- **Location:** Same directory as source images

---

## Integration with Project

This script is the **first step** in the analysis pipeline:

1. **image_converter.py** ← You are here (data preparation)
2. pca_dataset.py (unsupervised dimensionality reduction)
3. lda_dataset.py (supervised classification)
4. regression_analysis.py (statistical modeling)

**Prepared images are used as input for PCA and LDA analyses.**
