# Image Converter

A Python script to convert images to smaller resolutions and black & white format.

## Features

- Converts all images in a specified folder to grayscale (black & white) by default
- Optional flag to keep images in color
- Resizes images to custom dimensions (optional - can keep original resolution)
- Multiple resize modes to handle aspect ratio:
  - **fit**: Maintain aspect ratio, resize to fit within dimensions (default)
  - **contain**: Maintain aspect ratio, fit inside dimensions with padding
  - **stretch**: Ignore aspect ratio, stretch to exact dimensions
  - **cover**: Maintain aspect ratio, crop to fill dimensions
- Supports multiple image formats: JPG, PNG, BMP, GIF, TIFF, WEBP
- Saves converted images in the same folder with customizable prefix
- Default prefix is "converted_", can be customized via `--prefix` parameter
- Automatically skips already processed files to avoid reprocessing

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (with resizing)
```bash
python image_converter.py --folder <folder_name> --width <width> --height <height>
```

### Convert to B&W Without Resizing (keep original resolution)
```bash
python image_converter.py --folder <folder_name>
```

### With Resize Mode
```bash
python image_converter.py --folder <folder_name> --width <width> --height <height> --mode <mode>
```

### Keep Images in Color
```bash
python image_converter.py --folder <folder_name> --width <width> --height <height> --color
```

### Keep Original Resolution and Color
```bash
python image_converter.py --folder <folder_name> --color
```

### Custom Prefix for Output Files
```bash
python image_converter.py --folder <folder_name> --width 800 --height 600 --prefix thumbnail
```

## Examples

### Example 1: Resize to 800x600 with fit mode (default, black & white)
```bash
python image_converter.py --folder images --width 800 --height 600
```

### Example 2: Resize to 1920x1080 with contain mode (letterbox/pillarbox)
```bash
python image_converter.py --folder photos --width 1920 --height 1080 --mode contain
```

### Example 3: Resize to 512x512 with cover mode (crop to fill)
```bash
python image_converter.py --folder pics --width 512 --height 512 --mode cover
```

### Example 4: Resize to exact dimensions with stretch mode
```bash
python image_converter.py --folder wallpapers --width 1024 --height 768 --mode stretch
```

### Example 5: Keep images in color (no black & white conversion)
```bash
python image_converter.py --folder vacation --width 1920 --height 1080 --color
```

### Example 6: Color images with cover mode
```bash
python image_converter.py --folder portraits --width 800 --height 800 --mode cover --color
```

### Example 7: Convert to black & white only (no resizing)
```bash
python image_converter.py --folder documents
```

### Example 8: Keep original resolution and color (no conversion)
```bash
python image_converter.py --folder originals --color
```

### Example 9: Custom prefix for output files
```bash
python image_converter.py --folder images --width 800 --height 600 --prefix thumbnail
```
This will create files like: `thumbnail_photo1.png`, `thumbnail_photo2.png`, etc.

### Example 10: Convert to B&W with custom prefix
```bash
python image_converter.py --folder documents --prefix bw
```
This will create files like: `bw_document1.png`, `bw_document2.png`, etc.

## Resize Modes Explained

- **fit**: The image is resized to fit within the specified dimensions while maintaining its aspect ratio. The resulting image may be smaller than the target dimensions.

- **contain**: The image is resized to fit within the specified dimensions while maintaining its aspect ratio, then centered on a white background to match the exact target dimensions.

- **stretch**: The image is resized to exactly match the target dimensions, ignoring the original aspect ratio. This may distort the image.

- **cover**: The image is resized to fill the target dimensions while maintaining its aspect ratio. Parts of the image may be cropped to achieve this.

## Folder Structure

The script expects:
- A folder with images at the same level as the script
- Converted images are saved in the same folder with a prefix

Example:
```
project/
├── image_converter.py
├── images/
│   ├── photo1.jpg
│   ├── photo2.png
│   ├── converted_photo1.png    (created after conversion)
│   └── converted_photo2.png    (created after conversion)
```

With custom prefix (e.g., `--prefix thumbnail`):
```
project/
├── image_converter.py
├── images/
│   ├── photo1.jpg
│   ├── photo2.png
│   ├── thumbnail_photo1.png    (created after conversion)
│   └── thumbnail_photo2.png    (created after conversion)
```

## Output

- All converted images are saved as PNG files in the same folder
- Original filenames are preserved with a customizable prefix (default: `converted_`)
- Files with the prefix are automatically excluded to avoid reprocessing
- Progress is displayed in the console
- Summary shows successful and failed conversions

## Important Notes

- **Prefix Format**: The prefix is added with an underscore separator (e.g., `converted_photo.png`)
- **Reprocessing Protection**: Files that already have the prefix are automatically skipped
- **Same Folder Output**: Converted images are saved in the same folder as the originals
- **PNG Format**: All output images are saved as PNG regardless of the input format
