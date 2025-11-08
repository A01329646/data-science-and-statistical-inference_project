"""
Image Converter Script
Converts all images in a folder with optional resizing and black & white conversion.

Usage:
    python image_converter.py --folder <folder_name> [--width <width>] [--height <height>] [--mode <mode>] [--color] [--prefix <prefix>]

Arguments:
    --folder: Name of the folder containing images (at same level as script)
    --width: Target width in pixels (optional, keeps original if not specified)
    --height: Target height in pixels (optional, keeps original if not specified)
    --mode: Resize mode - 'contain', 'stretch', 'fit', or 'cover'
            - contain: Maintain aspect ratio, fit inside dimensions (letterbox/pillarbox)
            - stretch: Ignore aspect ratio, stretch to exact dimensions
            - fit: Maintain aspect ratio, resize to fit within dimensions
            - cover: Maintain aspect ratio, crop to fill dimensions
    --color: Keep images in color (default: convert to black & white)
    --prefix: Prefix for output filenames (default: 'converted')
    
Note: If width and height are not provided, original resolution is maintained.
      Converted images are saved in the same folder with the prefix added to the filename.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image, ImageOps


def convert_image_to_bw(image):
    """Convert image to black and white (grayscale)."""
    return image.convert('L')


def resize_image(image, target_width, target_height, mode='fit', keep_color=False):
    """
    Resize image based on the specified mode.
    
    Args:
        image: PIL Image object
        target_width: Target width in pixels
        target_height: Target height in pixels
        mode: Resize mode ('contain', 'stretch', 'fit', 'cover')
        keep_color: Whether to keep the image in color
    
    Returns:
        Resized PIL Image object
    """
    if mode == 'stretch':
        # Ignore aspect ratio, stretch to exact dimensions
        return image.resize((target_width, target_height), Image.LANCZOS)
    
    elif mode == 'fit':
        # Maintain aspect ratio, fit within dimensions
        image.thumbnail((target_width, target_height), Image.LANCZOS)
        return image
    
    elif mode == 'contain':
        # Maintain aspect ratio, fit inside dimensions with padding
        image.thumbnail((target_width, target_height), Image.LANCZOS)
        # Create a new image with the target dimensions and paste the resized image
        # Use appropriate mode based on color setting
        image_mode = 'RGB' if keep_color else 'L'
        bg_color = (255, 255, 255) if keep_color else 255
        new_image = Image.new(image_mode, (target_width, target_height), color=bg_color)
        # Center the image
        x = (target_width - image.width) // 2
        y = (target_height - image.height) // 2
        new_image.paste(image, (x, y))
        return new_image
    
    elif mode == 'cover':
        # Maintain aspect ratio, crop to fill dimensions
        img_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Image is wider, scale by height
            scale_factor = target_height / image.height
            new_width = int(image.width * scale_factor)
            image = image.resize((new_width, target_height), Image.LANCZOS)
            # Crop width
            left = (new_width - target_width) // 2
            image = image.crop((left, 0, left + target_width, target_height))
        else:
            # Image is taller, scale by width
            scale_factor = target_width / image.width
            new_height = int(image.height * scale_factor)
            image = image.resize((target_width, new_height), Image.LANCZOS)
            # Crop height
            top = (new_height - target_height) // 2
            image = image.crop((0, top, target_width, top + target_height))
        
        return image
    
    else:
        raise ValueError(f"Unknown resize mode: {mode}")


def process_images(folder_path, target_width=None, target_height=None, mode='fit', keep_color=False, prefix='converted'):
    """
    Process all images in the specified folder.
    
    Args:
        folder_path: Path to the folder containing images
        target_width: Target width in pixels (None to keep original)
        target_height: Target height in pixels (None to keep original)
        mode: Resize mode
        keep_color: Whether to keep images in color (default: False, convert to B&W)
        prefix: Prefix for output filenames (default: 'converted')
    """
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Get all image files (exclude files that already have the prefix to avoid reprocessing)
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_formats
                   and not f.stem.startswith(f"{prefix}_")]
    
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        print(f"(Files with prefix '{prefix}_' are excluded to avoid reprocessing)")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    if target_width and target_height:
        print(f"Target resolution: {target_width}x{target_height}")
        print(f"Resize mode: {mode}")
    else:
        print(f"Resolution: Original (no resizing)")
    print(f"Color mode: {'Color' if keep_color else 'Black & White'}")
    print(f"Output prefix: {prefix}_")
    print(f"Output location: Same folder as source images")
    print("-" * 50)
    
    processed = 0
    failed = 0
    
    for image_file in image_files:
        try:
            print(f"Processing: {image_file.name}...", end=' ')
            
            # Open image
            with Image.open(image_file) as img:
                # Convert to RGB first (in case of RGBA or other modes)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Resize image if dimensions are specified
                if target_width and target_height:
                    resized_img = resize_image(img, target_width, target_height, mode, keep_color)
                else:
                    resized_img = img
                
                # Convert to black and white if needed
                if not keep_color:
                    final_img = convert_image_to_bw(resized_img)
                else:
                    final_img = resized_img
                
                # Save the converted image in the same folder with prefix
                output_path = folder_path / f"{prefix}_{image_file.stem}.png"
                final_img.save(output_path, 'PNG')
                
                print(f"✓ Saved to {output_path.name}")
                processed += 1
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            failed += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert images with optional resizing and black & white conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resize modes:
  fit      - Maintain aspect ratio, resize to fit within dimensions (default)
  contain  - Maintain aspect ratio, fit inside dimensions with padding
  stretch  - Ignore aspect ratio, stretch to exact dimensions
  cover    - Maintain aspect ratio, crop to fill dimensions

Examples:
  python image_converter.py --folder images --width 800 --height 600
  python image_converter.py --folder photos --width 1920 --height 1080 --mode contain
  python image_converter.py --folder pics --width 512 --height 512 --mode cover
  python image_converter.py --folder pics --width 1024 --height 768 --color
  python image_converter.py --folder images (keeps original resolution, converts to B&W)
  python image_converter.py --folder images --color (keeps original resolution and color)
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Name of the folder containing images (relative to script location)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        help='Target width in pixels (optional, keeps original if not specified)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        help='Target height in pixels (optional, keeps original if not specified)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['contain', 'stretch', 'fit', 'cover'],
        default='fit',
        help='Resize mode (default: fit)'
    )
    
    parser.add_argument(
        '--color',
        action='store_true',
        help='Keep images in color (default: convert to black & white)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='converted',
        help='Prefix for output filenames (default: converted)'
    )
    
    args = parser.parse_args()
    
    # Validate dimensions if provided
    if args.width is not None and args.width <= 0:
        print("Error: Width must be a positive integer")
        sys.exit(1)
    
    if args.height is not None and args.height <= 0:
        print("Error: Height must be a positive integer")
        sys.exit(1)
    
    # Check if only one dimension is provided
    if (args.width is None) != (args.height is None):
        print("Error: Both width and height must be specified together, or neither")
        sys.exit(1)
    
    # Get the folder path (relative to script location)
    script_dir = Path(__file__).parent
    folder_path = script_dir / args.folder
    
    # Check if folder exists
    if not folder_path.exists():
        print(f"Error: Folder '{args.folder}' not found at {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{args.folder}' is not a directory")
        sys.exit(1)
    
    # Process images
    process_images(folder_path, args.width, args.height, args.mode, args.color, args.prefix)


if __name__ == '__main__':
    main()
