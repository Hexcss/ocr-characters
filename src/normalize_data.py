import os
import shutil
import cv2
import uuid
import numpy as np
from tqdm import tqdm

# Config
SOURCE_ROOT = "data/hand"
OUTPUT_ROOT = "data/normalized"

# Allowed extensions
EXTS = ('.jpg', '.jpeg', '.png')

def create_output_dirs():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    # Create folders 0-9, A-Z, a-z
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for char in chars:
        os.makedirs(os.path.join(OUTPUT_ROOT, char), exist_ok=True)

def save_image(img, char_label, source_name):
    """
    Saves TWO versions of the image:
    1. {uuid}_raw.jpg : Original Grayscale (Dark text on Light Paper)
    2. {uuid}_bin.jpg : Thresholded (White text on Black Background - EMNIST style)
    """
    if img is None: return
    
    # Validate label
    if len(char_label) != 1 or not char_label.isalnum():
        return

    # 1. Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Create Binary Version (EMNIST Style)
    # We assume input is Dark Text on Light Paper, so we INV to get White Text
    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if image is mostly empty (noise filter using the binary mask)
    if cv2.countNonZero(img_bin) == 0: return

    # Generate unique ID
    file_id = uuid.uuid4()

    # Save Path Construction
    save_dir = os.path.join(OUTPUT_ROOT, char_label)
    
    # Save Raw (Original Texture)
    cv2.imwrite(os.path.join(save_dir, f"{file_id}_raw.jpg"), gray)
    
    # Save Binary (EMNIST Format)
    cv2.imwrite(os.path.join(save_dir, f"{file_id}_bin.jpg"), img_bin)

def process_mine():
    """Handles data/hand/mine"""
    path = os.path.join(SOURCE_ROOT, "mine")
    if not os.path.exists(path): return

    print("Processing 'mine' folder...")
    for f in os.listdir(path):
        if not f.lower().endswith(EXTS): continue
        
        name_no_ext = os.path.splitext(f)[0]
        file_path = os.path.join(path, f)
        img = cv2.imread(file_path)

        if "_" in name_no_ext: continue # Skip merged A_a files
        
        save_image(img, name_no_ext, "mine")

def process_primary():
    """Handles data/hand/primary/{lower,upper,number}"""
    base = os.path.join(SOURCE_ROOT, "primary")
    if not os.path.exists(base): return

    print("Processing 'primary' folder...")
    
    # 1. Numbers
    num_path = os.path.join(base, "number")
    if os.path.exists(num_path):
        for folder in os.listdir(num_path):
            current_dir = os.path.join(num_path, folder)
            if not os.path.isdir(current_dir): continue
            
            for f in os.listdir(current_dir):
                if f.lower().endswith(EXTS):
                    save_image(cv2.imread(os.path.join(current_dir, f)), folder, "primary_num")

    # 2. Upper
    upper_path = os.path.join(base, "upper")
    if os.path.exists(upper_path):
        for folder in os.listdir(upper_path):
            current_dir = os.path.join(upper_path, folder)
            if not os.path.isdir(current_dir): continue
            
            for f in os.listdir(current_dir):
                if f.lower().endswith(EXTS):
                    save_image(cv2.imread(os.path.join(current_dir, f)), folder, "primary_upper")

    # 3. Lower
    lower_path = os.path.join(base, "lower")
    if os.path.exists(lower_path):
        for folder in os.listdir(lower_path):
            current_dir = os.path.join(lower_path, folder)
            if not os.path.isdir(current_dir): continue
            
            for f in os.listdir(current_dir):
                if f.lower().endswith(EXTS):
                    save_image(cv2.imread(os.path.join(current_dir, f)), folder, "primary_lower")

def process_secondary():
    """Handles data/hand/secondary"""
    path = os.path.join(SOURCE_ROOT, "secondary")
    if not os.path.exists(path): return

    print("Processing 'secondary' folder...")
    for f in os.listdir(path):
        if not f.lower().endswith(EXTS): continue
        
        file_path = os.path.join(path, f)
        img = cv2.imread(file_path)
        
        name_no_ext = os.path.splitext(f)[0]
        label = name_no_ext[0] # First char rule
        
        if label.isalnum():
            save_image(img, label, "secondary")

def main():
    create_output_dirs()
    process_mine()
    process_primary()
    process_secondary()
    
    total = 0
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        total += len(files)
    
    print(f"\n✅ Normalization Complete!")
    print(f"Total images (Raw + Binary) in {OUTPUT_ROOT}: {total}")

if __name__ == "__main__":
    main()