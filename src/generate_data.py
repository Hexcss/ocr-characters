import cv2
import numpy as np
import os
import random
import string
from PIL import Image, ImageDraw, ImageFont

# Config
OUTPUT_DIR = "data/synthetic"
SAMPLES_PER_CHAR = 50
CANVAS_SIZE = 64 # Generate slightly larger, then resize later

def create_paper_texture(shape):
    """Generates a noisy, uneven background."""
    # Base gray/white color
    base = np.full(shape, random.randint(180, 240), dtype=np.uint8)
    
    # Add Gaussian Noise (Grain)
    noise = np.random.normal(0, 10, shape).astype(np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    # Add Lighting Gradient (Shadows)
    rows, cols = shape
    gradient = np.zeros(shape, dtype=np.float32)
    
    # Randomly darken one corner/side
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    if direction == 'horizontal':
        for col in range(cols): gradient[:, col] = col / cols
    elif direction == 'vertical':
        for row in range(rows): gradient[row, :] = row / rows
    else:
        for row in range(rows):
            for col in range(cols):
                gradient[row, col] = (row + col) / (rows + cols)
                
    # Apply gradient shadow
    shadow_intensity = random.randint(20, 80)
    img = cv2.subtract(img, (gradient * shadow_intensity).astype(np.uint8))
    
    return img

def generate_sample(char, save_path):
    # 1. Create Paper Background
    img_cv = create_paper_texture((CANVAS_SIZE, CANVAS_SIZE))
    
    # 2. Convert to PIL to use Fonts
    img_pil = Image.fromarray(img_cv)
    draw = ImageDraw.Draw(img_pil)
    
    # 3. Choose Font
    # Try to load a system font, fallback to default
    font_size = random.randint(40, 55)
    try:
        # Common Linux fonts paths
        fonts = [
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/gnu-free/FreeSansBold.ttf"
        ]
        valid_font = next((f for f in fonts if os.path.exists(f)), None)
        if valid_font:
            font = ImageFont.truetype(valid_font, font_size)
        else:
            raise Exception("No font found")
    except:
        font = ImageFont.load_default()

    # 4. Draw Text (Random Offset)
    # Get text size
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Center +/- random jitter
    x = (CANVAS_SIZE - text_w) // 2 + random.randint(-5, 5)
    y = (CANVAS_SIZE - text_h) // 2 + random.randint(-5, 5)
    
    # Ink color (not purely black, varies slightly)
    ink_color = random.randint(0, 50)
    draw.text((x, y), char, font=font, fill=ink_color)
    
    # 5. Convert back to OpenCV
    final_img = np.array(img_pil)
    
    # 6. Post-Processing Effects (Simulate bad camera)
    
    # Blur (Out of focus)
    if random.random() > 0.5:
        k = random.choice([3, 5])
        final_img = cv2.GaussianBlur(final_img, (k, k), 0)
        
    # Erosion/Dilation (Ink bleeding)
    if random.random() > 0.7:
        kernel = np.ones((2,2), np.uint8)
        if random.random() > 0.5:
            final_img = cv2.erode(final_img, kernel, iterations=1)
        else:
            final_img = cv2.dilate(final_img, kernel, iterations=1)

    # Save
    cv2.imwrite(save_path, final_img)

def main():
    print("🏭 Starting Sim-to-Real Generator...")
    
    # EMNIST Characters (0-9, A-Z, a-z overlap)
    # We'll just generate 0-9 and A-Z for simplicity in filenames
    chars = string.digits + string.ascii_uppercase
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    count = 0
    for char in chars:
        char_dir = os.path.join(OUTPUT_DIR, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
            
        print(f"Generating synthetic samples for: '{char}'")
        for i in range(SAMPLES_PER_CHAR):
            filename = f"{i}.png"
            save_path = os.path.join(char_dir, filename)
            generate_sample(char, save_path)
            count += 1
            
    print(f"✅ Generation Complete! Created {count} synthetic images in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()