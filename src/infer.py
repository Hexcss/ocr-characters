import sys
import os
import requests
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import NeuroOCR
from config.config import *
from src.corrector import SpellCorrector

def search_qdrant_http(vector):
    url = f"http://localhost:6333/collections/{COLLECTION_NAME}/points/search"
    payload = {"vector": vector, "limit": 1, "with_payload": True}
    try:
        response = requests.post(url, json=payload, timeout=1.0)
        return response.json().get("result", []) if response.status_code == 200 else None
    except: return None

def load_model():
    model = NeuroOCR(num_classes=47, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load("neuro_ocr_model.pth", map_location=DEVICE))
    model.eval()
    return model

def skeletonize(img):
    """
    Reduces the character to a 1-pixel wide skeleton.
    This removes variations in pen pressure/thickness.
    """
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    
    temp_img = img.copy()
    
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        
        zeros = size - cv2.countNonZero(temp_img)
        if zeros == size:
            done = True
            
    return skel

def standardize_stroke_width(img):
    """
    1. Skeletonize the character (shrink to 1px center line).
    2. Re-draw it with a specific fixed thickness (e.g. 3-4px).
    """
    h, w = img.shape
    if h == 0 or w == 0: return img

    # Sanity Check: Density
    non_zero = cv2.countNonZero(img)
    if (non_zero / (h*w)) < 0.02: return np.zeros_like(img)

    # 1. Upscale for precision
    target_h = 64
    scale = target_h / h
    target_w = int(w * scale)
    if target_w > 400: target_w = 400 # Safety limit
    
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Skeletonize (Get the structure)
    skel = skeletonize(img_resized)
    
    # 3. Regrow with Controlled Thickness
    # EMNIST digits are roughly 4-5px thick at 28x28 resolution.
    # At 64px height, we want a stroke of about 6-8px.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_regrown = cv2.dilate(skel, kernel, iterations=1)
    
    # 4. Smooth edges
    img_blur = cv2.GaussianBlur(img_regrown, (5, 5), 0)
    _, img_final = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY)
    
    return img_final

def smart_resize_pad(img, size=28):
    coords = cv2.findNonZero(img)
    if coords is None: return np.zeros((size, size), dtype=np.float32)
    
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    
    rows, cols = img.shape
    if rows == 0 or cols == 0: return np.zeros((size, size), dtype=np.float32)

    # Fit into 20x20 box inside 28x28 (Standard MNIST padding)
    factor = 20.0 / max(rows, cols)
    rows, cols = int(rows * factor), int(cols * factor)
    rows, cols = max(1, rows), max(1, cols)
    
    img = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
    
    final_img = np.zeros((28, 28), dtype=np.float32)
    col_center, row_center = (28 - cols) // 2, (28 - rows) // 2
    final_img[row_center:row_center+rows, col_center:col_center+cols] = img
    
    M = cv2.moments(final_img)
    if M["m00"] > 0:
        cX, cY = M["m10"] / M["m00"], M["m01"] / M["m00"]
        shift_x, shift_y = 14 - cX, 14 - cY
        M_affine = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final_img = cv2.warpAffine(final_img, M_affine, (28, 28))
    return final_img

def merge_close_boxes(boxes, distance_threshold=20):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    
    curr_x, curr_y, curr_w, curr_h = boxes[0]
    
    for i in range(1, len(boxes)):
        next_x, next_y, next_w, next_h = boxes[i]
        
        # Horizontal Proximity
        dist_x = next_x - (curr_x + curr_w)
        is_close_x = dist_x < distance_threshold
        
        # Vertical Alignment
        curr_cy = curr_y + curr_h / 2
        next_cy = next_y + next_h / 2
        is_aligned_y = abs(curr_cy - next_cy) < 30 
        
        # Containment
        is_contained = (next_x >= curr_x and (next_x + next_w) <= (curr_x + curr_w + 10))
        
        if (is_close_x and is_aligned_y) or is_contained:
            min_x = min(curr_x, next_x)
            min_y = min(curr_y, next_y)
            max_x = max(curr_x + curr_w, next_x + next_w)
            max_y = max(curr_y + curr_h, next_y + next_h)
            curr_x, curr_y, curr_w, curr_h = min_x, min_y, max_x - min_x, max_y - min_y
        else:
            merged.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = next_x, next_y, next_w, next_h
            
    merged.append((curr_x, curr_y, curr_w, curr_h))
    return merged

def preprocess_image(img_path):
    if not os.path.exists(img_path): sys.exit(f"File not found.")
    img = cv2.imread(img_path)
    if img is None: sys.exit("Read error.")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Boosting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_boosted = clahe.apply(gray)
    
    # 3. Denoising
    denoised = cv2.fastNlMeansDenoising(gray_boosted, None, 30, 7, 21)
    
    # 4. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    # 5. Connection Pass (Glue vertical gaps)
    # Smaller kernel here to avoid merging separate letters
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    thresh_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect)
    
    # 6. Clean Noise
    kernel_clean = np.ones((2,2), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh_connected, cv2.MORPH_OPEN, kernel_clean)

    # 7. Contours & Boxes
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes = [cv2.boundingRect(c) for c in contours]
    
    # 8. Filter Noise
    clean_boxes = []
    for (x,y,w,h) in all_boxes:
        if w > 4 and h > 10: clean_boxes.append((x,y,w,h))
            
    # 9. Merge Logic (tuned threshold)
    merged_boxes = merge_close_boxes(clean_boxes, distance_threshold=15)
    
    valid_crops, valid_coords = [], []
    img_h, img_w = thresh.shape
    
    for (x, y, w, h) in merged_boxes:
        if w > h * 6: continue 
        
        # Add padding to avoid cutting edges
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        
        # Crop from the Clean Threshold
        roi = thresh[y1:y2, x1:x2]
        
        # --- SKELETONIZE & REGROW ---
        # This standardizes thickness without merging neighbors
        roi = standardize_stroke_width(roi)
        # ----------------------------
        
        if cv2.countNonZero(roi) == 0: continue

        roi = roi.astype('float32') / 255.0
        roi = smart_resize_pad(roi, size=28)
        
        # Binary Force
        roi[roi > 0.2] = 1.0 
        roi[roi <= 0.2] = 0.0
        
        # Normalize
        roi = (roi - 0.5) / 0.5
        
        valid_crops.append(roi)
        valid_coords.append((x, y, w, h))
        
    return img, valid_crops, valid_coords

def recognize_text(image_path):
    model = load_model()
    original_img, crops, coords = preprocess_image(image_path)
    result_text = ""
    print(f"Found {len(crops)} chars. Reading...")
    
    if len(crops) > 0:
        debug_cols = min(len(crops), 15)
        plt.figure(figsize=(15, 2))
        for i in range(debug_cols):
            plt.subplot(1, debug_cols, i+1)
            plt.imshow(crops[i], cmap='gray')
            plt.axis('off')
            plt.title(f"#{i+1}")
        plt.suptitle("Input to AI (Skeletonized & Regrown)")
        plt.show(block=False)
        plt.pause(0.5)

    for i, crop in enumerate(crops):
        tensor_img = torch.tensor(crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): _, embedding = model(tensor_img)
        
        vector_list = embedding.cpu().numpy()[0].tolist()
        results = search_qdrant_http(vector_list)
        
        char, color = "?", (0, 0, 255)
        if results and results[0]['score'] > 0.45:
            char = results[0]['payload']['character']
            color = (0, 255, 0)
            
        result_text += char
        x, y, w, h = coords[i]
        cv2.rectangle(original_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(original_img, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    print(f"\nRAW OCR: {result_text}")
    
    final_text = result_text
    display_text = f"Raw: {result_text}"

    if SPELL_CHECK:
        corrector = SpellCorrector()
        corrected = corrector.correct(result_text)
        if corrected != result_text:
            print(f"✨ SPELL CHECK: {corrected}")
            final_text = corrected
            display_text += f"\nCorrected: {final_text}"
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(display_text, fontsize=14, color='blue')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1: recognize_text(sys.argv[1])
    else: print("Usage: python -m src.infer <image_path>")