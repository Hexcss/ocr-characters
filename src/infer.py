import sys
import os
import requests
import cv2
import torch
import numpy as np
import json
import glob
from pathlib import Path

# Adjust path to include root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_embedding import NeuroOCR
from config.config import *

# --- Setup Paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(ROOT_DIR, "tests")
OUT_DIR = os.path.join(ROOT_DIR, "out")

def search_qdrant_http(vector):
    url = f"http://localhost:6333/collections/{COLLECTION_NAME}/points/search"
    payload = {"vector": vector, "limit": 1, "with_payload": True}
    try:
        response = requests.post(url, json=payload, timeout=1.0)
        return response.json().get("result", []) if response.status_code == 200 else None
    except:
        return None


def load_model():
    model = NeuroOCR(num_classes=62, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load("neuro_ocr_model.pth", map_location=DEVICE))
    model.eval()
    return model


# --- CLASSICAL VISION PIPELINE ---

def auto_invert_to_paper_mode(img_gray):
    """
    NEW: Analyzes the image brightness. 
    If the background is dark (mean < 127), inverts it to create 
    Black Text on White Background. This fixes Test 6 & 9.
    """
    # Use a central crop to avoid black scanning borders affecting the mean
    h, w = img_gray.shape
    center = img_gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    
    if center.size == 0:
        mean_val = np.mean(img_gray)
    else:
        mean_val = np.mean(center)
    
    if mean_val < 127:
        print("    [Info] Detected Dark Background. Inverting.")
        return 255 - img_gray
    return img_gray


def skeletonize(img):
    """
    Iterative erosion to reduce characters to 1-pixel wide skeletons.
    Used to normalize stroke width.
    """
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    temp_img = img.copy()

    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            done = True
    return skel


def standardize_stroke_width(img):
    """
    RESTORED: The 'Magic' Normalizer.
    1. Resizes to 64px height (Low-pass filter against noise).
    2. Skeletonizes (Removes style/thickness variations).
    3. Re-dilates (Creates uniform fat strokes matching EMNIST training data).
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return img

    # Filter out empty crops
    non_zero = cv2.countNonZero(img)
    if (non_zero / (h * w)) < 0.01:
        return np.zeros_like(img)

    # 1. Resize to fixed height (64px)
    target_h = 64
    scale = target_h / h
    target_w = int(w * scale)
    if target_w > 400: target_w = 400 # Cap width

    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 2. Skeletonize
    skel = skeletonize(img_resized)

    # 3. Re-grow to standard thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_regrown = cv2.dilate(skel, kernel, iterations=1)

    # 4. Smooth edges
    img_blur = cv2.GaussianBlur(img_regrown, (5, 5), 0)
    _, img_final = cv2.threshold(img_blur, 50, 255, cv2.THRESH_BINARY)

    return img_final


def smart_resize_pad(img, size=28):
    """
    Standard MNIST-style resizing:
    1. Find bounding box of content.
    2. Resize content to fit in 20x20 box (preserving aspect ratio).
    3. Center using Center of Mass.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if float(np.max(img)) <= 1.0 else img.astype(np.uint8)

    # Threshold to ensure binary
    img = np.where(img > 0, 255, 0).astype(np.uint8)

    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((size, size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]

    rows, cols = img.shape
    if rows == 0 or cols == 0:
        return np.zeros((size, size), dtype=np.uint8)

    # Fit into 20x20 box inside 28x28
    factor = 20.0 / float(max(rows, cols))
    new_rows = max(1, int(round(rows * factor)))
    new_cols = max(1, int(round(cols * factor)))

    img = cv2.resize(img, (new_cols, new_rows), interpolation=cv2.INTER_NEAREST)

    final_img = np.zeros((size, size), dtype=np.uint8)
    row0 = (size - new_rows) // 2
    col0 = (size - new_cols) // 2
    final_img[row0 : row0 + new_rows, col0 : col0 + new_cols] = img

    # Center of Mass Alignment
    mom = cv2.moments(final_img)
    if mom["m00"] > 0:
        cX = mom["m10"] / mom["m00"]
        cY = mom["m01"] / mom["m00"]
        shift_x = (size / 2.0) - float(cX)
        shift_y = (size / 2.0) - float(cY)

        M_affine = np.array(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
            dtype=np.float32,
        )

        final_img = cv2.warpAffine(
            final_img,
            M_affine,
            (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return final_img

def merge_component_boxes(boxes):
    if not boxes:
        return []

    rects = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
    hs = np.array([h for (_, _, _, h) in boxes], dtype=float)
    median_h = float(np.median(hs)) if len(hs) else 0.0

    x_th = max(10.0, 0.10 * median_h)
    y_th = max(10.0, 0.10 * median_h)

    # Overlap thresholds
    y_ov_th = 0.30
    x_ov_th = 0.30

    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        ax1, ay1, ax2, ay2 = rects[i]
        aw, ah = ax2 - ax1, ay2 - ay1

        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = rects[j]
            bw, bh = bx2 - bx1, by2 - by1

            ov_x = min(ax2, bx2) - max(ax1, bx1)
            ov_y = min(ay2, by2) - max(ay1, by1)

            x_overlap = max(0.0, ov_x)
            y_overlap = max(0.0, ov_y)

            gap_x = 0.0 if ov_x >= 0 else -ov_x
            gap_y = 0.0 if ov_y >= 0 else -ov_y

            x_overlap_ratio = x_overlap / max(1.0, min(aw, bw))
            y_overlap_ratio = y_overlap / max(1.0, min(ah, bh))

            close_side_by_side = (gap_x <= x_th) and (y_overlap_ratio >= y_ov_th)
            close_stacked = (gap_y <= y_th) and (x_overlap_ratio >= x_ov_th)

            if close_side_by_side or close_stacked:
                union(i, j)

    groups = {}
    for i, b in enumerate(boxes):
        r = find(i)
        groups.setdefault(r, []).append(b)

    merged = []
    for g in groups.values():
        x1 = min(x for x, y, w, h in g)
        y1 = min(y for x, y, w, h in g)
        x2 = max(x + w for x, y, w, h in g)
        y2 = max(y + h for x, y, w, h in g)
        merged.append((x1, y1, x2 - x1, y2 - y1))

    # Basic sorting by x
    merged.sort(key=lambda b: b[0])
    return merged


def filter_merged_boxes(merged_boxes):
    if not merged_boxes:
        return []

    hs = np.array([h for x, y, w, h in merged_boxes], dtype=float)
    areas = np.array([w * h for x, y, w, h in merged_boxes], dtype=float)

    med_h = float(np.median(hs))
    med_a = float(np.median(areas))

    kept = []
    for x, y, w, h in merged_boxes:
        a = w * h
        # Basic garbage filter
        if h < 0.35 * med_h and a < 0.10 * med_a:
            continue
        kept.append((x, y, w, h))

    kept.sort(key=lambda b: b[0])
    return kept


def preprocess_image(img_path):
    if not os.path.exists(img_path):
        print("File not found.")
        return None, [], [], {}
        
    img = cv2.imread(img_path)
    if img is None:
        print("Read error.")
        return None, [], [], {}

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. AUTO-INVERT (The new addition)
    gray = auto_invert_to_paper_mode(gray)

    # 3. CLAHE (Contrast Boost)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_boosted = clahe.apply(gray)

    # 4. Denoise
    denoised = cv2.fastNlMeansDenoising(gray_boosted, None, 30, 7, 21)

    # 5. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )

    # 6. Morphological Connection & Cleaning
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    thresh_connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_v)
    thresh_connected = cv2.morphologyEx(thresh_connected, cv2.MORPH_CLOSE, kernel_h)

    kernel_clean = np.ones((2, 2), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh_connected, cv2.MORPH_OPEN, kernel_clean)

    debug_artifacts = {
        "original": img,
        "binary_clean": thresh_clean
    }

    # 7. CCA & Box merging
    img_h, img_w = thresh_clean.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_clean, connectivity=8)

    img_area = img_h * img_w
    min_area = max(30, int(0.00002 * img_area))

    comp_boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        if w <= 4 or h <= 10: continue
        comp_boxes.append((x, y, w, h))

    merged_boxes = merge_component_boxes(comp_boxes)
    merged_boxes = filter_merged_boxes(merged_boxes)

    valid_crops, valid_coords = [], []

    for (x, y, w, h) in merged_boxes:
        if w > h * 6: continue # Skip obviously long horizontal lines

        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        roi = thresh_clean[y1:y2, x1:x2]

        # 8. THE MAGIC NORMALIZER (Restored)
        # Resizes to 64px, skeletonizes, and regrows. 
        # This removes noise and normalizes stroke width.
        roi = standardize_stroke_width(roi)

        if cv2.countNonZero(roi) == 0:
            continue

        # 9. Smart Resize to 28x28
        roi_28 = smart_resize_pad(roi, size=28) 

        # 10. Normalize for Model
        roi_28 = (roi_28 > 0).astype(np.float32)
        roi_28 = (roi_28 - 0.5) / 0.5

        valid_crops.append(roi_28)
        valid_coords.append((x, y, w, h))

    # Basic sorting top-down
    valid = list(zip(valid_coords, valid_crops))
    valid.sort(key=lambda t: t[0][0]) # Sort by X first
    # (A robust sorter would sort by Y then X, but we stick to original for now)
    
    if valid:
        valid_coords, valid_crops = zip(*valid)
        valid_coords, valid_crops = list(valid_coords), list(valid_crops)
    else:
        valid_coords, valid_crops = [], []

    return img, valid_crops, valid_coords, debug_artifacts


def run_pipeline():
    try:
        model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found at: {TEST_DIR}")
        return
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    print(f"Found {len(image_files)} images in {TEST_DIR}")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        current_out_dir = os.path.join(OUT_DIR, base_name)
        crops_dir = os.path.join(current_out_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        print(f"Processing {filename}...")
        
        original_img, crops, coords, debug_artifacts = preprocess_image(img_path)
        
        if original_img is None:
            continue

        # Save Debug Images
        cv2.imwrite(os.path.join(current_out_dir, "00_original.png"), debug_artifacts['original'])
        cv2.imwrite(os.path.join(current_out_dir, "01_binary_clean.png"), debug_artifacts['binary_clean'])

        result_text = ""
        inference_data = []

        print(f"  > Found {len(crops)} chars. Inferring...")

        annotated_img = original_img.copy()

        for i, crop in enumerate(crops):
            # Convert crop back to uint8 0-255 for saving
            debug_crop_view = ((crop * 0.5 + 0.5) * 255).astype(np.uint8)
            
            tensor_img = torch.tensor(crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, embedding = model(tensor_img)

            vector_list = embedding.cpu().numpy()[0].tolist()
            results = search_qdrant_http(vector_list)

            char, color = "?", (0, 0, 255)
            score = 0.0
            
            if results:
                score = results[0].get("score", 0.0)
                if score > 0.45:
                    payload = results[0].get("payload") or {}
                    char = payload.get("character", "?")
                    color = (0, 255, 0)

            result_text += char
            x, y, w, h = coords[i]

            char_filename = f"char_{i:03d}_pred_{char if char.isalnum() else 'unk'}.png"
            cv2.imwrite(os.path.join(crops_dir, char_filename), debug_crop_view)

            inference_data.append({
                "index": i,
                "bbox": [int(x), int(y), int(w), int(h)],
                "prediction": char,
                "score": float(score)
            })

            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated_img,
                char,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        print(f"  > Result: {result_text}")
        
        with open(os.path.join(current_out_dir, "inference_data.json"), "w") as f:
            json.dump({
                "filename": filename,
                "full_text": result_text,
                "characters": inference_data
            }, f, indent=4)

        cv2.imwrite(os.path.join(current_out_dir, "02_result_overlay.png"), annotated_img)
        print(f"  > Saved debug data to {current_out_dir}")


if __name__ == "__main__":
    run_pipeline()