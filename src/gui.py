import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import cv2
import numpy as np
import torch
import sys
import os
import uuid
import requests
from PIL import Image, ImageDraw

# Fix Imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model_embedding import NeuroOCR
from config.config import *
from src.infer import smart_resize_pad, search_qdrant_http, load_model

class NeuroDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroOCR: Active Learning Slate")
        self.root.geometry("800x600")
        
        print("Loading AI Brain...")
        self.model = load_model()
        print("AI Ready.")

        # State to store data for correction
        self.last_vectors = [] 
        self.last_chars = []

        # --- UI SETUP ---
        self.main_frame = tk.Frame(root, bg="#2d2d2d")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Canvas
        self.canvas_width = 760
        self.canvas_height = 350
        self.canvas = tk.Canvas(
            self.main_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="black", 
            cursor="cross"
        )
        self.canvas.pack(pady=20, padx=20)

        # Hidden PIL Image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

        # 2. Results Area
        self.result_frame = tk.Frame(self.main_frame, bg="#2d2d2d")
        self.result_frame.pack(fill=tk.X, padx=20)

        self.lbl_result = tk.Label(
            self.result_frame, 
            text="Draw & Click Read", 
            font=("Consolas", 28, "bold"), 
            fg="#00ff00", 
            bg="#2d2d2d"
        )
        self.lbl_result.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 3. Controls
        self.controls_frame = tk.Frame(self.main_frame, bg="#2d2d2d")
        self.controls_frame.pack(fill=tk.X, pady=20, padx=20)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        self.btn_clear = ttk.Button(self.controls_frame, text="Clear (C)", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_correct = ttk.Button(self.controls_frame, text="Wrong? Teach Me!", command=self.teach_mistake)
        self.btn_correct.pack(side=tk.LEFT, padx=10)

        self.btn_read = ttk.Button(self.controls_frame, text="READ (Enter)", command=self.predict_drawing)
        self.btn_read.pack(side=tk.RIGHT, padx=10)

        # Bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)
        self.root.bind("<Return>", lambda e: self.predict_drawing())
        self.root.bind("<c>", lambda e: self.clear_canvas())
        
        self.last_x, self.last_y = None, None

    def reset_coords(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        # Thicker brush helps the CNN (matches EMNIST thickness)
        brush_size = 14  
        
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y, 
                width=brush_size, fill="white", capstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y], 
                fill="white", width=brush_size, joint="curve"
            )
        self.last_x = x
        self.last_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="...")
        self.last_vectors = []

    def get_merged_contours(self, img_np):
        """
        Merge diacritics (ñ, á, ü, i/j dots, etc.) into their base glyph
        so they become ONE detection box.

        Strategy:
        - Find contours -> bounding boxes
        - Classify small-height boxes as "marks" (diacritics)
        - Attach each mark to the best base box using:
            * horizontal overlap
            * mark is clearly above/below the base center (so "-" doesn't get merged)
            * reasonable vertical gap
        """

        # 1) Binarize (stable contours)
        _, bw = cv2.threshold(img_np, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        boxes = [cv2.boundingRect(c) for c in contours]

        # 2) Remove tiny noise early
        filtered = []
        for (x, y, w, h) in boxes:
            if w >= 4 and h >= 4 and (w * h) >= 25:
                filtered.append((x, y, w, h))
        boxes = filtered
        if not boxes:
            return []

        # Sort left-to-right for consistency
        boxes.sort(key=lambda b: b[0])

        # 3) Estimate typical char height (median is robust)
        hs = np.array([b[3] for b in boxes], dtype=np.float32)
        median_h = float(np.median(hs)) if len(hs) else 0.0
        if median_h <= 0:
            return boxes

        # Tunables (adjust if needed)
        MARK_H_RATIO = 0.55          # below this height -> likely diacritic mark
        X_OVERLAP_THRESH = 0.25      # overlap relative to smaller width
        MAX_GAP_RATIO = 0.90         # max vertical gap relative to base height
        CENTER_BAND_RATIO = 0.18     # if mark sits near base center, don't merge (prevents '-' merging)

        mark_h_thresh = median_h * MARK_H_RATIO

        merged = [list(b) for b in boxes]  # mutable copies
        consumed_marks = set()

        base_idxs = []
        mark_idxs = []
        for idx, (x, y, w, h) in enumerate(boxes):
            if h < mark_h_thresh:
                mark_idxs.append(idx)
            else:
                base_idxs.append(idx)

        def x_overlap_ratio(a, b):
            ax, ay, aw, ah = a
            bx, by, bw_, bh_ = b
            ax2, bx2 = ax + aw, bx + bw_
            inter = max(0, min(ax2, bx2) - max(ax, bx))
            denom = min(aw, bw_)
            return (inter / denom) if denom > 0 else 0.0

        # 4) Attach marks to bases
        for mi in mark_idxs:
            if mi in consumed_marks:
                continue

            mx, my, mw, mh = boxes[mi]
            mark_cy = my + mh / 2.0

            best_bi = None
            best_ov = 0.0
            best_gap = 1e9

            for bi in base_idxs:
                bx, by, bw_, bh_ = merged[bi]
                base_cy = by + bh_ / 2.0

                # Must overlap in X (ñ tilde overlaps the n in x-range)
                ov = x_overlap_ratio((mx, my, mw, mh), (bx, by, bw_, bh_))
                if ov < X_OVERLAP_THRESH:
                    continue

                # Mark must be clearly ABOVE or BELOW the base center
                # (prevents '-' or middle strokes from being incorrectly merged)
                if abs(mark_cy - base_cy) < (CENTER_BAND_RATIO * bh_):
                    continue

                # Compute vertical gap (0 if overlapping)
                if my + mh <= by:
                    gap = by - (my + mh)          # mark above base
                elif by + bh_ <= my:
                    gap = my - (by + bh_)         # mark below base (e.g., ç)
                else:
                    gap = 0

                if gap > (MAX_GAP_RATIO * bh_):
                    continue

                # Score: prefer bigger overlap, then smaller gap
                if (ov > best_ov) or (abs(ov - best_ov) < 1e-6 and gap < best_gap):
                    best_bi = bi
                    best_ov = ov
                    best_gap = gap

            if best_bi is not None:
                bx, by, bw_, bh_ = merged[best_bi]
                min_x = min(bx, mx)
                min_y = min(by, my)
                max_x = max(bx + bw_, mx + mw)
                max_y = max(by + bh_, my + mh)
                merged[best_bi] = [min_x, min_y, max_x - min_x, max_y - min_y]
                consumed_marks.add(mi)

        # 5) Return merged boxes excluding consumed marks
        out = [tuple(merged[i]) for i in range(len(merged)) if i not in consumed_marks]
        out.sort(key=lambda b: b[0])
        return out

    def predict_drawing(self):
        img_np = np.array(self.image)
        boxes = self.get_merged_contours(img_np)
        
        if not boxes:
            self.lbl_result.config(text="Canvas Empty")
            return

        self.last_vectors = [] # Clear previous memory
        result_text = ""
        
        # Visualize detection (Draw green boxes on canvas temporarily)
        self.canvas.delete("debug_box")

        for (x, y, w, h) in boxes:
            # Noise Filter: Ignore tiny specs
            if w < 10 or h < 10: continue

            # Visual Debug
            self.canvas.create_rectangle(x, y, x+w, y+h, outline="green", tags="debug_box")

            # Crop
            roi = img_np[y:y+h, x:x+w]
            roi = roi.astype('float32') / 255.0
            processed_roi = smart_resize_pad(roi, size=28)
            processed_roi = processed_roi.astype(np.float32) / 255.0 
            processed_roi = (processed_roi - 0.5) / 0.5
            
            # Infer
            # FIX: Explicitly cast to float32 here to prevent RuntimeError
            tensor_img = torch.tensor(processed_roi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, embedding = self.model(tensor_img)
            
            vector_list = embedding.cpu().numpy()[0].tolist()
            
            # Store for teaching
            self.last_vectors.append(vector_list)
            
            results = search_qdrant_http(vector_list)
            
            if results and results[0]['score'] > 0.55:
                char = results[0]['payload']['character']
                result_text += char
            else:
                result_text += "?"

        self.lbl_result.config(text=result_text)

    def teach_mistake(self):
        if not self.last_vectors:
            messagebox.showinfo("Info", "Draw and Read something first!")
            return

        # Ask user for correct string
        current_text = self.lbl_result.cget("text")
        correct_text = simpledialog.askstring("Teach AI", f"The AI read: '{current_text}'\nWhat did you actually write?", parent=self.root)
        
        if not correct_text: return
        
        if len(correct_text) != len(self.last_vectors):
            messagebox.showerror("Error", f"Character count mismatch.\nAI saw {len(self.last_vectors)} chars, you typed {len(correct_text)}.\nTry clearing noise first.")
            return

        # Upload Corrections
        url = f"http://localhost:6333/collections/{COLLECTION_NAME}/points"
        
        points = []
        for i, char in enumerate(correct_text):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": self.last_vectors[i],
                "payload": {"character": char, "source": "gui_correction"}
            })
        
        try:
            requests.put(url, json={"points": points})
            messagebox.showinfo("Success", f"Learned! The AI now knows your style for: {correct_text}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroDrawApp(root)
    root.mainloop()