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
        Advanced heuristic: Merges dots (i, j, !, ?) with their main body.
        """
        contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return []

        # Convert to bounding boxes: (x, y, w, h)
        boxes = [cv2.boundingRect(c) for c in contours]
        # Store as list of dicts to handle merging status
        box_data = [{'box': b, 'merged': False} for b in boxes]

        # Sort by X position
        box_data.sort(key=lambda item: item['box'][0])

        final_boxes = []
        
        for i in range(len(box_data)):
            if box_data[i]['merged']: continue
            
            x1, y1, w1, h1 = box_data[i]['box']
            
            # Check for a "dot" roughly above or below this box
            # This is O(N^2) but N is small (< 20 letters usually)
            merged_box = [x1, y1, w1, h1]
            
            for j in range(len(box_data)):
                if i == j or box_data[j]['merged']: continue
                
                x2, y2, w2, h2 = box_data[j]['box']
                
                # Heuristic: If boxes are horizontally aligned (close in X)
                # and one is vertically stacked
                
                # Centers
                c1_x = x1 + w1/2
                c2_x = x2 + w2/2
                
                if abs(c1_x - c2_x) < 20: # Horizontal overlap
                    # It's a match! Merge them.
                    box_data[j]['merged'] = True
                    
                    # Create new bounding box encasing both
                    min_x = min(x1, x2)
                    min_y = min(y1, y2)
                    max_x = max(x1+w1, x2+w2)
                    max_y = max(y1+h1, y2+h2)
                    
                    merged_box = [min_x, min_y, max_x - min_x, max_y - min_y]
            
            final_boxes.append(tuple(merged_box))
            
        # Re-sort final list left-to-right
        final_boxes.sort(key=lambda b: b[0])
        return final_boxes

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
            processed_roi = (processed_roi - 0.5) / 0.5
            
            # Infer
            tensor_img = torch.tensor(processed_roi).unsqueeze(0).unsqueeze(0).to(DEVICE)
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