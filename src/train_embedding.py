import sys
import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from PIL import Image, ImageOps

# ---- Local imports ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_embedding import NeuroOCR
from config.config import *

# ----- cuDNN autotune -----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
USE_CHANNELS_LAST = False

# -------------------- 1. MAPPINGS --------------------
EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}
CHAR_TO_LABEL = {v: k for k, v in EMNIST_MAPPING.items()}

# -------------------- 2. LIGHTWEIGHT CPU AUGS --------------------
def orientation_fix(img):
    # EMNIST needs rotate -90 + hflip to match standard orientation
    img = transforms.functional.rotate(img, -90)
    img = transforms.functional.hflip(img)
    return img

# -------------------- 3. CUSTOM DATASETS --------------------

class CustomHandwritingDataset(Dataset):
    """
    Loads your personal/classmate data from data/normalized.
    Handles both _raw.jpg (needs inversion) and _bin.jpg (ready to go).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: Custom folder '{root_dir}' not found.")
            return
            
        for char_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, char_folder)
            if not os.path.isdir(folder_path): continue
            
            # Check if this character exists in our mapping
            label_idx = None
            if char_folder in CHAR_TO_LABEL:
                label_idx = CHAR_TO_LABEL[char_folder]
            
            if label_idx is not None:
                for f in os.listdir(folder_path):
                    if f.endswith(('.jpg', '.png')):
                        self.samples.append((os.path.join(folder_path, f), label_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        
        # LOGIC:
        # _bin.jpg is already White text on Black bg (EMNIST style) -> Keep as is.
        # _raw.jpg is Dark text on Light paper -> Needs Inversion.
        if "_raw" in path:
            img = ImageOps.invert(img)
            
        if self.transform:
            img = self.transform(img)
        return img, label

class KaggleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        if not os.path.exists(root_dir): return
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            char_key = folder_name.replace("_caps", "")
            label_idx = None
            if char_key in CHAR_TO_LABEL: label_idx = CHAR_TO_LABEL[char_key]
            elif char_key.upper() in CHAR_TO_LABEL: label_idx = CHAR_TO_LABEL[char_key.upper()]
            
            if label_idx is not None:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(folder_path, file), label_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        img = ImageOps.invert(img)  # Kaggle is black-on-white; invert
        if self.transform: img = self.transform(img)
        return img, label

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        if not os.path.exists(root_dir): return
        for char in os.listdir(root_dir):
            p = os.path.join(root_dir, char)
            if os.path.isdir(p) and char in CHAR_TO_LABEL:
                idx = CHAR_TO_LABEL[char]
                for f in os.listdir(p):
                    if f.endswith('.png'): self.samples.append((os.path.join(p, f), idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert('L')
        if self.transform: img = self.transform(img)
        return img, l

# -------------------- 4. GPU AUGMENTATION --------------------
@torch.no_grad()
def gpu_augment(x, p_affine=0.60, p_noise=0.45, p_thin=0.50):
    B, C, H, W = x.shape
    device = x.device

    # Affine
    if p_affine > 0:
        mask = torch.rand(B, device=device) < p_affine
        if mask.any():
            n = int(mask.sum().item())
            angles = (torch.rand(n, device=device) * 30.0 - 15.0) * (math.pi / 180.0)
            scales = 0.8 + torch.rand(n, device=device) * 0.4
            tx = (torch.rand(n, device=device) * 0.2 - 0.1)
            ty = (torch.rand(n, device=device) * 0.2 - 0.1)
            cos = torch.cos(angles) * scales
            sin = torch.sin(angles) * scales
            theta = torch.zeros(n, 2, 3, device=device)
            theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 1, 0] = sin; theta[:, 1, 1] = cos
            theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
            x_sel = x[mask]
            grid = F.affine_grid(theta, x_sel.size(), align_corners=False)
            x[mask] = F.grid_sample(x_sel, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Thinning
    if p_thin > 0:
        mask = torch.rand(B, device=device) < p_thin
        if mask.any():
            x_sel = x[mask]
            inv = 1.0 - x_sel
            dil = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
            x[mask] = 1.0 - dil

    # Noise
    if p_noise > 0:
        mask = torch.rand(B, device=device) < p_noise
        if mask.any():
            x_sel = x[mask] + torch.randn_like(x[mask]) * 0.15
            x[mask] = x_sel.clamp_(0.0, 1.0)

    return x.clamp_(0.0, 1.0)

# -------------------- 5. TRAIN + INDEX --------------------
def _worker_init_fn(_):
    try: torch.set_num_threads(1)
    except: pass
    os.environ["OMP_NUM_THREADS"] = "1"

def train_and_index():
    print("🚀 Preparing Training Pipeline...")

    # CPU Transforms
    transform_emnist = transforms.Compose([orientation_fix, transforms.ToTensor()])
    transform_external = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    print("Loading datasets...")
    ds_emnist_c = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_emnist)
    ds_emnist_p = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_emnist)
    
    # Extra datasets
    ds_synth = SyntheticDataset('data/synthetic', transform=transform_external)
    ds_kaggle = KaggleDataset('data/kaggle', transform=transform_external)
    ds_custom = CustomHandwritingDataset('data/normalized', transform=transform_external)

    full_dataset = ConcatDataset([ds_emnist_c, ds_emnist_p, ds_synth, ds_kaggle, ds_custom])

    print(f"📊 Training Data Stats:")
    print(f"   EMNIST (Clean+Polluted): {len(ds_emnist_c) + len(ds_emnist_p)}")
    print(f"   Synthetic: {len(ds_synth)}")
    print(f"   Kaggle:    {len(ds_kaggle)}")
    print(f"   CUSTOM:    {len(ds_custom)} (Your Handwriting)")
    print(f"   TOTAL:     {len(full_dataset)}")

    NUM_WORKERS = min(8, max(4, (os.cpu_count() or 8) // 2))
    train_loader = DataLoader(
        full_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn
    )
    
    # Use standard EMNIST as reference index
    index_loader = DataLoader(ds_emnist_c, batch_size=1, shuffle=False, num_workers=0)

    model = NeuroOCR(num_classes=47, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"🔥 Starting Training on {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        start_t = time.time()
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # GPU Augmentation
            images = gpu_augment(images, p_affine=0.60, p_noise=0.45, p_thin=0.50)
            images = (images - 0.5) / 0.5 

            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=f"{loss.item():.4f}")

        dt = time.time() - start_t
        print(f"⏱️ Epoch {epoch+1} finished in {dt:.1f}s")

    torch.save(model.state_dict(), "neuro_ocr_model.pth")
    print("✅ Model saved.")

    # ------- Qdrant Indexing -------
    print("Resetting Qdrant Collection...")
    client = QdrantClient("localhost", port=6333)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    model.eval()
    points = []
    samples_per_class = {i: 0 for i in range(47)}
    max_samples = 30

    print("Indexing Reference Knowledge...")
    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(index_loader)):
            label_idx = label.item()
            if samples_per_class[label_idx] >= max_samples: continue

            img = img.to(DEVICE, non_blocking=True)
            img = (img - 0.5) / 0.5
            _, embedding = model(img)
            
            char_label = EMNIST_MAPPING.get(label_idx, "?")
            points.append(PointStruct(id=i, vector=embedding.cpu().numpy()[0].tolist(), payload={"character": char_label}))
            samples_per_class[label_idx] += 1

            if len(points) >= 1000:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    if points: client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("🎉 System Ready!")

if __name__ == "__main__":
    train_and_index()