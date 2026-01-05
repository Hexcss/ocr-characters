import sys
import os
import math
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_embedding import NeuroOCR
from config.config import *

# ----- cuDNN autotune -----
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# -------------------- 1) EMNIST orientation fix --------------------
def orientation_fix(img):
    # EMNIST needs rotate -90 + hflip to match standard orientation
    img = transforms.functional.rotate(img, -90)
    img = transforms.functional.hflip(img)
    return img


# -------------------- 2) Datasets that use the *same* mapping --------------------
class CustomHandwritingDataset(Dataset):
    """
    Loads personal/classmate data from data/normalized/<char>/*.png|jpg
    Handles both _raw (invert) and _bin (already white-on-black).
    """
    def __init__(self, root_dir, char_to_label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.char_to_label = char_to_label
        self.samples = []

        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: Custom folder '{root_dir}' not found.")
            return

        for char_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, char_folder)
            if not os.path.isdir(folder_path):
                continue

            if char_folder not in self.char_to_label:
                continue

            label_idx = self.char_to_label[char_folder]
            for f in os.listdir(folder_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        # _bin => already white-on-black, keep
        # _raw => dark ink on paper, invert to white-on-black
        if "_raw" in path:
            img = ImageOps.invert(img)

        if self.transform:
            img = self.transform(img)

        return img, label


class KaggleDataset(Dataset):
    """
    Loads data/kaggle/<char or char_caps>/*.png|jpg|jpeg
    Kaggle tends to be black-on-white, we invert to match EMNIST (white-on-black).
    """
    def __init__(self, root_dir, char_to_label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.char_to_label = char_to_label
        self.samples = []

        if not os.path.exists(root_dir):
            return

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            char_key = folder_name.replace("_caps", "")
            label_idx = None

            if char_key in self.char_to_label:
                label_idx = self.char_to_label[char_key]
            elif char_key.upper() in self.char_to_label:
                label_idx = self.char_to_label[char_key.upper()]

            if label_idx is None:
                continue

            for file in os.listdir(folder_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(folder_path, file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = ImageOps.invert(img)  # Kaggle is usually black-on-white
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticDataset(Dataset):
    """
    Loads data/synthetic/<char>/*.png (assumed already white-on-black)
    """
    def __init__(self, root_dir, char_to_label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.char_to_label = char_to_label
        self.samples = []

        if not os.path.exists(root_dir):
            return

        for char in os.listdir(root_dir):
            p = os.path.join(root_dir, char)
            if os.path.isdir(p) and (char in self.char_to_label):
                idx = self.char_to_label[char]
                for f in os.listdir(p):
                    if f.lower().endswith(".png"):
                        self.samples.append((os.path.join(p, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, l


# -------------------- 3) GPU augmentation (unchanged) --------------------
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
            theta[:, 0, 0] = cos
            theta[:, 0, 1] = -sin
            theta[:, 1, 0] = sin
            theta[:, 1, 1] = cos
            theta[:, 0, 2] = tx
            theta[:, 1, 2] = ty
            x_sel = x[mask]
            grid = F.affine_grid(theta, x_sel.size(), align_corners=False)
            x[mask] = F.grid_sample(x_sel, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

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


# -------------------- 4) Loader worker init --------------------
def _worker_init_fn(_):
    try:
        torch.set_num_threads(1)
    except:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


# -------------------- 5) Train + reindex (62 classes) --------------------
def train_and_index():
    print("🚀 Preparing Training Pipeline (EMNIST byclass => 62 classes)...")

    # EMNIST uses special orientation
    transform_emnist = transforms.Compose([orientation_fix, transforms.ToTensor()])

    # External datasets: already upright, just resize
    transform_external = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    print("Loading EMNIST byclass...")
    ds_emnist = datasets.EMNIST(
        root="./data",
        split="byclass",
        train=True,
        download=True,
        transform=transform_emnist,
    )

    # Build mapping from the dataset itself (no guessing)
    if not hasattr(ds_emnist, "classes") or not ds_emnist.classes:
        raise RuntimeError("Torchvision EMNIST dataset did not expose .classes. Cannot build mapping safely.")

    idx_to_char = {i: c for i, c in enumerate(ds_emnist.classes)}
    char_to_idx = {c: i for i, c in idx_to_char.items()}
    num_classes = len(idx_to_char)

    print(f"✅ EMNIST classes: {num_classes}")
    print("   First 20:", ds_emnist.classes[:20])
    print("   Last 20:", ds_emnist.classes[-20:])

    # Extra datasets using the SAME mapping
    ds_synth = SyntheticDataset("data/synthetic", char_to_label=char_to_idx, transform=transform_external)
    ds_kaggle = KaggleDataset("data/kaggle", char_to_label=char_to_idx, transform=transform_external)
    ds_custom = CustomHandwritingDataset("data/normalized", char_to_label=char_to_idx, transform=transform_external)

    full_dataset = ConcatDataset([ds_emnist, ds_synth, ds_kaggle, ds_custom])

    print(f"📊 Training Data Stats:")
    print(f"   EMNIST byclass: {len(ds_emnist)}")
    print(f"   Synthetic:      {len(ds_synth)}")
    print(f"   Kaggle:         {len(ds_kaggle)}")
    print(f"   CUSTOM:         {len(ds_custom)}")
    print(f"   TOTAL:          {len(full_dataset)}")

    num_workers = min(8, max(2, (os.cpu_count() or 8) // 2))
    train_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    # Use EMNIST as the reference indexing set (clean, labeled)
    index_loader = DataLoader(ds_emnist, batch_size=1, shuffle=False, num_workers=0)

    print(f"🧠 Building model num_classes={num_classes}, embedding_dim={EMBEDDING_DIM}")
    model = NeuroOCR(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(DEVICE)

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

    # Save (overwrite your old 47-class model)
    torch.save(model.state_dict(), "neuro_ocr_model.pth")
    print("✅ Model saved to neuro_ocr_model.pth")

    # ------- Qdrant Reindex -------
    print("🗃️ Resetting Qdrant Collection...")
    client = QdrantClient("localhost", port=6333)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    model.eval()
    points = []
    samples_per_class = {i: 0 for i in range(num_classes)}
    max_samples = 40  # slightly higher for 62 classes

    print("📌 Indexing Reference Knowledge (EMNIST byclass)...")

    point_id = 0
    with torch.no_grad():
        for img, label in tqdm(index_loader):
            label_idx = int(label.item())
            if samples_per_class[label_idx] >= max_samples:
                continue

            img = img.to(DEVICE, non_blocking=True)
            img = (img - 0.5) / 0.5

            _, embedding = model(img)
            ch = idx_to_char.get(label_idx, "?")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.detach().cpu().numpy()[0].tolist(),
                    payload={"character": ch, "class_id": label_idx},
                )
            )
            point_id += 1
            samples_per_class[label_idx] += 1

            if len(points) >= 1000:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print("🎉 Retrain + Reindex completed (62-class).")
    print("👉 IMPORTANT: update your inference load_model() to use num_classes=62 (or read it from EMNIST classes).")


if __name__ == "__main__":
    train_and_index()
