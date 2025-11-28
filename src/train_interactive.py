import cv2
import torch
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.model import NeuroOCR
from src.infer import preprocess_image # Re-use our crop function
from config.config import *
import uuid

def learn_new_character(image_path, label):
    print(f"Teaching the AI that this image is: '{label}'")
    
    # 1. Load the trained brain (CNN)
    model = NeuroOCR(num_classes=47, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load("neuro_ocr_model.pth"))
    model.eval()
    
    # 2. Process the image to get the vector
    # We assume the image contains ONE character for teaching
    original_img, crops, _ = preprocess_image(image_path)
    
    if len(crops) == 0:
        print("Error: Could not find a character in the image.")
        return
    
    # Take the largest crop (assuming it's the main character)
    target_crop = crops[0] 
    
    tensor_img = torch.tensor(target_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, embedding = model(tensor_img)
        
    # 3. Upload to Qdrant
    client = QdrantClient("localhost", port=6333)
    
    # Create a unique ID for this new point
    point_id = str(uuid.uuid4())
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding.cpu().numpy()[0].tolist(),
                payload={"character": label}
            )
        ]
    )
    
    print(f"Success! The AI has now learned '{label}'. Run infer.py again to test it.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/teach.py <image_path> <character_name>")
    else:
        learn_new_character(sys.argv[1], sys.argv[2])