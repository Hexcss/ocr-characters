import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1536
EPOCHS = 20
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256
COLLECTION_NAME = "ocr_reference_book"

# Toggle Spell Checker
SPELL_CHECK = False