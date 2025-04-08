from dotenv import load_dotenv
import io
import torch
import logging
from PIL import Image
import torchvision.transforms as T
from fastapi import HTTPException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv() 

MODEL_NAME = 'swin_s3_base_224'
CLASS_NAMES = [
    "no finding", "atelectasis", "cardiomegaly", "effusion", "infiltration",
    "mass", "nodule", "pneumonia", "pneumothorax", "consolidation",
    "edema", "emphysema", "fibrosis", "pleural_checking", "hernia"
]
IMG_SIZE = (224, 224)
THRESHOLD = 0.0
MODEL_PATH = f'./best_{MODEL_NAME}-pascal-cardio/model.safetensors'
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convertir les bytes de l'image en tenseur préprocessé"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0)  # Ajouter une dimension de batch
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")
