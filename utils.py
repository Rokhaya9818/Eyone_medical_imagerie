from dotenv import load_dotenv
import io
import torch
import logging
from PIL import Image
import torchvision.transforms as T
from fastapi import HTTPException
from torchvision import models
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv() 

# Configuration pour Swin Transformer (déjà existant)
MODEL_NAME_SWIN = 'swin_s3_base_224'
CLASS_NAMES_SWIN = [
    "no finding", "atelectasis", "cardiomegaly", "effusion", "infiltration",
    "mass", "nodule", "pneumonia", "pneumothorax", "consolidation",
    "edema", "emphysema", "fibrosis", "pleural_checking", "hernia"
]
MODEL_PATH_SWIN = f'./best_{MODEL_NAME_SWIN}-pascal-cardio/model.safetensors'

# Configuration pour le modèle cervical (ResNet50)
MODEL_PATH_CERVICAL = '/home/ubuntu/best_model.pth'
CLASS_NAMES_CERVICAL = [
    "Col Sain", "CIN1(Lésion légère)", "CIN2(Lésion Modérée)", "CIN3(Lésion sévère)"
]
IMG_SIZE = (224, 224)
THRESHOLD = 0.0

# Transformation pour le modèle Swin Transformer
transform_swin = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Transformation pour le modèle cervical
transform_cervical = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # pour ResNet50
])

# Charger le modèle Swin Transformer
def load_swin_model():
    model = create_model(MODEL_NAME_SWIN, num_classes=len(CLASS_NAMES_SWIN))
    load_model(model, MODEL_PATH_SWIN)
    model.eval()
    return model

# Charger le modèle cervical (ResNet50)
def load_cervical_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES_CERVICAL))
    model.load_state_dict(torch.load(MODEL_PATH_CERVICAL, map_location=torch.device('cpu')))
    model.eval()
    return model

# Fonction pour prétraiter l'image pour chaque modèle
def preprocess_image(image_bytes: bytes, model_type="cervical"):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if model_type == "swin":
            return transform_swin(image).unsqueeze(0)
        elif model_type == "cervical":
            return transform_cervical(image).unsqueeze(0)
        else:
            raise ValueError("Model type not recognized")
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")
