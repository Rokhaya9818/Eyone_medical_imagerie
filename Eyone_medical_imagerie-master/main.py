from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from schemas import *
from utils import *
import torchvision.transforms as T
from timm import create_model
from safetensors.torch import load_model

# Configuration de base


app = FastAPI(
    title="Medical imagery diagnostic classification API",
    description="API for multi-label classification of chest X-rays using Swin Transformer",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle Pydantic pour la réponse



# Endpoints
@app.on_event("startup")
async def loading_model():
    """Charge le modèle au démarrage de l'application"""
    global model, transform
    
    try:
        logger.info("Loading model...")
        model = create_model(MODEL_NAME, num_classes=len(CLASS_NAMES))
        load_model(model, MODEL_PATH)
        model.eval()
        
        transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    
@app.get("/", response_model=HealthCheck, tags=["Monitoring"])
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ready": model is not None,
        "documentation": "/docs"
    }

@app.post("/predict_cardio_pneumology", response_model=PredictionResult, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Effectue une prédiction sur une image de radiographie thoracique
    
    Args:
        file: Fichier image (JPEG/PNG) à analyser
        
    Returns:
        Un objet contenant:
        - predictions: Probabilités et prédictions pour chaque classe
        - diagnosis: Liste des diagnostics positifs
        - top_prediction: Diagnostic avec la plus haute probabilité
    """
    try:
        # Vérification du type de fichier
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Seuls les fichiers JPEG/PNG sont acceptés")

        # Lecture et prétraitement
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # Prédiction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            positive_diagnoses.append({
                
                class_name: float(prob)
            })

        logger.info("Prediction successful")
        ordered_data = sorted(
                [{k: v} for item in positive_diagnoses for k, v in item.items()],
                key=lambda x: list(x.values())[0],
                reverse=True
            )
        
        return PredictionResult(results = ordered_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
