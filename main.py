from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from schemas import *
from utils import *
import torchvision.transforms as T


# Configuration de base
app = FastAPI(
    title="Medical imagery diagnostic classification API",
    description="API for multi-label classification of chest X-rays using Swin Transformer and cervical, kidney and acute lymphoblastic leukemia cancer detection",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.on_event("startup")
async def loading_model():
    """Charge les modèles au démarrage de l'application"""
    global model_swin, model_Lung_and_Colon, model_KIDNEY, model_ACUTE_LYMPHOBLASTIC_LEUKEMIA, model_cervical, transform_swin, transform_cervical
    
    try:
        logger.info("Loading models...")
        
        # Charger le modèle Swin Transformer
        model_swin = load_swin_model()
        
        # Charger le modèle cervical (ResNet50 ou autre modèle)
        model_cervical = load_resnet_model(CLASS_NAME=CLASS_NAMES_CERVICAL, MODEL_PATH=MODEL_PATH_CERVICAL)

        # Charger le modèle cervical (ResNet50 ou autre modèle)
        model_KIDNEY = load_resnet_model(CLASS_NAME=CLASS_NAMES_KIDNEY, MODEL_PATH=MODEL_PATH_KIDNEY)

        # Charger le modèle cervical (ResNet50 ou autre modèle)
        model_ACUTE_LYMPHOBLASTIC_LEUKEMIA = load_resnet_model(CLASS_NAME=CLASS_NAMES_ACUTE_LYMPHOBLASTIC_LEUKEMIA, MODEL_PATH=MODEL_PATH_ACUTE_LYMPHOBLASTIC_LEUKEMIA)
        
        # Charger le modèle cervical (ResNet50 ou autre modèle)
        model_Lung_and_Colon = load_resnet_model(CLASS_NAME=CLASS_NAMES_Lung_and_Colon, MODEL_PATH=MODEL_PATH_Lung_and_Colon)

        model_swin.eval()
        model_cervical.eval()
        model_ACUTE_LYMPHOBLASTIC_LEUKEMIA.eval()
        model_KIDNEY.eval()
        
        model_Lung_and_Colon.eval()
        
        # Transformation pour Swin Transformer
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
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.get("/", tags=["Monitoring"])
async def health_check():
    """Vérifie l'état de l'API et des modèles"""
    return {
        "status": "healthy",
        "model_loaded": {
            "swin": model_swin is not None,
            "cervical": model_cervical is not None,
            "ACUTE_LYMPHOBLASTIC_LEUKEMIA": model_ACUTE_LYMPHOBLASTIC_LEUKEMIA is not None,
            "Lung and Colon": model_Lung_and_Colon is not None,
            "KIDNEY": model_KIDNEY is not None
        },
        "ready": model_swin is not None and model_cervical is not None,
        "documentation": "/docs"
    }

@app.post("/predict_cardio_pneumology", response_model=PredictionResult, tags=["Predictions"])
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
        input_tensor = preprocess_image(image_bytes, model_type="swin")

        # Prédiction
        with torch.no_grad():
            logits = model_swin(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES_SWIN, probabilities)):
            positive_diagnoses.append({class_name: float(prob)})

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

@app.post("/predict_cervical_cancer", response_model=PredictionResult, tags=["Predictions"])
async def predict_cervical_cancer(file: UploadFile = File(...)):
    
    try:
        # Vérification du type de fichier
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Seuls les fichiers JPEG/PNG sont acceptés")

        # Lecture et prétraitement
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes, model_type="cervical")

        # Prédiction
        with torch.no_grad():
            logits = model_cervical(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES_CERVICAL, probabilities)):
            positive_diagnoses.append({class_name: float(prob)})

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


@app.post("/predict_ACUTE_LYMPHOBLASTIC_LEUKEMIA_cancer", response_model=PredictionResult, tags=["Predictions"])
async def predict_ACUTE_LYMPHOBLASTIC_LEUKEMIA_cancer(file: UploadFile = File(...)):
    """Leucémie aiguë lymphoblastique"""
    try:
        # Vérification du type de fichier
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Seuls les fichiers JPEG/PNG sont acceptés")

        # Lecture et prétraitement
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes, model_type="cervical")

        # Prédiction
        with torch.no_grad():
            logits = model_ACUTE_LYMPHOBLASTIC_LEUKEMIA(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        print(probs)
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES_ACUTE_LYMPHOBLASTIC_LEUKEMIA, probabilities)):
            
            
            positive_diagnoses.append({class_name: float(prob)})

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



@app.post("/predict_KIDNEY_cancer", response_model=PredictionResult, tags=["Predictions"])
async def predict_KIDNEY_cancer(file: UploadFile = File(...)):
    """KIDNEY (Cancer du rein)"""
    try:
        # Vérification du type de fichier
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Seuls les fichiers JPEG/PNG sont acceptés")

        # Lecture et prétraitement
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes, model_type="cervical")

        # Prédiction
        with torch.no_grad():
            logits = model_KIDNEY(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        print(probs)
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES_KIDNEY, probabilities)):
            
            
            positive_diagnoses.append({class_name: float(prob)})

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



@app.post("/predict_Lung_and_Colon_cancer", response_model=PredictionResult, tags=["Predictions"])
async def predict_Lung_and_Colon_cancer(file: UploadFile = File(...)):
    """Lung and Colon cancer"""
    try:
        # Vérification du type de fichier
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Seuls les fichiers JPEG/PNG sont acceptés")

        # Lecture et prétraitement
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes, model_type="cervical")

        # Prédiction
        with torch.no_grad():
            logits = model_Lung_and_Colon(input_tensor)
            probs = torch.sigmoid(logits)
            probabilities = probs.squeeze().numpy()

        positive_diagnoses = []
        
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES_Lung_and_Colon, probabilities)):
            
            
            positive_diagnoses.append({class_name: float(prob)})

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
