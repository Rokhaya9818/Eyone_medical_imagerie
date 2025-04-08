from pydantic import BaseModel
from typing import List

class PredictionResult(BaseModel):
    results: List

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    ready: bool
    documentation: str

    
class Response(BaseModel):
    status:int
    message:str
