from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import logging
from typing import Optional
from PIL import Image
from io import BytesIO
import tensorflow as tf

from src.pipeline.predict_pipeline import PredictPipeline
from src.constants import CLASS_DISPLAY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soilnet_api")

app = FastAPI(
    title="SoilNet – Soil Classification API",
    description="Classify soil types from images using a lightweight CNN.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------
# Load model at startup
# ------------------------------------------------------------
pipeline = PredictPipeline()
logger.info("PredictPipeline loaded successfully.")

# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict
    class_list: list


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "SoilNet – Soil Classification API v1.0"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if pipeline.model is not None else "unhealthy",
        model_loaded=pipeline.model is not None,
        num_classes=len(pipeline.class_names) if pipeline.class_names else 0,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_arr = np.array(img)

        result = pipeline.predict(img_arr)

        return PredictionResponse(
            predicted_class=result["class"],
            confidence=result["confidence"],
            all_probabilities=result["all_probs"],
            class_list=pipeline.display_names,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))