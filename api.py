from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .features import extract_features, vectorize_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/interview_scorer.joblib"
STATIC_DIR = ROOT / "static"


def load_model():
    if not MODEL_PATH.exists():
        from .train import main as train_model

        train_model()
    return joblib.load(MODEL_PATH)


model = load_model()
app = FastAPI(title="Synapse AI Interview ML API")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictionRequest(BaseModel):
    question: str
    answer: str
    expected_concepts: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictionRequest):
    reference_text = req.expected_concepts or req.question
    features = extract_features(req.answer, reference_text)
    score = float(model.predict(vectorize_rows([features]))[0])
    return {"score": round(max(0, min(10, score)), 2), "features": features}
