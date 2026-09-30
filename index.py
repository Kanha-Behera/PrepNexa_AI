from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features import extract_features, vectorize_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "interview_scorer.joblib"


def load_model():
    if not MODEL_PATH.exists():
        from src.train import main as train_model

        train_model()
    return joblib.load(MODEL_PATH)


model = load_model()
app = FastAPI(title="Synapse AI Interview ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    question: str
    answer: str
    expected_concepts: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictionRequest):
    reference_text = req.expected_concepts or req.question
    features = extract_features(req.answer, reference_text)
    score = float(model.predict(vectorize_rows([features]))[0])
    return {"score": round(max(0, min(10, score)), 2), "features": features}
