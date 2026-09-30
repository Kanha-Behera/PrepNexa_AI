from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .api import model
from .features import extract_features, vectorize_rows

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/interview_responses.csv"
OUTPUT_PATH = ROOT / "data/processed/scored_responses.csv"


def score_dataset(input_path: str | Path = DATA_PATH, output_path: str | Path = OUTPUT_PATH):
    df = pd.read_csv(input_path)
    if not {"question", "answer", "overall_score"}.issubset(df.columns):
        raise ValueError("Input CSV must include 'question', 'answer', and 'overall_score' columns.")

    rows = [extract_features(answer, question) for question, answer in zip(df["question"], df["answer"])]
    features = pd.DataFrame(rows)
    scores = model.predict(vectorize_rows(rows))
    features["predicted_score"] = scores
    features["actual_score"] = df["overall_score"].astype(float).to_numpy()

    output = df.copy()
    output["predicted_score"] = scores
    output["error"] = output["predicted_score"] - output["overall_score"]
    output.to_csv(output_path, index=False)

    print(json.dumps({
        "rows_scored": int(len(output)),
        "output_path": str(output_path),
        "average_error": round(float(output["error"].abs().mean()), 4),
    }, indent=2))

    return output


if __name__ == "__main__":
    score_dataset()
