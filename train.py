from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from .features import extract_features, vectorize_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/interview_responses.csv"
MODEL = ROOT / "models/interview_scorer.joblib"


def build_training_data(data_path: str | Path = DATA):
    df = pd.read_csv(data_path)
    required_columns = {"question", "answer", "overall_score"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df[["question", "answer", "overall_score"]].copy()


def main():
    df = build_training_data()
    rows = [extract_features(a, q) for q, a in zip(df["question"], df["answer"])]
    X = vectorize_rows(rows)
    y = df["overall_score"].astype(float).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }
    print(metrics)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)


if __name__ == "__main__":
    main()
