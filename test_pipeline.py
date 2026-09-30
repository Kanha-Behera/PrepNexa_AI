import pandas as pd

from src.train import build_training_data


def test_build_training_data_uses_current_dataset_schema():
    df = build_training_data()

    assert {"question", "answer", "overall_score"}.issubset(df.columns)
    assert len(df) > 0
    assert df["overall_score"].between(0, 10).all()
