# Synapse AI — Interview Performance ML

End-to-end ML service for scoring interview answers using the current interview dataset.

## Data model
The project now uses the live dataset in `data/raw/interview_responses.csv`, which contains:
- `question`
- `answer`
- `role`
- `category`
- `technical_score`
- `relevance_score`
- `communication_score`
- `overall_score`

The model predicts the `overall_score` for a candidate response based on the question/answer pair.

## MVP
- Input: interview question and candidate answer
- Features: answer length, keyword coverage, filler-word rate, semantic similarity (baseline TF-IDF)
- Target: interview quality score (0–10)
- Model: baseline RandomForestRegressor
- API: FastAPI `/predict`

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api:app --reload
```

## Example request
```json
{
  "question": "How do you handle missing values in a dataset?",
  "answer": "I would use imputation and check the missingness pattern before modeling."
}
```
