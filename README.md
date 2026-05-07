# SpamShield — SMS Spam Detector

A production-grade SMS spam detection system combining classical ML and deep learning models with a Groq-powered AI explainability layer.

**Live Demo:** _coming soon_ · **Backend:** FastAPI · **Frontend:** React + Vite · **LLM:** LLaMA 3.3 70B via Groq

---

## Overview

SpamShield detects spam in SMS messages using 6 trained models — from Multinomial Naive Bayes to Bidirectional LSTM — and explains every prediction using a Groq-hosted LLM. It includes a full-stack web interface with batch scanning, model comparison, and evasion detection.

---

## Features

- **6 ML/DL models** — MNB (TF-IDF & Count), SimpleRNN, LSTM, GRU, BiLSTM
- **AI Explainability** — LLaMA 3.3 70B explains every prediction via Groq
- **Evasion Detection** — detects obfuscation tricks spammers use to bypass filters
- **Batch Scanning** — scan hundreds of messages at once with CSV export
- **Model Comparison** — side-by-side metrics with sortable leaderboard
- **GloVe Embeddings** — 300-dimensional pretrained word vectors
- **Custom Thresholds** — per-model optimized decision boundaries

---

## Model Performance

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| MNB · TF-IDF | 98.6% | 94.4% | 99.2% | 90.1% | 99.4% |
| MNB · Count | 98.6% | 94.5% | 97.6% | 91.6% | 98.5% |
| Simple RNN | 98.1% | 92.4% | 91.7% | 93.1% | 99.2% |
| LSTM | 98.5% | 93.7% | 97.5% | 90.1% | 99.5% |
| GRU | 98.6% | 94.4% | 98.3% | 90.8% | 99.6% |
| **BiLSTM** | **98.2%** | **92.8%** | **92.4%** | **93.1%** | **99.4%** |

> Trained on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) · 5,574 messages · 87/13 ham/spam split

---

## Project Structure

```
sms-spam-detector/
├── app.py                  # FastAPI backend — all endpoints
├── intelligence.py         # Groq LLM layer (explain, analyze, evasion)
├── vocab.py                # Custom vocabulary + GloVe loader
├── artifacts/
│   ├── bilstm.pt           # Trained PyTorch model weights
│   ├── lstm.pt
│   ├── gru.pt
│   ├── simplernn.pt
│   ├── mnb_tfidf.pkl       # Sklearn MNB classifiers
│   ├── mnb_count.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── count_vectorizer.pkl
│   ├── vocab.pkl
│   ├── emb_matrix.npy      # GloVe embedding matrix
│   ├── thresholds.json     # Per-model optimal thresholds
│   └── results.csv         # Training metrics
└── sms-spam-frontend/      # React + Vite frontend
    ├── src/
    │   ├── pages/          # Predict, Batch, Models, Intelligence
    │   ├── components/     # Layout, PredictResult, ExplainCard
    │   └── lib/api.js      # API client
    └── vite.config.js      # Dev proxy → localhost:8000
```

---

## Tech Stack

**Backend**
- Python 3.11, FastAPI, Uvicorn
- PyTorch (RNN models), Scikit-learn (MNB)
- NLTK, NumPy, Pandas
- Groq SDK (LLaMA 3.3 70B)
- GloVe 6B 300d embeddings

**Frontend**
- React 18, Vite
- React Router DOM v6
- Tailwind CSS v3

---

## Local Setup

### Backend

```bash
git clone https://github.com/Amitabh0954/sms-spam-detector.git
cd sms-spam-detector

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

```bash
uvicorn app:app --reload
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend

```bash
cd sms-spam-frontend
npm install
npm run dev
# App running at http://localhost:5173
# Proxies /api/* → http://localhost:8000
```

> Make sure the FastAPI backend is running on port 8000 before starting the frontend.

### Pages
- `/` — Single message prediction
- `/batch` — Multi-message batch scan
- `/models` — Model comparison & metrics
- `/explain` — AI explainability (Groq LLaMA 3.3 70B)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Single message prediction |
| POST | `/predict/batch` | Batch prediction (up to 100 messages) |
| POST | `/intelligence/explain` | AI explanation of a prediction |
| POST | `/intelligence/analyze` | AI batch analysis |
| POST | `/intelligence/rewrite-check` | Spam evasion detection |
| GET | `/models` | All models with metrics |
| GET | `/models/{name}` | Single model info |
| GET | `/metrics` | Raw training metrics |
| GET | `/health` | API health check |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! You have been selected for a cash prize. Call now!", "model": "bilstm"}'
```

```json
{
  "label": "spam",
  "probability": 0.9821,
  "confidence": 0.9821,
  "threshold": 0.49,
  "latency_ms": 12.4
}
```

---

## Model Training

The models were trained in the Jupyter notebook:

```bash
jupyter notebook sms-spam-detection-mnb-lstm-gru-bi-lstm.ipynb
```

Training pipeline:
1. Preprocess text (lowercase, URL/number tokens, stopword removal)
2. Build vocabulary from training corpus
3. Load GloVe 6B 300d embeddings → embedding matrix
4. Train each model with optimized thresholds via F1 maximization
5. Export artifacts to `artifacts/`

---

## License

MIT