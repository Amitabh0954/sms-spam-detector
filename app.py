

from __future__ import annotations

import json
import logging
import pickle
import re
import string
import time
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vocab import Vocabulary
from intelligence import Intelligence
from dotenv import load_dotenv

load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY")

_intel = Intelligence(api_key=_groq_key)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
MAX_WORDS   = 10_000
MAX_SEQ_LEN = 70

class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_words: int = MAX_WORDS):
        self.max_words = max_words
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        for word, _ in counter.most_common(self.max_words - 2):
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str], max_len: int = MAX_SEQ_LEN):
        return np.array([self.encode(t, max_len) for t in texts], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.word2idx)


log = logging.getLogger(__name__)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = Path("artifacts")
EMBEDDING_DIM = 300

import nltk
from nltk.corpus import stopwords

for pkg in ("stopwords",):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_stop_words = set(stopwords.words("english"))
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE     = re.compile(r"\d+")
_PUNCT_TBL  = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    text = text.translate(_PUNCT_TBL)
    tokens = [t for t in text.split() if len(t) > 1]
    tokens = [t for t in tokens if t not in _stop_words or t in ("free", "win", "prize")]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Model name normalizer  ← FIXES the metrics matching
# ---------------------------------------------------------------------------
def _normalize_model_name(raw: str) -> str:
    s = raw.lower().replace(" ", "").replace("+", "")
    if "tfidf" in s:             return "mnb_tfidf"
    if "countvectorizer" in s:   return "mnb_count"
    if s == "bilstm":            return "bilstm"
    if s == "simplernn":         return "simplernn"
    if s == "lstm":              return "lstm"
    if s == "gru":               return "gru"
    return s


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(rnn_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (rnn_out * weights).sum(dim=1)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_matrix, rnn_type="lstm",
                 hidden=128, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix, dtype=torch.float32), requires_grad=False)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type

        rnn_cls = {"simplernn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            EMBEDDING_DIM, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rnn_out_dim = hidden * (2 if bidirectional else 1)
        self.attn   = AttentionPool(rnn_out_dim)
        self.drop   = nn.Dropout(0.4)
        self.fc1    = nn.Linear(rnn_out_dim, 64)
        self.bn     = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        emb    = self.emb_drop(self.embedding(x))
        out, _ = self.rnn(emb)
        pooled = self.attn(out)
        pooled = self.drop(pooled)
        h      = torch.relu(self.bn(self.fc1(pooled)))
        return self.fc_out(h).squeeze(1)


DEEP_MODEL_CONFIGS = {
    "simplernn": dict(rnn_type="simplernn", bidirectional=False, num_layers=1),
    "lstm":      dict(rnn_type="lstm",      bidirectional=False, num_layers=2),
    "gru":       dict(rnn_type="gru",       bidirectional=False, num_layers=2),
    "bilstm":    dict(rnn_type="lstm",      bidirectional=True,  num_layers=2),
}

MNB_VECTORIZER_MAP = {
    "mnb_tfidf":  "tfidf_vectorizer",
    "mnb_count":  "count_vectorizer",
}


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------
class ArtifactStore:
    def __init__(self):
        self._vocab       = None
        self._emb_matrix  = None
        self._thresholds: Dict[str, float] = {}
        self._deep_models: Dict[str, RNNClassifier] = {}
        self._mnb_models:  Dict[str, tuple] = {}
        self._results_df  = None
        self._loaded      = False

    def load(self):
        if self._loaded:
            return

        log.info("Loading artifacts from %s …", ARTIFACTS_DIR)

        with open(ARTIFACTS_DIR / "vocab.pkl", "rb") as f:
            self._vocab = pickle.load(f)
        log.info("Vocab: %d words", len(self._vocab))

        self._emb_matrix = np.load(str(ARTIFACTS_DIR / "emb_matrix.npy"))
        self._thresholds = json.loads((ARTIFACTS_DIR / "thresholds.json").read_text())
        log.info("Thresholds: %s", self._thresholds)

        for mnb_name, vec_name in MNB_VECTORIZER_MAP.items():
            clf_path = ARTIFACTS_DIR / f"{mnb_name}.pkl"
            vec_path = ARTIFACTS_DIR / f"{vec_name}.pkl"
            if clf_path.exists() and vec_path.exists():
                with open(clf_path, "rb") as f:
                    clf = pickle.load(f)
                with open(vec_path, "rb") as f:
                    vec = pickle.load(f)
                self._mnb_models[mnb_name] = (clf, vec)
                log.info("Loaded MNB: %s", mnb_name)
            else:
                log.warning("Missing: %s or %s", clf_path, vec_path)

        for name, cfg in DEEP_MODEL_CONFIGS.items():
            pt = ARTIFACTS_DIR / f"{name}.pt"
            if pt.exists():
                try:
                    model = RNNClassifier(
                        len(self._vocab), self._emb_matrix,
                        rnn_type=cfg["rnn_type"],
                        bidirectional=cfg["bidirectional"],
                        num_layers=cfg["num_layers"],
                    ).to(DEVICE)
                    model.load_state_dict(torch.load(str(pt), map_location=DEVICE))
                    model.eval()
                    self._deep_models[name] = model
                    log.info("Loaded deep model: %s", name)
                except Exception as e:
                    log.warning("Could not load %s: %s", name, e)

        results_path = ARTIFACTS_DIR / "results.csv"
        if results_path.exists():
            self._results_df = pd.read_csv(results_path)

        self._loaded = True
        log.info("Ready — MNB: %d, Deep: %d", len(self._mnb_models), len(self._deep_models))

    def available_models(self) -> List[str]:
        return list(self._mnb_models.keys()) + list(self._deep_models.keys())

    def threshold_for(self, name: str) -> float:
        return self._thresholds.get(name, 0.5)

    def predict(self, model_name: str, text: str) -> dict:
        if model_name not in self.available_models():
            raise ValueError(f"Model '{model_name}' not available. "
                             f"Available: {self.available_models()}")

        if model_name in self._mnb_models:
            clf, vec = self._mnb_models[model_name]
            prob = float(clf.predict_proba(vec.transform([preprocess(text)]))[0, 1])
        else:
            model = self._deep_models[model_name]
            ids = self._vocab.encode_batch([preprocess(text)])
            X   = torch.tensor(ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                prob = float(torch.sigmoid(model(X)).item())

        thresh = self.threshold_for(model_name)
        return {
            "probability": round(prob, 4),
            "threshold":   round(thresh, 4),
            "label":       "spam" if prob >= thresh else "ham",
        }

    def results_as_json(self) -> list:
        return [] if self._results_df is None else self._results_df.to_dict("records")


_store = ArtifactStore()

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SMS Spam Detector API",
    description="ML spam detection + Groq-powered explainability",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    _store.load()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text:  str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="bilstm")

class PredictResponse(BaseModel):
    text:        str
    model:       str
    label:       str
    probability: float
    confidence:  float
    threshold:   float
    latency_ms:  float

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    model: str = Field(default="bilstm")

class BatchPredictResponse(BaseModel):
    model:    str
    results:  List[PredictResponse]
    total_ms: float

class ExplainRequest(BaseModel):
    text:        str
    label:       str
    probability: float
    model:       str = "bilstm"

class BatchAnalyzeRequest(BaseModel):
    results: List[dict]
    model:   str = "bilstm"

class RewriteCheckRequest(BaseModel):
    text: str

class ModelInfo(BaseModel):
    name:      str
    available: bool
    type:      str
    threshold: Optional[float]
    metrics:   Optional[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "device":  str(DEVICE),
        "models":  _store.available_models(),
        "llm":     "groq/llama-3.3-70b",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    available = _store.available_models()
    if req.model not in available:
        raise HTTPException(400, f"Model '{req.model}' not available. Choose from: {available}")

    t0     = time.perf_counter()
    result = _store.predict(req.model, req.text)
    prob   = result["probability"]
    conf   = prob if result["label"] == "spam" else 1 - prob

    return PredictResponse(
        text=req.text,
        model=req.model,
        label=result["label"],
        probability=prob,
        confidence=round(conf, 4),
        threshold=result["threshold"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    available = _store.available_models()
    if req.model not in available:
        raise HTTPException(400, f"Model '{req.model}' not available. Choose from: {available}")

    t0      = time.perf_counter()
    results = []
    for text in req.texts:
        try:
            r    = _store.predict(req.model, text)
            prob = r["probability"]
            conf = prob if r["label"] == "spam" else 1 - prob
            results.append(PredictResponse(
                text=text, model=req.model,
                label=r["label"], probability=prob,
                confidence=round(conf, 4),
                threshold=r["threshold"], latency_ms=0.0,
            ))
        except Exception as e:
            log.warning("Batch predict error for '%s': %s", text[:40], e)

    return BatchPredictResponse(
        model=req.model,
        results=results,
        total_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/intelligence/explain")
async def explain(req: ExplainRequest):
    try:
        result = await asyncio.wait_for(
            _intel.explain_prediction(
                req.text, req.label, req.probability, req.model
            ),
            timeout=10
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intelligence/analyze")
async def analyze(req: BatchAnalyzeRequest):
    result = await _intel.analyze_batch(req.results, req.model)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.post("/intelligence/rewrite-check")
async def rewrite_check(req: RewriteCheckRequest):
    result = await _intel.detect_evasion(req.text)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    all_names = ["mnb_tfidf", "mnb_count", "simplernn", "lstm", "gru", "bilstm"]
    available = _store.available_models()

    # Build results_map using robust name normalizer
    results_map: Dict[str, dict] = {}
    for row in _store.results_as_json():
        key = _normalize_model_name(row.get("name", ""))
        results_map[key] = row

    infos = []
    for name in all_names:
        row     = results_map.get(name)
        metrics = None
        if row:
            metrics = {
                "accuracy":  round(row.get("accuracy",  0), 4),
                "f1":        round(row.get("f1",        0), 4),
                "precision": round(row.get("precision", 0), 4),
                "recall":    round(row.get("recall",    0), 4),
                "roc_auc":   round(row.get("roc_auc",   0), 4),
            }
        infos.append(ModelInfo(
            name=name,
            available=name in available,
            type="sklearn" if name.startswith("mnb") else "deep",
            threshold=_store.threshold_for(name) if name in available else None,
            metrics=metrics,
        ))
    return infos


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    all_names = ["mnb_tfidf", "mnb_count", "simplernn", "lstm", "gru", "bilstm"]
    if model_name not in all_names:
        raise HTTPException(404, f"Unknown model: {model_name}")

    available = _store.available_models()

    # Build results_map using robust name normalizer
    results_map: Dict[str, dict] = {}
    for row in _store.results_as_json():
        key = _normalize_model_name(row.get("name", ""))
        results_map[key] = row

    row     = results_map.get(model_name)
    metrics = None
    if row:
        metrics = {
            "accuracy":   round(row.get("accuracy",   0), 4),
            "f1":         round(row.get("f1",         0), 4),
            "precision":  round(row.get("precision",  0), 4),
            "recall":     round(row.get("recall",     0), 4),
            "roc_auc":    round(row.get("roc_auc",    0), 4),
            "threshold":  round(row.get("threshold",  0.5), 4),
            "train_time": round(row.get("train_time", 0), 3),
        }

    return ModelInfo(
        name=model_name,
        available=model_name in available,
        type="sklearn" if model_name.startswith("mnb") else "deep",
        threshold=_store.threshold_for(model_name) if model_name in available else None,
        metrics=metrics,
    )


@app.get("/metrics")
async def get_metrics():
    rows = _store.results_as_json()
    if not rows:
        raise HTTPException(404, "No results.csv found. Run training first.")
    return {"models": rows}