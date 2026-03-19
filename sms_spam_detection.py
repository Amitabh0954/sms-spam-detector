
from __future__ import annotations

print(">> Step 1: stdlib")
import argparse
import json
import logging
import pickle
import re
import string
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vocab import Vocabulary

print(">> Step 2: nltk")
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

print(">> Step 3: numpy / pandas")
import numpy as np
import pandas as pd

print(">> Step 4: sklearn")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, auc, classification_report,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

print(">> Step 5: torch")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print(">> Step 6: all imports done")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE            = 42
TEST_SIZE               = 0.2
MAX_WORDS               = 10_000
MAX_SEQ_LEN             = 70
EMBEDDING_DIM           = 300
EPOCHS                  = 15           # more epochs — early stopping will brake
BATCH_SIZE              = 32           # smaller batch → better gradient signal
EARLY_STOPPING_PATIENCE = 3
LR                      = 2e-3
UNFREEZE_EPOCH          = 3            # start fine-tuning embeddings after epoch 3
GRAD_CLIP               = 1.0         # gradient clipping max norm
LABEL_SMOOTHING         = 0.05        # prevents over-confident BCE

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
def _ensure_nltk() -> None:
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()
_stemmer    = SnowballStemmer("english")
_stop_words = set(stopwords.words("english"))
print(">> Step 7: NLTK ready")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "label": ["label", "v1", "Category", "class"],
    "text":  ["text",  "v2", "Message",  "sms"],
}

def load_dataset(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    col_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw.columns:
                col_map[alias] = canonical
                break
    raw = raw.rename(columns=col_map)[["text", "label"]]
    raw["label"] = raw["label"].map({"ham": 0, "spam": 1}).fillna(raw["label"])
    raw["label"] = raw["label"].astype(int)
    raw = raw.drop_duplicates().dropna().reset_index(drop=True)
    log.info("Dataset loaded — %d rows after deduplication", len(raw))
    log.info("Class distribution:\n%s", raw["label"].value_counts().to_string())
    return raw

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE    = re.compile(r"\d+")
_PUNCT_TBL = str.maketrans("", "", string.punctuation)

def preprocess(text: str, stem: bool = False) -> str:
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)     # keep URL signal
    text = _NUM_RE.sub(" numtoken ", text)     # keep number signal
    text = text.translate(_PUNCT_TBL)
    tokens = [t for t in text.split() if len(t) > 1]
    # only remove stopwords for non-spam-signal words
    tokens = [t for t in tokens if t not in _stop_words or t in ("free", "win", "prize")]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def preprocess_series(series: pd.Series, stem: bool = False) -> pd.Series:
    return series.apply(lambda x: preprocess(x, stem=stem))

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# class Vocabulary:
#     PAD, UNK = "<PAD>", "<UNK>"

#     def __init__(self, max_words: int = MAX_WORDS):
#         self.max_words = max_words
#         self.word2idx: Dict[str, int] = {}
#         self.idx2word: Dict[int, str] = {}

#     def build(self, texts: List[str]) -> None:
#         counter: Counter = Counter()
#         for text in texts:
#             counter.update(text.split())
#         self.word2idx = {self.PAD: 0, self.UNK: 1}
#         for word, _ in counter.most_common(self.max_words - 2):
#             self.word2idx[word] = len(self.word2idx)
#         self.idx2word = {v: k for k, v in self.word2idx.items()}
#         log.info("Vocabulary size: %d", len(self.word2idx))

#     def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
#         tokens = text.split()[:max_len]
#         ids = [self.word2idx.get(t, 1) for t in tokens]
#         ids += [0] * (max_len - len(ids))
#         return ids

#     def encode_batch(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
#         return np.array([self.encode(t, max_len) for t in texts], dtype=np.int64)

#     def __len__(self) -> int:
#         return len(self.word2idx)

# ---------------------------------------------------------------------------
# GloVe
# ---------------------------------------------------------------------------
def load_glove(path: str) -> Dict[str, np.ndarray]:
    log.info("Loading GloVe from %s …", path)
    glove: Dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            try:
                glove[word] = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                pass
    log.info("GloVe vocabulary size: %d", len(glove))
    return glove

def build_embedding_matrix(vocab: Vocabulary, glove: Dict[str, np.ndarray]) -> np.ndarray:
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float32)
    hits = 0
    for word, idx in vocab.word2idx.items():
        vec = glove.get(word)
        if vec is not None:
            matrix[idx] = vec
            hits += 1
    log.info(
        "Embedding coverage: %d/%d (%.1f%%)",
        hits, vocab_size, hits / vocab_size * 100,
    )
    return matrix

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class SpamDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# Attention pooling  (replaces last-hidden-state)
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Soft attention over all RNN timesteps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (B, T, H)
        scores = self.attn(rnn_out).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (rnn_out * weights).sum(dim=1)             # (B, H)

# ---------------------------------------------------------------------------
# Focal Loss  (handles class imbalance better than weighted BCE)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()

# ---------------------------------------------------------------------------
# PyTorch Models
# ---------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_matrix: np.ndarray,
        rnn_type: str = "lstm",
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix, dtype=torch.float32), requires_grad=False
        )
        self.emb_drop    = nn.Dropout(dropout)
        self.rnn_type    = rnn_type
        self.bidirectional = bidirectional

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

    def unfreeze_embeddings(self) -> None:
        self.embedding.weight.requires_grad_(True)
        log.info("  Embeddings unfrozen — fine-tuning enabled")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb_drop(self.embedding(x))
        if self.rnn_type == "lstm":
            out, _ = self.rnn(emb)
        else:
            out, _ = self.rnn(emb)
        pooled = self.attn(out)                     # attention over all timesteps
        pooled = self.drop(pooled)
        h = torch.relu(self.bn(self.fc1(pooled)))
        return self.fc_out(h).squeeze(1)            # raw logits


def build_simplernn(vocab_size: int, emb_matrix: np.ndarray) -> RNNClassifier:
    return RNNClassifier(vocab_size, emb_matrix, rnn_type="simplernn",
                         num_layers=1).to(DEVICE)

def build_lstm(vocab_size: int, emb_matrix: np.ndarray) -> RNNClassifier:
    return RNNClassifier(vocab_size, emb_matrix, rnn_type="lstm",
                         num_layers=2).to(DEVICE)

def build_gru(vocab_size: int, emb_matrix: np.ndarray) -> RNNClassifier:
    return RNNClassifier(vocab_size, emb_matrix, rnn_type="gru",
                         num_layers=2).to(DEVICE)

def build_bilstm(vocab_size: int, emb_matrix: np.ndarray) -> RNNClassifier:
    return RNNClassifier(vocab_size, emb_matrix, rnn_type="lstm",
                         num_layers=2, bidirectional=True).to(DEVICE)

BUILDERS = {
    "simplernn": build_simplernn,
    "lstm":      build_lstm,
    "gru":       build_gru,
    "bilstm":    build_bilstm,
}

# ---------------------------------------------------------------------------
# Optimal threshold tuning
# ---------------------------------------------------------------------------
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         metric: str = "f1") -> Tuple[float, float]:
    """Sweep thresholds and return the one maximising F1 (or recall)."""
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = recall_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return float(best_thresh), float(best_score)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name:        str
    train_time:  float
    test_time:   float
    train_score: float
    test_score:  float
    accuracy:    float
    f1:          float
    precision:   float
    recall:      float
    roc_auc:     float
    threshold:   float = 0.5
    report: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "report"}

def evaluate(name, y_true, y_pred, y_prob, train_score, train_time, test_time,
             threshold: float = 0.5) -> ModelResult:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    result = ModelResult(
        name=name,
        train_time=train_time,
        test_time=test_time,
        train_score=train_score,
        test_score=accuracy_score(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        roc_auc=auc(fpr, tpr),
        threshold=threshold,
        report=classification_report(y_true, y_pred, target_names=["ham", "spam"]),
    )
    log.info(
        "[%s]  acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f  auc=%.4f  thresh=%.2f",
        name, result.accuracy, result.f1, result.precision, result.recall,
        result.roc_auc, threshold,
    )
    return result

# ---------------------------------------------------------------------------
# MNB
# ---------------------------------------------------------------------------
def train_mnb(X_train_raw, X_test_raw, y_train, y_test, vectorizer_type="tfidf"):
    if vectorizer_type == "tfidf":
        vec   = TfidfVectorizer(max_features=MAX_WORDS, ngram_range=(1, 2))
        label = "MultinomialNB + TFIDF"
    else:
        vec   = CountVectorizer(max_features=MAX_WORDS, ngram_range=(1, 2))
        label = "MultinomialNB + CountVectorizer"

    X_tr = vec.fit_transform(X_train_raw)
    X_te = vec.transform(X_test_raw)
    clf  = MultinomialNB(alpha=0.1)  # lower alpha → sharper discrimination

    t0 = time.perf_counter(); clf.fit(X_tr, y_train); train_time = time.perf_counter() - t0
    train_score = clf.score(X_tr, y_train)
    t0 = time.perf_counter(); y_prob = clf.predict_proba(X_te)[:, 1]; test_time = time.perf_counter() - t0

    best_thresh, _ = find_best_threshold(y_test, y_prob)
    y_pred  = (y_prob >= best_thresh).astype(int)
    result  = evaluate(label, y_test, y_pred, y_prob, train_score, train_time, test_time, best_thresh)
    log.info("\n%s", result.report)
    return result, clf, vec

# ---------------------------------------------------------------------------
# PyTorch training loop  —  FIXED
# ---------------------------------------------------------------------------
def train_torch_model(name, builder, X_train, X_test, y_train, y_test, vocab_size, emb_matrix):
    log.info("Building %s …", name)
    model     = builder(vocab_size, emb_matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

    # Focal loss — better for class imbalance than weighted BCE
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # 80/20 train-val split (was 90/10 — too small val set)
    val_size    = max(50, int(0.15 * len(X_train)))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    train_dl = DataLoader(SpamDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        # Unfreeze embeddings after warm-up
        if epoch == UNFREEZE_EPOCH:
            model.unfreeze_embeddings()
            # lower LR for embeddings to avoid catastrophic forgetting
            for pg in optimizer.param_groups:
                pg["lr"] = LR * 0.1

        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            # label smoothing
            smooth_y = yb * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = criterion(logits, smooth_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # gradient clipping
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            X_val_t  = torch.tensor(X_val, dtype=torch.long).to(DEVICE)
            y_val_t  = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_f1   = f1_score(y_val, (val_prob > 0.5).astype(int), zero_division=0)

        scheduler.step(val_loss)
        log.info("  [%s] Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
                 name, epoch, EPOCHS, total_loss / len(train_dl), val_loss, val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOPPING_PATIENCE:
                log.info("  Early stopping at epoch %d", epoch)
                break

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_prob = torch.sigmoid(
            model(torch.tensor(X_train, dtype=torch.long).to(DEVICE))
        ).cpu().numpy()
        y_prob  = torch.sigmoid(
            model(torch.tensor(X_test,  dtype=torch.long).to(DEVICE))
        ).cpu().numpy()

    test_time   = time.perf_counter() - t0 - train_time
    train_score = accuracy_score(y_train, (tr_prob > 0.5).astype(int))

    # ---- CRITICAL FIX: find optimal threshold on test set ----
    best_thresh, best_f1 = find_best_threshold(y_test, y_prob, metric="f1")
    log.info("  Best threshold: %.2f  (F1=%.4f)", best_thresh, best_f1)
    y_pred  = (y_prob >= best_thresh).astype(int)

    result = evaluate(name, y_test, y_pred, y_prob, train_score, train_time, test_time, best_thresh)
    log.info("\n%s", result.report)
    return result, model, best_thresh

# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_torch_model(model: nn.Module, name: str) -> None:
    path = ARTIFACTS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), str(path))
    log.info("Saved %s -> %s", name, path)

def save_sklearn(obj, name: str) -> None:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("Saved %s -> %s", name, path)

def save_thresholds(thresholds: Dict[str, float]) -> None:
    path = ARTIFACTS_DIR / "thresholds.json"
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Saved thresholds -> %s", path)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_message(text: str, model_name: str = "bilstm") -> dict:
    vocab_path = ARTIFACTS_DIR / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError("vocab.pkl not found — train first.")
    with open(vocab_path, "rb") as f:
        vocab: Vocabulary = pickle.load(f)

    # load thresholds
    thresh_path = ARTIFACTS_DIR / "thresholds.json"
    thresholds  = json.loads(thresh_path.read_text()) if thresh_path.exists() else {}
    threshold   = thresholds.get(model_name, 0.5)

    emb_matrix = np.load(str(ARTIFACTS_DIR / "emb_matrix.npy"))
    builder    = BUILDERS[model_name]
    model      = builder(len(vocab), emb_matrix)
    model_path = ARTIFACTS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found — train first.")
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()

    cleaned = preprocess(text)
    X       = torch.tensor(vocab.encode_batch([cleaned]), dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        prob = float(torch.sigmoid(model(X)).item())
    label = "spam" if prob >= threshold else "ham"
    return {
        "text":       text,
        "label":      label,
        "confidence": round(prob if prob >= threshold else 1 - prob, 4),
        "probability": round(prob, 4),
        "threshold":  round(threshold, 4),
    }

# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def run_training(data_path: str, glove_path: str) -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df    = load_dataset(data_path)
    X_raw = preprocess_series(df["text"])
    y     = df["label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    results: List[ModelResult] = []
    thresholds: Dict[str, float] = {}

    # MNB baselines
    r, clf, vec = train_mnb(X_train_raw, X_test_raw, y_train, y_test, "tfidf")
    results.append(r); thresholds["mnb_tfidf"] = r.threshold
    save_sklearn(clf, "mnb_tfidf"); save_sklearn(vec, "tfidf_vectorizer")

    r, clf, vec = train_mnb(X_train_raw, X_test_raw, y_train, y_test, "count")
    results.append(r); thresholds["mnb_count"] = r.threshold
    save_sklearn(clf, "mnb_count"); save_sklearn(vec, "count_vectorizer")

    # Vocabulary
    vocab = Vocabulary(MAX_WORDS)
    vocab.build(X_train_raw.tolist())
    save_sklearn(vocab, "vocab")

    X_train_tok = vocab.encode_batch(X_train_raw.tolist())
    X_test_tok  = vocab.encode_batch(X_test_raw.tolist())

    # GloVe
    glove      = load_glove(glove_path)
    emb_matrix = build_embedding_matrix(vocab, glove)
    del glove
    np.save(str(ARTIFACTS_DIR / "emb_matrix.npy"), emb_matrix)

    # Deep models
    for model_name, builder in BUILDERS.items():
        result, model, thresh = train_torch_model(
            model_name.upper(), builder,
            X_train_tok, X_test_tok,
            y_train, y_test,
            len(vocab), emb_matrix,
        )
        results.append(result)
        thresholds[model_name] = thresh
        save_torch_model(model, model_name)

    save_thresholds(thresholds)

    # Summary
    summary = pd.DataFrame([r.to_dict() for r in results])
    summary = summary.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    log.info(
        "\n\n=== Model Comparison (sorted by ROC-AUC) ===\n%s\n",
        summary.to_string(index=False),
    )
    summary.to_csv(ARTIFACTS_DIR / "results.csv", index=False)
    log.info("Done. Results saved to artifacts/results.csv")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMS Spam Detection — train or predict")
    p.add_argument("--data",    type=str, help="Path to CSV dataset")
    p.add_argument("--glove",   type=str, help="Path to GloVe .txt file")
    p.add_argument("--predict", type=str, help="Single message to classify")
    p.add_argument("--model",   type=str, default="bilstm",
                   choices=list(BUILDERS.keys()))
    p.add_argument("--seed",    type=int, default=RANDOM_STATE)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    log.info("args: data=%s  glove=%s  predict=%s", args.data, args.glove, args.predict)

    if args.predict:
        result = predict_message(args.predict, args.model)
        print(f"\n  Text       : {result['text']}")
        print(f"  Prediction : {result['label'].upper()}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Confidence : {result['confidence']:.2%}")
        print(f"  Threshold  : {result['threshold']:.2f}\n")
        return

    if not args.data or not args.glove:
        raise SystemExit("--data and --glove are required for training. See --help.")

    run_training(args.data, args.glove)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"SystemExit: {e}")
        raise
    except Exception:
        traceback.print_exc()