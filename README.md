#  SMS Spam Detection using MNB, LSTM, GRU & BiLSTM

This project focuses on detecting spam messages in SMS texts using various machine learning and deep learning models — namely, **Multinomial Naive Bayes (MNB)**, **LSTM**, **GRU**, and **Bidirectional LSTM (BiLSTM)**. It demonstrates a comparative approach to evaluate performance and accuracy across traditional and neural network-based models.

---

##  Problem Statement

With the ever-growing number of unsolicited spam messages, there’s a need for an intelligent system that can differentiate between spam and ham (legitimate) SMS messages. This project aims to build such a system using different models and evaluate their performance on key metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

##  Tech Stack

- **Languages:** Python 3  
- **Libraries:** 
  - `scikit-learn`
  - `NLTK`
  - `TensorFlow` / `Keras`
  - `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- **Notebook:** Jupyter (`.ipynb`)

---

##  Models Implemented

1. **Multinomial Naive Bayes (MNB):**
   - Classical ML model using TF-IDF vectorization.
   - Fast and effective for simple NLP tasks.

2. **LSTM (Long Short-Term Memory):**
   - Captures long-term dependencies in text sequences.

3. **GRU (Gated Recurrent Unit):**
   - Similar to LSTM but more lightweight.

4. **Bidirectional LSTM (BiLSTM):**
   - Reads sequences both forwards and backwards for better context.

---

##  Dataset

- **Source:** [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Format:** CSV with two columns: `label` (spam/ham) and `message`
- **Preprocessing:** Lowercasing, stopword removal, stemming, tokenization, padding

---

##  Performance Highlights

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Naive Bayes   | ~98.5%   | High      | High   | High     |
| LSTM          | ~98%     | High      | High   | High     |
| GRU           | ~98.3%   | High      | High   | High     |
| BiLSTM        | **~98.5%** | High     | High   | High     |

> 🔧 Note: Metrics may vary slightly based on random seed and dataset split.

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detector.git
   cd sms-spam-detector
