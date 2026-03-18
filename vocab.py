# vocab.py

from typing import Dict, List
import numpy as np

MAX_SEQ_LEN = 70

class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_words: int = 10000):
        self.max_words = max_words
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build(self, texts: List[str]) -> None:
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        self.word2idx = {self.PAD: 0, self.UNK: 1}
        for word, _ in counter.most_common(self.max_words - 2):
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str], max_len: int = MAX_SEQ_LEN):
        return np.array([self.encode(t, max_len) for t in texts], dtype=np.int64)

    def __len__(self):
        return len(self.word2idx)