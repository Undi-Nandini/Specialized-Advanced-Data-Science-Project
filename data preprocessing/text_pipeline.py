"""
pipeline/text_pipeline.py
Text cleaning, vocabulary management, and sequence encoding for NLP tasks.
Uses a config-driven design with dataclasses.
"""

import re
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import List, Tuple, Optional

logging.basicConfig(
    format="[%(levelname)s] %(name)s :: %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TextPipeline")


@dataclass
class PipelineConfig:
    """Central configuration for the text processing pipeline."""
    max_vocab:      int   = 8000
    max_seq_len:    int   = 90
    min_word_freq:  int   = 1
    lowercase:      bool  = True
    strip_urls:     bool  = True
    strip_html:     bool  = True
    drop_stopwords: bool  = True
    test_ratio:     float = 0.20
    val_ratio:      float = 0.10
    random_seed:    int   = 42

    PAD: str = "<PAD>"   # index 0
    UNK: str = "<UNK>"   # index 1
    LABEL_MAP: dict = field(default_factory=lambda: {0: "Negative", 1: "Neutral", 2: "Positive"})

    def to_json(self, path: str):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as fh:
            return cls(**json.load(fh))


# ── Common English stopwords ──────────────────────────────────────────────────
_STOP = frozenset([
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","this","that","with","was","are","be","been","by","from","as","so",
    "if","its","my","we","i","he","she","they","you","me","him","her","us",
    "all","more","also","just","than","then","into","out","about","up","no",
    "have","has","had","do","did","would","could","should","will","can","may"
])


class TextCleaner:
    """
    Stateless utility: apply a sequence of regex-based transformations to text.
    Each step can be toggled via PipelineConfig.
    """

    _URL_RE   = re.compile(r"https?://\S+|www\.\S+")
    _HTML_RE  = re.compile(r"<[^>]+>")
    _ALPHA_RE = re.compile(r"[^a-z0-9\s]")
    _SPACE_RE = re.compile(r"\s{2,}")

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def transform(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            return ""
        s = raw
        if self.cfg.lowercase:   s = s.lower()
        if self.cfg.strip_urls:  s = self._URL_RE.sub(" ", s)
        if self.cfg.strip_html:  s = self._HTML_RE.sub(" ", s)
        s = self._ALPHA_RE.sub(" ", s)
        s = self._SPACE_RE.sub(" ", s).strip()
        if self.cfg.drop_stopwords:
            s = " ".join(w for w in s.split() if w not in _STOP)
        return s

    def fit_transform_df(self, df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
        out = df.copy()
        out["clean_text"] = out[col].apply(self.transform)
        out["word_count"]  = out["clean_text"].str.split().str.len()
        log.info("Cleaned %d rows | avg words: %.1f", len(out), out["word_count"].mean())
        return out


class Vocabulary:
    """
    Builds and manages word-to-index mappings. Serialisable to JSON.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.w2i: dict[str, int] = {}
        self.i2w: dict[int, str] = {}
        self._built = False

    def build(self, corpus: List[str]):
        freq = Counter(tok for doc in corpus for tok in doc.split())
        # Remove rare words
        filtered = {w: c for w, c in freq.items() if c >= self.cfg.min_word_freq}
        # Sort by freq desc, keep top (max_vocab - 2) to leave room for PAD/UNK
        top_words = sorted(filtered, key=filtered.__getitem__, reverse=True)
        top_words = top_words[: self.cfg.max_vocab - 2]

        self.w2i = {self.cfg.PAD: 0, self.cfg.UNK: 1}
        for idx, word in enumerate(top_words, start=2):
            self.w2i[word] = idx
        self.i2w = {v: k for k, v in self.w2i.items()}
        self._built = True
        log.info("Vocabulary: %d tokens (freq_cutoff=%d)", len(self.w2i), self.cfg.min_word_freq)

    def encode(self, texts: List[str]) -> np.ndarray:
        assert self._built, "Call build() before encode()."
        L = self.cfg.max_seq_len
        out = np.zeros((len(texts), L), dtype=np.int32)
        for i, text in enumerate(texts):
            tokens = text.split()[:L]
            for j, tok in enumerate(tokens):
                out[i, j] = self.w2i.get(tok, 1)   # 1 = UNK
        return out

    @property
    def size(self) -> int:
        return len(self.w2i)

    def save(self, path: str):
        payload = {"w2i": self.w2i, "cfg": asdict(self.cfg)}
        with open(path, "w") as fh:
            json.dump(payload, fh)
        log.info("Vocab saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as fh:
            data = json.load(fh)
        cfg = PipelineConfig(**data["cfg"])
        v = cls(cfg)
        v.w2i = {k: int(vi) for k, vi in data["w2i"].items()}
        v.i2w = {int(vi): k for k, vi in data["w2i"].items()}
        v._built = True
        return v


def load_split_data(
    csv_path: str,
    cfg: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, shuffle, and split into (train, val, test).
    Returns DataFrames in that order.
    """
    df = (
        pd.read_csv(csv_path)
          .dropna(subset=["text", "label"])
          .sample(frac=1, random_state=cfg.random_seed)
          .reset_index(drop=True)
    )
    n = len(df)
    n_test = int(n * cfg.test_ratio)
    n_val  = int(n * cfg.val_ratio)
    test  = df.iloc[:n_test]
    val   = df.iloc[n_test: n_test + n_val]
    train = df.iloc[n_test + n_val:]
    log.info("Split — train:%d  val:%d  test:%d", len(train), len(val), len(test))
    return train, val, test
