"""
server/engine.py
Inference engine exposing a context-manager interface for safe resource handling.
Wraps the model + vocabulary + cleaner into one self-contained unit.
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.text_pipeline import PipelineConfig, TextCleaner, Vocabulary
from network.sentiment_net import ReviewSentimentNet

log = logging.getLogger("InferenceEngine")

LABELS  = {0: "Negative", 1: "Neutral", 2: "Positive"}
ICONS   = {0: "😠", 1: "😐", 2: "😊"}


@dataclass
class Prediction:
    """Immutable result object returned by the engine."""
    text:       str
    clean_text: str
    label:      str
    label_id:   int
    icon:       str
    confidence: float
    scores:     dict
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "clean_text":  self.clean_text,
            "label":       self.label,
            "label_id":    self.label_id,
            "icon":        self.icon,
            "confidence":  round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency_ms":  round(self.latency_ms, 2),
        }


class InferenceEngine:
    """
    Loads model + vocab + config and provides single/batch prediction.
    Designed to be used as a context manager:

        with InferenceEngine.from_dir("src/network/saved") as engine:
            result = engine.run("This product is amazing!")
    """

    def __init__(self, model: ReviewSentimentNet, vocab: Vocabulary,
                 cleaner: TextCleaner):
        self._model   = model
        self._vocab   = vocab
        self._cleaner = cleaner
        self._calls   = 0
        self._lat_sum = 0.0
        log.info("InferenceEngine ready  (vocab=%d tokens)", vocab.size)

    # ── context manager ────────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        log.info("Engine closed after %d calls (avg %.1fms)", self._calls,
                 self._lat_sum / max(self._calls, 1))

    # ── factory ────────────────────────────────────────────────────────────────
    @classmethod
    def from_dir(cls, model_dir: str) -> "InferenceEngine":
        cfg     = PipelineConfig.from_json(f"{model_dir}/config.json")
        vocab   = Vocabulary.load(f"{model_dir}/vocab.json")
        cleaner = TextCleaner(cfg)
        model   = ReviewSentimentNet.load(f"{model_dir}/checkpoint.pkl")
        return cls(model, vocab, cleaner)

    # ── single prediction ──────────────────────────────────────────────────────
    def run(self, text: str) -> Prediction:
        t0 = time.perf_counter()
        clean = self._cleaner.transform(text)
        seq   = self._vocab.encode([clean])
        _, probs = self._model.predict(seq)
        probs = probs[0]
        lid   = int(np.argmax(probs))
        ms    = (time.perf_counter() - t0) * 1000

        self._calls   += 1
        self._lat_sum += ms

        return Prediction(
            text       = text,
            clean_text = clean,
            label      = LABELS[lid],
            label_id   = lid,
            icon       = ICONS[lid],
            confidence = float(probs[lid]),
            scores     = {LABELS[i]: float(probs[i]) for i in range(3)},
            latency_ms = ms,
        )

    # ── batch prediction ───────────────────────────────────────────────────────
    def run_batch(self, texts: List[str]) -> List[Prediction]:
        t0     = time.perf_counter()
        cleans = [self._cleaner.transform(t) for t in texts]
        seqs   = self._vocab.encode(cleans)
        _, probs = self._model.predict(seqs)
        total_ms = (time.perf_counter() - t0) * 1000

        results = []
        for i, (text, clean) in enumerate(zip(texts, cleans)):
            p   = probs[i]
            lid = int(np.argmax(p))
            results.append(Prediction(
                text       = text,
                clean_text = clean,
                label      = LABELS[lid],
                label_id   = lid,
                icon       = ICONS[lid],
                confidence = float(p[lid]),
                scores     = {LABELS[j]: float(p[j]) for j in range(3)},
                latency_ms = total_ms / len(texts),
            ))
        self._calls += len(texts)
        log.info("Batch of %d → %.1fms total", len(texts), total_ms)
        return results

    # ── stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "total_calls":    self._calls,
            "avg_latency_ms": round(self._lat_sum / max(self._calls, 1), 2),
        }


@contextmanager
def open_engine(model_dir: str):
    """Convenience context manager."""
    engine = InferenceEngine.from_dir(model_dir)
    try:
        yield engine
    finally:
        engine.__exit__(None, None, None)


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[2])

    samples = [
        "Absolutely outstanding quality — would buy again in a heartbeat.",
        "Fairly mediocre. It works but nothing to write home about.",
        "Appalling experience. Returned it on the same day.",
        "Decent product for the price. No major complaints.",
        "Brilliant from start to finish. Truly exceeded every expectation.",
    ]

    with open_engine("src/network/saved") as eng:
        print(f"\n{'═'*62}")
        print("  SAMPLE PREDICTIONS")
        print(f"{'═'*62}")
        for s in samples:
            r = eng.run(s)
            print(f"\n  Input   : {r.text}")
            print(f"  Result  : {r.icon} {r.label} ({r.confidence*100:.1f}%)")
            print(f"  Scores  : Neg={r.scores['Negative']:.3f} | "
                  f"Neu={r.scores['Neutral']:.3f} | "
                  f"Pos={r.scores['Positive']:.3f}")
            print(f"  Latency : {r.latency_ms:.2f}ms")
        print(f"\n  Engine stats: {eng.stats()}")
        print(f"{'═'*62}\n")
