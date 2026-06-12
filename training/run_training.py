"""
trainer/run_training.py
Training loop with a callback/hook system for extensibility.
Supports early stopping, LR decay, and JSON-serialised run logs.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.text_pipeline import PipelineConfig, TextCleaner, Vocabulary, load_split_data
from network.sentiment_net import ReviewSentimentNet

log = logging.getLogger("Trainer")
logging.basicConfig(format="[%(levelname)s] %(name)s :: %(message)s", level=logging.INFO)

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ── Loss & Metrics ────────────────────────────────────────────────────────────

def xent_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(p + 1e-9)))

def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=-1) == y))

def batch_forward(model, X: np.ndarray, batch_size: int = 64):
    chunks = [model.forward(X[i:i+batch_size]) for i in range(0, len(X), batch_size)]
    return np.vstack(chunks)

def per_class_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=-1)
    out = {}
    for c, name in LABELS.items():
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2*p*r / (p+r+1e-9)
        out[name] = {"precision": round(p,4), "recall": round(r,4),
                     "f1": round(f1,4), "support": int(np.sum(y == c))}
    return out


# ── Callback Base ─────────────────────────────────────────────────────────────

class Callback:
    """Override any hook to add custom behaviour during training."""
    def on_epoch_start(self, epoch: int, state: dict): pass
    def on_epoch_end(self,   epoch: int, state: dict): pass
    def on_train_end(self,              state: dict): pass


class EarlyStopper(Callback):
    """Stop training when val_acc hasn't improved for `patience` epochs."""
    def __init__(self, patience: int = 5):
        self.patience  = patience
        self._best     = 0.0
        self._wait     = 0
        self.triggered = False

    def on_epoch_end(self, epoch: int, state: dict):
        val_acc = state.get("val_acc", 0)
        if val_acc > self._best + 1e-4:
            self._best = val_acc
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                log.info("EarlyStopper triggered at epoch %d (patience=%d)", epoch, self.patience)
                self.triggered = True
                state["stop"] = True


class CheckpointSaver(Callback):
    """Save model whenever val_acc improves."""
    def __init__(self, model: "ReviewSentimentNet", save_path: str):
        self._model     = model
        self._save_path = save_path
        self._best      = 0.0

    def on_epoch_end(self, epoch: int, state: dict):
        va = state.get("val_acc", 0)
        if va > self._best:
            self._best = va
            self._model.save(self._save_path)
            log.info("  ✔ Checkpoint saved (val_acc=%.4f)", va)


class LRDecayCallback(Callback):
    """Reduce learning rate by `factor` every `step` epochs."""
    def __init__(self, step: int = 5, factor: float = 0.7):
        self.step   = step
        self.factor = factor

    def on_epoch_end(self, epoch: int, state: dict):
        if epoch % self.step == 0:
            state["lr"] *= self.factor
            log.info("  LR decayed → %.5f", state["lr"])


# ── Training Step ─────────────────────────────────────────────────────────────

def _train_step(model: ReviewSentimentNet, X: np.ndarray, y: np.ndarray,
                lr: float, batch_size: int) -> tuple:
    """One epoch: shuffle → mini-batches → output-layer SGD update."""
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    total_loss, total_correct, nb = 0.0, 0, 0
    for s in range(0, len(X), batch_size):
        xb, yb = X[s:s+batch_size], y[s:s+batch_size]
        B = len(xb)

        # Forward
        probs = model.forward(xb)
        total_loss    += xent_loss(probs, yb)
        total_correct += int(np.sum(np.argmax(probs, axis=-1) == yb))
        nb += 1

        # Partial backprop: update fc2 (output layer) weights only
        dz = probs.copy()
        dz[np.arange(B), yb] -= 1
        dz /= B

        # Need fc1 output for gradient
        emb   = model.embed.forward(xb)
        fw    = model.gru1_fw.forward(emb, return_seq=True)
        bw    = model.gru1_bw.forward(emb[:,::-1], return_seq=True)[:, ::-1]
        bi    = np.concatenate([fw, bw], axis=-1)
        h2    = model.gru2.forward(bi)
        h3    = model.fc1.forward(h2)

        model.fc2.W -= lr * (h3.T @ dz)
        model.fc2.b -= lr * dz.sum(axis=0)

    return total_loss / nb, total_correct / len(X)


# ── Main Train Function ───────────────────────────────────────────────────────

def train(
    data_csv:   str  = "data/reviews.csv",
    output_dir: str  = "src/network/saved",
    epochs:     int  = 18,
    lr:         float= 0.06,
    batch_size: int  = 32,
    callbacks:  Optional[List[Callback]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    callbacks = callbacks or []

    cfg = PipelineConfig(max_vocab=6000, max_seq_len=90, random_seed=7)
    cfg.to_json(f"{output_dir}/config.json")

    # ── Data ──────────────────────────────────────────────────────────────────
    cleaner = TextCleaner(cfg)
    train_df, val_df, test_df = load_split_data(data_csv, cfg)
    train_df = cleaner.fit_transform_df(train_df, col="text")
    val_df   = cleaner.fit_transform_df(val_df,   col="text")
    test_df  = cleaner.fit_transform_df(test_df,  col="text")

    vocab = Vocabulary(cfg)
    vocab.build(train_df["clean_text"].tolist())
    vocab.save(f"{output_dir}/vocab.json")

    X_tr  = vocab.encode(train_df["clean_text"].tolist())
    X_va  = vocab.encode(val_df["clean_text"].tolist())
    X_te  = vocab.encode(test_df["clean_text"].tolist())
    y_tr  = train_df["label"].values.astype(np.int32)
    y_va  = val_df["label"].values.astype(np.int32)
    y_te  = test_df["label"].values.astype(np.int32)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ReviewSentimentNet(vocab_size=vocab.size)
    model.summary()

    # ── Loop ──────────────────────────────────────────────────────────────────
    history = {"tr_loss":[], "tr_acc":[], "va_loss":[], "va_acc":[]}
    state   = {"lr": lr, "stop": False}
    best_va = 0.0
    t0      = time.time()

    log.info("Training: epochs=%d  lr=%.4f  batch=%d", epochs, lr, batch_size)

    for cb in callbacks:
        cb.on_epoch_start(0, state)

    for ep in range(1, epochs + 1):
        for cb in callbacks:
            cb.on_epoch_start(ep, state)

        tl, ta = _train_step(model, X_tr, y_tr, state["lr"], batch_size)
        vp     = batch_forward(model, X_va)
        vl, va = xent_loss(vp, y_va), accuracy(vp, y_va)

        history["tr_loss"].append(round(tl, 5))
        history["tr_acc"].append(round(ta, 5))
        history["va_loss"].append(round(vl, 5))
        history["va_acc"].append(round(va, 5))
        state["val_acc"] = va

        if ep % 3 == 0 or ep == 1:
            log.info("ep %3d | tr_loss=%.4f acc=%.4f | va_loss=%.4f acc=%.4f",
                     ep, tl, ta, vl, va)

        for cb in callbacks:
            cb.on_epoch_end(ep, state)

        if state.get("stop"):
            log.info("Training stopped early at epoch %d.", ep)
            break

    elapsed = time.time() - t0

    # ── Evaluate best checkpoint ───────────────────────────────────────────────
    best_model = ReviewSentimentNet.load(f"{output_dir}/checkpoint.pkl")
    tp  = batch_forward(best_model, X_te)
    tst_acc = accuracy(tp, y_te)
    cls_met = per_class_metrics(tp, y_te)

    results = {
        "history":       history,
        "best_val_acc":  float(max(history["va_acc"])),
        "test_accuracy": round(tst_acc, 4),
        "class_metrics": cls_met,
        "runtime_secs":  round(elapsed, 2),
        "run_config":    {"epochs":epochs,"lr":lr,"batch":batch_size},
    }
    with open(f"{output_dir}/run_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    for cb in callbacks:
        cb.on_train_end(results)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'━'*52}")
    print(f"  TRAINING COMPLETE")
    print(f"{'━'*52}")
    print(f"  Best Val Accuracy  : {max(history['va_acc'])*100:.2f}%")
    print(f"  Test Accuracy      : {tst_acc*100:.2f}%")
    print(f"  Runtime            : {elapsed:.1f}s")
    print(f"\n  Per-Class Results:")
    for cls, m in cls_met.items():
        print(f"    {cls:12s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")
    print(f"{'━'*52}\n")
    return results


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    saver    = CheckpointSaver(None, "src/network/saved/checkpoint.pkl")
    stopper  = EarlyStopper(patience=6)
    lr_decay = LRDecayCallback(step=5, factor=0.75)

    # Patch saver to get model reference after build
    class PatchedSaver(CheckpointSaver):
        def on_epoch_end(self, ep, state):
            if "model" in state:
                self._model = state["model"]
            super().on_epoch_end(ep, state)

    results = train(
        data_csv   = "data/reviews.csv",
        output_dir = "src/network/saved",
        epochs     = 18,
        lr         = 0.06,
        callbacks  = [stopper, lr_decay],
    )
