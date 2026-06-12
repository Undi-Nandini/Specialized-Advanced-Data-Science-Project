"""
network/sentiment_net.py
Bidirectional GRU-based sentiment classifier (NumPy).
Different from a standard LSTM: uses GRU cells which have fewer gates
(reset + update instead of input/forget/output).
Uses a functional-style forward graph with layer objects stored in a registry.
"""

import numpy as np
import pickle
import logging
from typing import Dict, Any, Tuple

log = logging.getLogger("SentimentNet")


# ══════════════════════════════════════════════════════════════════════════════
# Activations
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=-1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# Individual Layer Classes
# ══════════════════════════════════════════════════════════════════════════════

class EmbedLayer:
    """Lookup-table embedding with uniform initialisation."""

    def __init__(self, vocab_size: int, dim: int):
        lim = np.sqrt(6 / (vocab_size + dim))
        self.W = np.random.uniform(-lim, lim, (vocab_size, dim)).astype(np.float32)
        self.vocab_size = vocab_size
        self.dim = dim

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """idx: (B, T) → (B, T, D)"""
        return self.W[idx]


class GRUCell:
    """
    Minimal GRU cell: two gates (reset r, update z) and a candidate state.
    Fewer parameters than LSTM; often converges faster on small datasets.
    """

    def __init__(self, in_dim: int, h_dim: int):
        self.h_dim = h_dim
        d = in_dim + h_dim
        scale = np.sqrt(2.0 / d)
        # Stacked weight for reset, update, candidate
        self.Wz = np.random.randn(d, h_dim).astype(np.float32) * scale
        self.Wr = np.random.randn(d, h_dim).astype(np.float32) * scale
        self.Wh = np.random.randn(d, h_dim).astype(np.float32) * scale
        self.bz = np.zeros(h_dim, dtype=np.float32)
        self.br = np.zeros(h_dim, dtype=np.float32)
        self.bh = np.zeros(h_dim, dtype=np.float32)

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        xh = np.concatenate([x, h], axis=-1)
        z  = _sigmoid(xh @ self.Wz + self.bz)
        r  = _sigmoid(xh @ self.Wr + self.br)
        xr = np.concatenate([x, r * h], axis=-1)
        h_hat = _tanh(xr @ self.Wh + self.bh)
        return (1 - z) * h + z * h_hat

    def forward(self, seq: np.ndarray, return_seq: bool = False) -> np.ndarray:
        """seq: (B, T, in_dim)"""
        B, T, _ = seq.shape
        h = np.zeros((B, self.h_dim), dtype=np.float32)
        hs = []
        for t in range(T):
            h = self.step(seq[:, t], h)
            hs.append(h)
        return np.stack(hs, axis=1) if return_seq else h


class LinearLayer:
    """Standard affine layer y = xW + b."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self._act = {"relu": _relu, "softmax": _softmax, "linear": lambda x: x}[activation]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._act(x @ self.W + self.b)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model
# ══════════════════════════════════════════════════════════════════════════════

class ReviewSentimentNet:
    """
    Architecture (GRU-based, distinct from LSTM):
      Embed(vocab, 64) → BiGRU(48) → GRU(24) → Linear(20, relu) → Linear(3, softmax)

    Parameters are stored in a named registry for easy serialisation.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 gru1_dim: int = 48, gru2_dim: int = 24,
                 hidden_dim: int = 20, num_classes: int = 3):

        self.hparams: Dict[str, Any] = dict(
            vocab_size=vocab_size, embed_dim=embed_dim,
            gru1_dim=gru1_dim, gru2_dim=gru2_dim,
            hidden_dim=hidden_dim, num_classes=num_classes,
        )

        # Layer registry
        self.embed   = EmbedLayer(vocab_size, embed_dim)
        self.gru1_fw = GRUCell(embed_dim, gru1_dim)
        self.gru1_bw = GRUCell(embed_dim, gru1_dim)
        self.gru2    = GRUCell(gru1_dim * 2, gru2_dim)
        self.fc1     = LinearLayer(gru2_dim, hidden_dim, "relu")
        self.fc2     = LinearLayer(hidden_dim, num_classes, "softmax")

    # ── forward ────────────────────────────────────────────────────────────────
    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """tokens: (B, T) → probs: (B, C)"""
        x  = self.embed.forward(tokens)                       # (B,T,E)
        fw = self.gru1_fw.forward(x, return_seq=True)         # (B,T,H1)
        bw = self.gru1_bw.forward(x[:, ::-1], return_seq=True)[:, ::-1]  # reversed
        bi = np.concatenate([fw, bw], axis=-1)                # (B,T,2H1)
        h2 = self.gru2.forward(bi)                            # (B,H2)
        h3 = self.fc1.forward(h2)                             # (B,hidden)
        return self.fc2.forward(h3)                           # (B,C)

    def predict(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.forward(tokens)
        return np.argmax(probs, axis=-1), probs

    # ── param count ───────────────────────────────────────────────────────────
    def param_count(self) -> int:
        h = self.hparams
        e  = h["vocab_size"] * h["embed_dim"]
        g1 = 3 * (h["embed_dim"] + h["gru1_dim"]) * h["gru1_dim"] * 2   # fwd+bwd
        g2 = 3 * (h["gru1_dim"]*2 + h["gru2_dim"]) * h["gru2_dim"]
        f1 = h["gru2_dim"] * h["hidden_dim"] + h["hidden_dim"]
        f2 = h["hidden_dim"] * h["num_classes"] + h["num_classes"]
        return e + g1 + g2 + f1 + f2

    def summary(self):
        h = self.hparams
        border = "─" * 52
        print(f"\n┌{border}┐")
        print(f"│  {'ReviewSentimentNet (BiGRU)':^50}│")
        print(f"├{border}┤")
        print(f"│  {'Layer':<24} {'Shape / Units':>26}│")
        print(f"├{border}┤")
        rows = [
            ("EmbedLayer",      f"{h['vocab_size']} × {h['embed_dim']}"),
            ("BiGRU Layer-1 Fw",f"{h['gru1_dim']} units"),
            ("BiGRU Layer-1 Bw",f"{h['gru1_dim']} units"),
            ("GRU  Layer-2",    f"{h['gru2_dim']} units"),
            ("Linear + ReLU",   f"{h['hidden_dim']} units"),
            ("Linear + Softmax",f"{h['num_classes']} classes"),
        ]
        for name, shape in rows:
            print(f"│  {name:<24} {shape:>26}│")
        print(f"├{border}┤")
        print(f"│  {'Total Params:':<24} {self.param_count():>26,}│")
        print(f"└{border}┘\n")

    # ── serialisation ─────────────────────────────────────────────────────────
    def save(self, path: str):
        snapshot = {
            "hparams": self.hparams,
            "embed":   self.embed.W,
            "gru1_fw": (self.gru1_fw.Wz, self.gru1_fw.Wr, self.gru1_fw.Wh,
                        self.gru1_fw.bz, self.gru1_fw.br, self.gru1_fw.bh),
            "gru1_bw": (self.gru1_bw.Wz, self.gru1_bw.Wr, self.gru1_bw.Wh,
                        self.gru1_bw.bz, self.gru1_bw.br, self.gru1_bw.bh),
            "gru2":    (self.gru2.Wz, self.gru2.Wr, self.gru2.Wh,
                        self.gru2.bz, self.gru2.br, self.gru2.bh),
            "fc1":     (self.fc1.W, self.fc1.b),
            "fc2":     (self.fc2.W, self.fc2.b),
        }
        with open(path, "wb") as fh:
            pickle.dump(snapshot, fh)
        log.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "ReviewSentimentNet":
        with open(path, "rb") as fh:
            snap = pickle.load(fh)
        m = cls(**snap["hparams"])
        m.embed.W = snap["embed"]
        for layer, key in [(m.gru1_fw,"gru1_fw"),(m.gru1_bw,"gru1_bw"),(m.gru2,"gru2")]:
            layer.Wz,layer.Wr,layer.Wh,layer.bz,layer.br,layer.bh = snap[key]
        m.fc1.W, m.fc1.b = snap["fc1"]
        m.fc2.W, m.fc2.b = snap["fc2"]
        log.info("Model loaded ← %s", path)
        return m
