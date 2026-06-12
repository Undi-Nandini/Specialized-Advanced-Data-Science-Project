"""
watcher/monitor.py
Rolling-window performance monitor with structured event log.
Uses a ring-buffer approach instead of defaultdict; different from prior version.
"""

from __future__ import annotations

import json
import time
import math
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional

log = logging.getLogger("Watcher")

_LABELS = ["Negative", "Neutral", "Positive"]


@dataclass
class Event:
    ts:         float     # unix timestamp
    label:      str
    confidence: float
    latency_ms: float
    is_error:   bool = False


@dataclass
class WindowStats:
    """Statistics computed over a rolling window of events."""
    window:      int
    n_events:    int
    n_errors:    int
    error_rate:  float
    label_dist:  Dict[str, float]
    avg_conf:    Dict[str, float]
    lat_mean:    float
    lat_p50:     float
    lat_p95:     float
    lat_p99:     float
    drift_score: float
    drift_alert: bool


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Symmetric KL divergence between two label distributions."""
    eps = 1e-9
    kl = 0.0
    for lbl in _LABELS:
        pi = p.get(lbl, 0) + eps
        qi = q.get(lbl, 1/3) + eps
        kl += pi * math.log(pi / qi)
    return round(kl, 5)


# ── Monitor class ─────────────────────────────────────────────────────────────

class RollingMonitor:
    """
    Maintains a ring buffer of N recent events and computes live stats.
    Can emit JSON snapshots to disk.
    """

    BASELINE = {lbl: 1/3 for lbl in _LABELS}   # uniform prior
    DRIFT_THRESHOLD = 0.12

    def __init__(self, window: int = 600, log_dir: str = "monitoring/logs"):
        self._window:  int              = window
        self._buf:     Deque[Event]     = deque(maxlen=window)
        self._all_n:   int              = 0
        self._all_err: int              = 0
        self._log_dir: Path             = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._start:   float            = time.time()

    # ── ingest ────────────────────────────────────────────────────────────────

    def record(self, label: str, confidence: float,
               latency_ms: float, is_error: bool = False):
        evt = Event(ts=time.time(), label=label, confidence=confidence,
                    latency_ms=latency_ms, is_error=is_error)
        self._buf.append(evt)
        self._all_n += 1
        if is_error:
            self._all_err += 1

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self) -> WindowStats:
        events = list(self._buf)
        n      = len(events)
        errors = sum(e.is_error for e in events)

        # Label distribution
        counts = {lbl: sum(1 for e in events if e.label == lbl and not e.is_error)
                  for lbl in _LABELS}
        total_ok = max(sum(counts.values()), 1)
        dist = {lbl: counts[lbl] / total_ok for lbl in _LABELS}

        # Confidence by label
        avg_conf: Dict[str, float] = {}
        for lbl in _LABELS:
            vals = [e.confidence for e in events if e.label == lbl and not e.is_error]
            avg_conf[lbl] = round(sum(vals) / len(vals), 4) if vals else 0.0

        # Latency
        lats = [e.latency_ms for e in events if not e.is_error]
        drift = _kl_divergence(dist, self.BASELINE)

        return WindowStats(
            window      = self._window,
            n_events    = n,
            n_errors    = errors,
            error_rate  = round(errors / max(n, 1), 4),
            label_dist  = {k: round(v, 4) for k, v in dist.items()},
            avg_conf    = avg_conf,
            lat_mean    = round(sum(lats) / max(len(lats), 1), 2),
            lat_p50     = round(_percentile(lats, 50), 2),
            lat_p95     = round(_percentile(lats, 95), 2),
            lat_p99     = round(_percentile(lats, 99), 2),
            drift_score = drift,
            drift_alert = drift > self.DRIFT_THRESHOLD,
        )

    # ── report ────────────────────────────────────────────────────────────────

    def report(self) -> dict:
        s = self.compute()
        return {
            "uptime_s":       round(time.time() - self._start, 1),
            "total_events":   self._all_n,
            "total_errors":   self._all_err,
            **asdict(s),
        }

    def print_report(self):
        r = self.report()
        s = self.compute()
        print(f"\n{'╔' + '═'*52 + '╗'}")
        print(f"║  {'📊  ROLLING MONITOR DASHBOARD':^50}║")
        print(f"{'╠' + '═'*52 + '╣'}")
        print(f"║  {'Events in window':<28} {s.n_events:>22} ║")
        print(f"║  {'Total events (all time)':<28} {r['total_events']:>22} ║")
        print(f"║  {'Error rate':<28} {s.error_rate*100:>21.2f}%║")
        print(f"║  {'Avg latency':<28} {s.lat_mean:>20.1f}ms║")
        print(f"║  {'p95 latency':<28} {s.lat_p95:>20.1f}ms║")
        print(f"{'╠' + '═'*52 + '╣'}")
        print(f"║  {'Label Distribution':^50}  ║")
        for lbl, frac in s.label_dist.items():
            bar = '█' * int(frac * 28)
            print(f"║  {lbl:<12} {bar:<28} {frac*100:>5.1f}%║")
        print(f"{'╠' + '═'*52 + '╣'}")
        alert_str = "⚠️  DRIFT DETECTED" if s.drift_alert else "✅ No significant drift"
        print(f"║  {'Drift (KL score)':<28} {s.drift_score:>22.5f}║")
        print(f"║  {alert_str:^52}║")
        print(f"{'╚' + '═'*52 + '╝'}\n")

    def snapshot(self) -> str:
        path = self._log_dir / f"snap_{int(time.time())}.json"
        with open(path, "w") as fh:
            json.dump(self.report(), fh, indent=2, default=float)
        log.info("Snapshot → %s", path)
        return str(path)


if __name__ == "__main__":
    import random
    random.seed(42)
    mon = RollingMonitor(window=400)

    for _ in range(300):
        lbl  = random.choice(_LABELS)
        conf = random.uniform(0.55, 0.99)
        lat  = random.uniform(8, 95)
        err  = random.random() < 0.01
        mon.record(lbl, conf, lat, err)

    mon.print_report()
    mon.snapshot()
