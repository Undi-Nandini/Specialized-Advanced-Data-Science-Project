"""
server/api.py
FastAPI application – endpoint naming and middleware intentionally
different from the prior project version.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

log = logging.getLogger("API")

# ── Request / Response Schemas ────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    review: str = Field(..., min_length=2, max_length=5000,
                        example="Outstanding quality — highly recommend!")

    @validator("review")
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("review must not be blank")
        return v


class AnalyseResponse(BaseModel):
    review:     str
    sentiment:  str
    score_id:   int
    icon:       str
    confidence: float
    breakdown:  dict
    ms:         float


class BulkRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, max_items=200)


class StatusResponse(BaseModel):
    status:        str
    model_ready:   bool
    uptime_s:      float
    build:         str = "1.0.0"


class StatsResponse(BaseModel):
    calls:         int
    avg_ms:        float
    uptime_s:      float
    req_total:     int
    error_total:   int


# ── App & Middleware ───────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Review Sentiment API",
    description = "GRU-based NLP sentiment classifier — Negative / Neutral / Positive",
    version     = "1.0.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
    allow_credentials = False,
)

# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0  = time.time()
    res = await call_next(request)
    res.headers["X-Response-Time-Ms"] = str(round((time.time() - t0)*1000, 2))
    return res

# ── App State ─────────────────────────────────────────────────────────────────

_boot_time    = time.time()
_engine       = None
_req_count    = 0
_err_count    = 0


@app.on_event("startup")
async def startup_event():
    global _engine
    try:
        from server.engine import InferenceEngine
        _engine = InferenceEngine.from_dir("src/network/saved")
        log.info("Model loaded at startup.")
    except Exception as exc:
        log.error("Startup failed: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Review Sentiment API",
        "version": "1.0.0",
        "routes": ["/analyse", "/analyse/bulk", "/status", "/stats", "/api/docs"],
    }


@app.post("/analyse", response_model=AnalyseResponse, tags=["Sentiment"])
async def analyse(req: AnalyseRequest):
    """Analyse sentiment of a single review."""
    global _req_count, _err_count
    _req_count += 1
    if _engine is None:
        _err_count += 1
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model not available. Try again shortly.")
    try:
        p = _engine.run(req.review)
        return AnalyseResponse(
            review     = p.text,
            sentiment  = p.label,
            score_id   = p.label_id,
            icon       = p.icon,
            confidence = p.confidence,
            breakdown  = p.scores,
            ms         = p.latency_ms,
        )
    except Exception as exc:
        _err_count += 1
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyse/bulk", response_model=List[AnalyseResponse], tags=["Sentiment"])
async def analyse_bulk(req: BulkRequest):
    """Analyse up to 200 reviews in one request."""
    global _req_count, _err_count
    _req_count += 1
    if _engine is None:
        _err_count += 1
        raise HTTPException(status_code=503, detail="Model not available.")
    try:
        results = _engine.run_batch(req.reviews)
        return [AnalyseResponse(review=p.text, sentiment=p.label, score_id=p.label_id,
                                icon=p.icon, confidence=p.confidence, breakdown=p.scores,
                                ms=p.latency_ms) for p in results]
    except Exception as exc:
        _err_count += 1
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status", response_model=StatusResponse, tags=["Ops"])
async def get_status():
    """Service health check."""
    return StatusResponse(
        status      = "ok" if _engine else "degraded",
        model_ready = _engine is not None,
        uptime_s    = round(time.time() - _boot_time, 1),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Ops"])
async def get_stats():
    """Runtime performance statistics."""
    eng_stats = _engine.stats() if _engine else {"total_calls": 0, "avg_latency_ms": 0}
    return StatsResponse(
        calls       = eng_stats["total_calls"],
        avg_ms      = eng_stats["avg_latency_ms"],
        uptime_s    = round(time.time() - _boot_time, 1),
        req_total   = _req_count,
        error_total = _err_count,
    )
