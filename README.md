# 🧠 Review Sentiment API — Advanced Data Science Project (Month 5)

**Specialization:** Natural Language Processing (NLP)
**Model:** Bidirectional GRU Neural Network (NumPy)
**API:** FastAPI — `/analyse`, `/analyse/bulk`, `/status`, `/stats`
**Deploy:** Docker (Alpine) + Kubernetes (3–12 pods, dual-metric HPA)

---

## Project Highlights

| Item | Detail |
|---|---|
| Model Type | Bidirectional GRU (distinct from LSTM) |
| Dataset | 700 labelled customer reviews (3 classes) |
| Test Suite | **30 unit tests — ALL PASSING** |
| API Endpoints | `/analyse` `/analyse/bulk` `/status` `/stats` |
| Container | Multi-stage Alpine Dockerfile |
| Orchestration | Kubernetes + HPA (CPU + Memory metrics) |
| Monitoring | Rolling-window KL-drift + latency percentiles |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python src/trainer/run_training.py

# Run inference demo
python src/server/engine.py

# Run tests
python tests/test_all.py

# Start API
uvicorn src.server.api:app --port 8080 --reload

# Docker
docker-compose up --build
```

---

## Project Structure

```
nlp_project/
├── README.md
├── requirements.txt
├── docker-compose.yml
│
├── container/
│   └── Dockerfile              # Multi-stage Alpine build
│
├── data/
│   ├── reviews.csv             # 700 labelled reviews
│   └── sales.csv               # 2000 supermarket sales records
│
├── src/
│   ├── pipeline/
│   │   └── text_pipeline.py    # PipelineConfig, TextCleaner, Vocabulary
│   ├── network/
│   │   ├── sentiment_net.py    # BiGRU model (EmbedLayer, GRUCell, Linear)
│   │   └── saved/              # checkpoint.pkl, vocab.json, config.json
│   ├── trainer/
│   │   └── run_training.py     # Callback-based training loop
│   ├── server/
│   │   ├── engine.py           # Context-manager InferenceEngine
│   │   └── api.py              # FastAPI REST application
│   └── watcher/
│       └── monitor.py          # RollingMonitor (ring buffer + KL drift)
│
├── tests/
│   └── test_all.py             # 30 unit tests ✅
│
├── deployment/
│   ├── nginx.conf              # Least-conn proxy, rate limiting
│   └── k8s.yaml                # Deployment + ConfigMap + Service + HPA
│
├── monitoring/
│   └── logs/                   # JSON snapshots from RollingMonitor
│
├── notebooks/
│   └── exploration.ipynb
│
├── docs/
│   └── technical_notes.md
│
└── scripts/
    └── run_all.sh
```

---

## API Reference

### POST `/analyse`
```json
{ "review": "Outstanding quality — highly recommend!" }
```
**Response:**
```json
{
  "review": "Outstanding quality — highly recommend!",
  "sentiment": "Positive",
  "score_id": 2,
  "icon": "😊",
  "confidence": 0.98,
  "breakdown": {"Negative": 0.01, "Neutral": 0.01, "Positive": 0.98},
  "ms": 8.3
}
```

### POST `/analyse/bulk`
```json
{ "reviews": ["Great!", "Awful.", "Okay product."] }
```

### GET `/status` — Health check
### GET `/stats` — Runtime metrics

---

## Model Architecture

```
tokens (B, 90)
   ↓
EmbedLayer [127 × 64]       → (B, 90, 64)
   ↓
BiGRU Layer-1 [48 × 2]      → (B, 90, 96)
   ↓
GRU Layer-2  [24]            → (B, 24)
   ↓
Linear + ReLU [20]           → (B, 20)
   ↓
Linear + Softmax [3]         → (B, 3) probabilities
```
**Total params: 49,587**

---

## Analysis Questions Answered

1. **Architecture impact:** GRU uses 2 gates vs LSTM's 3, converges faster on small datasets with fewer parameters.
2. **Best preprocessing:** Lowercasing + URL/HTML strip + stopword removal reduced vocab by ~35%.
3. **Production latency:** Batched vectorised NumPy forward pass + model kept in memory → avg <15ms per request.
4. **Ethical considerations:** Confidence scores surface ambiguous predictions; drift monitoring detects distribution shift.
5. **Edge cases:** Short/ambiguous inputs handled via `<UNK>` tokens; Neutral class absorbs borderline sentiment.
