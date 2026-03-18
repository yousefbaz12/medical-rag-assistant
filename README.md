<div align="center">

# 🩺 Medical RAG Assistant

**A production-style medical Q&A system powered by local LLMs, semantic retrieval, and safety guardrails.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Redis](https://img.shields.io/badge/Redis-cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-logging-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

*Built for educational and portfolio purposes — not a clinical diagnostic system.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Setup](#setup)
- [Run with Docker](#run-with-docker)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Safety Guardrails](#safety-guardrails)
- [Engineering Concepts](#engineering-concepts)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Medical RAG Assistant combines **FastAPI**, **FAISS**, and **Ollama** to answer medical questions using retrieval-augmented generation (RAG). Every question is answered by first retrieving the most relevant context from the [MedQuAD](https://github.com/abachaa/MedQuAD) knowledge base, then generating a grounded response with a local LLM — keeping everything private, offline, and source-attributable.

The system includes a **Redis-backed cache**, **PostgreSQL logging**, **emergency keyword safety detection**, **chunk deduplication**, **confidence scoring**, and a **semantic evaluation pipeline**.

---

## System Architecture

<div align="center">
<img width="1023" height="752" alt="image" src="https://github.com/user-attachments/assets/27da0c4c-0cc2-4aa8-8392-30221533609d" />
</div>

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            Medical RAG Assistant                                 │
│                                                                                  │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────────┐    │
│  │ HTTP Client │───▶│                  FastAPI Layer                       │    │
│  │  (curl /    │    │                                                      │    │
│  │   Swagger)  │    │  GET /health  │  POST /chat  │  POST /debug/retrieve │    │
│  └─────────────┘    └────────────────────┬─────────────────────────────────┘    │
│                                          │                                       │
│                     ┌────────────────────▼─────────────────────────────────┐    │
│                     │                 RAG Service                           │    │
│                     │                                                       │    │
│                     │  Step 1 ── Cache Lookup ──────────────▶ Redis        │    │
│                     │              │ HIT → return cached                   │    │
│                     │              │ MISS → continue                       │    │
│                     │              ▼                                       │    │
│                     │  Step 2 ── Safety Check ──── keywords ▶ alert flag  │    │
│                     │              │                                       │    │
│                     │              ▼                                       │    │
│                     │  Step 3 ── Embed Question ────────────▶ HuggingFace │    │
│                     │              │                          all-MiniLM   │    │
│                     │              ▼                                       │    │
│                     │  Step 4 ── FAISS Retrieval ── top-k ──▶ MedQuAD    │    │
│                     │              │                + dedup                │    │
│                     │              ▼                                       │    │
│                     │  Step 5 ── Context Assembly                         │    │
│                     │              │                                       │    │
│                     │              ▼                                       │    │
│                     │  Step 6 ── LLM Generation ────────────▶ Ollama      │    │
│                     │              │                          llama3.1:8b  │    │
│                     │              ▼                                       │    │
│                     │  Step 7 ── Cache Write + DB Log ───────▶ PostgreSQL │    │
│                     └────────────────────┬─────────────────────────────────┘    │
│                                          │                                       │
│                     ┌────────────────────▼─────────────────────────────────┐    │
│                     │  Response: answer · sources · safety_alert ·         │    │
│                     │           latency_ms · low_confidence · triage label │    │
│                     └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Technology | Role |
|---|---|---|
| **API Layer** | FastAPI | Request routing, Pydantic validation, CORS, structured logging |
| **Safety Check** | Rule-based keyword match | Detects emergency symptoms before inference |
| **Embedding** | `all-MiniLM-L6-v2` (HuggingFace) | Encodes question into a 384-dim vector |
| **Vector Store** | FAISS | Similarity search over ~16K MedQuAD Q&A pairs |
| **Deduplication** | Custom | Removes near-duplicate retrieved chunks before context assembly |
| **LLM** | Ollama (`llama3.1:8b`) | Grounded generation with triage label |
| **Cache** | Redis → in-memory fallback | Skips inference for repeated questions (TTL: 30 min) |
| **Database** | PostgreSQL | Logs query, answer, latency, and safety flag per request |

### RAG Pipeline — Traced Example

```
Question: "What are the symptoms of asthma?"
    │
    ├─▶ [Cache]   MISS — no cached result for this session + question
    │
    ├─▶ [Safety]  No emergency keywords matched → safety_alert: false
    │
    ├─▶ [Embed]   "symptoms asthma" → 384-dim float32 vector
    │
    ├─▶ [FAISS]   Top-4 chunks retrieved (L2 distance, lower = better)
    │             score=0.31  confidence=high  source=NHLBI  focus=Asthma/symptoms
    │             score=0.44  confidence=high  source=NHLBI  focus=Asthma/diagnosis
    │             score=0.88  confidence=high  source=NHLBI  focus=Asthma/treatment
    │             score=1.35  confidence=low   → removed by deduplication
    │
    ├─▶ [LLM]    Ollama generates answer grounded in retrieved context only
    │             Ends with triage classification: PRIMARY-CARE
    │
    └─▶ [Store]  Response cached in Redis (TTL=1800s) + logged to PostgreSQL
```

---

## Features

| | Feature | Details |
|---|---|---|
| ⚡ | **FastAPI REST API** | Auto-docs at `/docs`, async handlers, Pydantic validation |
| 🔍 | **Semantic retrieval** | FAISS index with `all-MiniLM-L6-v2` embeddings |
| 🧠 | **Local LLM inference** | Runs via Ollama — no API keys, fully offline |
| 📚 | **MedQuAD knowledge base** | NIH-sourced medical Q&A pairs |
| 🛡️ | **Safety guardrails** | Emergency keyword detection with structured `safety_alert` flag |
| 🏷️ | **Triage labelling** | Every answer ends with SELF-CARE / PRIMARY-CARE / URGENT-CARE / EMERGENCY |
| 🔁 | **Chunk deduplication** | Removes near-duplicate chunks before LLM context assembly |
| 📊 | **Confidence scoring** | Flags low-confidence retrievals via FAISS L2 distance threshold |
| 🐛 | **Retrieval debugging** | `/debug/retrieve` exposes raw chunks with scores and confidence labels |
| 💾 | **PostgreSQL logging** | Persists query, answer, latency, and safety flag per request |
| ⚡ | **Redis / memory cache** | Configurable TTL; auto-falls back to memory if Redis is unavailable |
| 🧪 | **Semantic evaluation** | `evaluate.py` scores answers by embedding cosine similarity |
| 🐳 | **Docker Compose** | One command to run API + PostgreSQL + Redis |
| 🧾 | **Structured logging** | Per-request method / path / status / latency via FastAPI middleware |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115 + Uvicorn |
| LLM inference | Ollama (`llama3.1:8b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (CPU) |
| RAG orchestration | LangChain + LangChain-Ollama + LangChain-HuggingFace |
| Dataset | MedQuAD (`lavita/MedQuAD` via HuggingFace Datasets) |
| Cache | Redis 7 (with in-memory fallback) |
| Database | PostgreSQL 16 + SQLAlchemy 2 |
| Containerisation | Docker + Docker Compose |
| Testing | Pytest + FastAPI TestClient |
| Config | pydantic-settings + `.env` |

---

## Project Structure

```
medical-rag-assistant/
├── app/
│   ├── cache.py          # Redis + in-memory fallback cache
│   ├── config.py         # env + settings (pydantic-settings)
│   ├── db.py             # PostgreSQL models + session (SQLAlchemy)
│   ├── main.py           # FastAPI app, routes, CORS, logging middleware
│   ├── prompts.py        # LLM system prompt + triage instructions
│   ├── rag.py            # RAG pipeline: embed, retrieve, dedup, safety, generate
│   └── schemas.py        # Pydantic request/response models
├── scripts/
│   ├── build_index.py    # Downloads MedQuAD + builds FAISS index
│   └── evaluate.py       # Semantic similarity evaluation (cosine scoring)
├── tests/
│   └── test_api.py       # FastAPI smoke tests (pytest + TestClient)
├── storage/              # FAISS index saved here (git-ignored)
├── .dockerignore
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## API Reference

### `GET /health`

Returns service health, active model, and cache backend stats.

**Response**
```json
{
  "status": "ok",
  "app": "Medical RAG API",
  "environment": "dev",
  "model": "llama3.1:8b",
  "cache": {
    "backend": "redis",
    "hits": 42,
    "misses": 8
  }
}
```

---

### `POST /chat`

Generates a grounded medical response using the full RAG pipeline.

**Request**
```json
{
  "session_id": "demo-1",
  "question": "What are the symptoms of asthma?",
  "top_k": 4
}
```

> `top_k` is optional (1–20). Defaults to `TOP_K` set in `.env`.

**Response**
```json
{
  "answer": "Common symptoms of asthma include wheezing, chest tightness, shortness of breath, and coughing...\n\nTriage: PRIMARY-CARE",
  "sources": [
    {
      "source": "NHLBI",
      "focus": "Asthma",
      "question": "What are the symptoms of Asthma?"
    }
  ],
  "cached": false,
  "safety_alert": false,
  "safety_message": "",
  "latency_ms": 1842.3,
  "low_confidence": false
}
```

---

### `POST /debug/retrieve`

Returns raw retrieved chunks without calling the LLM. Useful for inspecting retrieval quality and tuning `TOP_K`.

**Request**
```json
{
  "session_id": "demo-1",
  "question": "What are the symptoms of asthma?"
}
```

**Response**
```json
{
  "question": "What are the symptoms of asthma?",
  "top_k": 4,
  "results": [
    {
      "score": 0.312,
      "confidence": "high",
      "source": "NHLBI",
      "focus": "Asthma",
      "related_question": "What are the symptoms of Asthma?",
      "content": "Question: What are the symptoms of Asthma?\nAnswer: ..."
    }
  ]
}
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- Docker Desktop (optional — required for PostgreSQL + Redis)

---

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Create a virtual environment

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
ENVIRONMENT=dev
LLM_PROVIDER=ollama
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_INDEX_PATH=./storage/faiss_medquad
TOP_K=4

POSTGRES_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/medical_rag
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=1800
```

### 5. Pull a local model via Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

Alternative models you can use by changing `OLLAMA_MODEL` in `.env`:

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | ~5 GB | Default — best quality |
| `qwen2.5:7b` | ~4.5 GB | Fast and capable |
| `gemma2:9b` | ~5.5 GB | Strong reasoning |
| `phi3:mini` | ~2 GB | Fastest / lowest memory |

### 6. Start PostgreSQL and Redis

```bash
docker compose up postgres redis -d
```

### 7. Build the FAISS index

```bash
python scripts/build_index.py
```

This downloads `lavita/MedQuAD` from HuggingFace (~16K Q&A pairs), creates embeddings, and saves the FAISS index to `./storage/faiss_medquad`. Runs once — safe to re-run to rebuild.

### 8. Run the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI at **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## Run with Docker

Full stack (API + PostgreSQL + Redis) in one command:

```bash
cp .env.example .env
mkdir -p storage
python scripts/build_index.py        # build index on host first
docker compose up --build
```

> The API container connects to Ollama on your host machine. Keep `ollama serve` running. The default `OLLAMA_BASE_URL` in `.env` should be set to `http://host.docker.internal:11434` when running inside Docker.

**Quick smoke test:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-1",
    "question": "What are the symptoms of asthma?"
  }'
```

---

## Evaluation

```bash
python scripts/evaluate.py
```

Each test case is scored on two metrics:

| Metric | Method | Pass threshold |
|---|---|---|
| **Keyword presence** | Expected medical terms present in the answer | ≥ 50% of expected keywords |
| **Semantic similarity** | Cosine similarity between expected and actual answer embeddings | ≥ 0.50 |

Results are saved to `storage/eval_results.json`:

```json
{
  "total": 5,
  "passed": 4,
  "failed": 1,
  "pass_rate": "80.0%",
  "avg_semantic_similarity": 0.6341,
  "rows": [ ... ]
}
```

---

## Tests

```bash
pytest tests/test_api.py -v
```

| Test | What it verifies |
|---|---|
| `test_health` | `/health` returns `status: ok` |
| `test_chat` | `/chat` returns answer, sources, and `cached: false` |
| `test_safety_alert` | Chest pain question triggers `safety_alert: true` |
| `test_debug_retrieve` | `/debug/retrieve` returns chunks with scores |
| `test_custom_top_k` | `top_k` field is accepted and forwarded correctly |

---

## Safety Guardrails

The API includes lightweight rule-based detection for urgent symptom patterns. When triggered, the response sets `safety_alert: true`, includes a `safety_message`, and prepends an `URGENT NOTICE` to the answer.

**Detected patterns:**

`severe chest pain` · `chest pain` · `can't breathe` · `cannot breathe` · `shortness of breath` · `passed out` · `fainting` · `seizure` · `coughing blood` · `vomiting blood` · `blue lips` · `confusion` · `stroke` · `heart attack` · `severe bleeding` · `unconscious` · `not breathing` · `loss of consciousness`

**Example safety response:**

```json
{
  "answer": "URGENT NOTICE: This question may describe urgent symptoms. The user should seek immediate medical attention or emergency care.\n\n...",
  "safety_alert": true,
  "safety_message": "This question may describe urgent symptoms. The user should seek immediate medical attention or emergency care."
}
```

> ⚕️ **Disclaimer:** This project is for educational and engineering demonstration purposes only. It is **not** a real clinical diagnostic system and should never be used as a substitute for professional medical advice.

---

## Engineering Concepts

| Concept | Implementation |
|---|---|
| **Retrieval-Augmented Generation (RAG)** | Grounds LLM outputs in a trusted domain knowledge base to reduce hallucination |
| **Local LLM orchestration** | Privacy-preserving offline inference via Ollama — no data sent to external APIs |
| **Semantic search** | FAISS vector index with sentence-transformer embeddings for fast similarity lookup |
| **Chunk deduplication** | Filters near-duplicate context chunks before LLM generation to improve answer quality |
| **Confidence scoring** | FAISS L2 distance threshold surfaces low-confidence retrievals in the API response |
| **Cache layering** | Redis as primary cache with automatic in-memory fallback — no hard dependency on Redis |
| **Observability** | Per-request latency, safety flags, and full context stored in PostgreSQL for analytics |
| **Semantic evaluation** | Embedding cosine similarity as a meaningful answer quality metric beyond keyword matching |
| **Structured logging** | FastAPI middleware logs every request with method, path, status code, and latency |
| **DB resilience** | SQLAlchemy errors are caught with rollback — DB failures do not crash the API request |

---

## Roadmap

- [ ] Cross-encoder reranking for retrieved chunks
- [ ] Streaming responses via `/chat/stream`
- [ ] Authentication and rate limiting
- [ ] GitHub Actions CI pipeline
- [ ] OpenAI provider fallback (config already supports it via `LLM_PROVIDER=openai`)
- [ ] Expanded evaluation dataset with clinical Q&A pairs

---

## License

MIT — see [LICENSE](LICENSE) for details.
