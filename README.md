<div align="center">

# 🩺 Medical RAG Assistant

**A production-style medical Q&A system powered by local LLMs, semantic retrieval, and safety guardrails.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Built for educational and portfolio purposes — not a clinical diagnostic system.*

</div>

---

## Overview

Medical RAG Assistant combines **FastAPI**, **FAISS**, and **Ollama** to answer medical questions using retrieval-augmented generation. Questions are answered by retrieving relevant context from the [MedQuAD](https://github.com/abachaa/MedQuAD) knowledge base before passing it to a local LLM — keeping everything private and offline.

```
User Question
    ↓
FastAPI (/chat)
    ↓
RAG Service
    ├── 🛡️  Safety Check          ← rule-based urgent symptom detection
    ├── 🔍  FAISS Retrieval        ← semantic search over MedQuAD
    ├── 📝  Context Assembly
    └── 🧠  LLM Generation         ← Ollama (phi3:mini, local)
    ↓
Response + Sources + Safety Metadata
    ↓
SQLite Logging + In-Memory Cache
```

---

## Features

| | Feature | Details |
|---|---|---|
| ⚡ | **FastAPI REST API** | Auto-docs at `/docs`, async handlers, Pydantic validation |
| 🔍 | **Semantic retrieval** | FAISS index with `all-MiniLM-L6-v2` embeddings |
| 🧠 | **Local LLM inference** | Runs via Ollama — no API keys, fully offline |
| 📚 | **MedQuAD knowledge base** | NIH-sourced medical Q&A pairs |
| 🛡️ | **Safety guardrails** | Detects urgent symptoms, flags and prepends warnings |
| 🐛 | **Retrieval debugging** | `/debug/retrieve` exposes raw retrieved chunks |
| 💾 | **SQLite logging** | Lightweight query logging for local development |
| ⚡ | **In-memory caching** | Configurable TTL to skip repeated retrievals |

---

## Tech Stack

`Python` · `FastAPI` · `FAISS` · `LangChain` · `Ollama` · `Sentence Transformers` · `SQLite` · `Pydantic` · `SQLAlchemy`

---

## Project Structure

```
medical-rag-assistant/
├── app/
│   ├── cache.py        # in-memory query cache
│   ├── config.py       # env + settings
│   ├── db.py           # SQLite logging
│   ├── main.py         # FastAPI app + routes
│   ├── prompts.py      # LLM prompt templates
│   ├── rag.py          # core RAG service
│   └── schemas.py      # Pydantic request/response models
├── scripts/
│   └── build_index.py  # builds FAISS index from MedQuAD
├── .env.example
├── ARCHITECTURE.md
├── README.md
└── requirements.txt
```

---

## API Reference

### `GET /health`
Returns service health status.

---

### `POST /chat`
Generates a grounded medical response using RAG.

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
  "answer": "Common symptoms of asthma include wheezing, chest tightness, shortness of breath, and coughing...",
  "sources": [
    {
      "source": "NHLBI",
      "focus": "unknown",
      "question": "What are the symptoms of Asthma?"
    }
  ],
  "cached": false,
  "safety_alert": false,
  "safety_message": ""
}
```

---

### `POST /debug/retrieve`
Returns the raw retrieved chunks used during semantic search — useful for inspecting retrieval quality and tuning `TOP_K`.

**Request**
```json
{
  "session_id": "demo-1",
  "question": "What are the symptoms of asthma?"
}
```

---

## Setup

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed locally

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Create virtual environment

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
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_INDEX_PATH=./storage/faiss_medquad
TOP_K=4
POSTGRES_URL=sqlite:///./medical_rag.db
CACHE_TTL_SECONDS=1800
```

### 5. Start Ollama and pull model

```bash
ollama serve
ollama pull phi3:mini
```

### 6. Build the FAISS index

```bash
python scripts/build_index.py
```

### 7. Run the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI at **http://localhost:8000/docs**

---

## Safety Guardrails

The API includes lightweight rule-based detection for urgent symptom patterns. When triggered, the response includes `safety_alert: true`, a warning message, and an urgent notice prepended to the answer.

Detected patterns include: `severe chest pain` · `difficulty breathing` · `fainting` · `seizure` · `severe bleeding` · `vomiting blood`

> ⚕️ **Disclaimer:** This project is for educational and engineering demonstration purposes only. It is **not** a real clinical diagnostic system and should not be used for medical advice.

---

## Engineering Concepts

This project demonstrates:

- **Retrieval-Augmented Generation (RAG)** — grounding LLM outputs in retrieved knowledge
- **Local LLM orchestration** — privacy-preserving inference with Ollama
- **Semantic search** — FAISS vector index with sentence-transformer embeddings
- **Observability** — retrieval debugging endpoint for inspecting pipeline internals
- **Safety escalation** — rule-based guardrails with structured API metadata
- **Backend API design** — production-style FastAPI with caching, logging, and validation

---

## Roadmap

- [ ] Reranking for retrieved chunks
- [ ] Streaming responses
- [ ] Structured evaluation pipeline
- [ ] PostgreSQL + Redis production profiles
- [ ] Authentication and rate limiting
- [ ] Tests and CI pipeline
- [ ] Docker support

---

## License

MIT — see [LICENSE](LICENSE) for details.
