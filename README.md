# Medical RAG API Demo (Free Model Version)

A production-style RAG project for interview preparation using a free local model.

## What this project includes
- FastAPI service
- RAG pipeline with LangChain
- FAISS vector index
- Hugging Face embeddings
- Free local LLM through Ollama
- Redis cache
- PostgreSQL logging
- Docker + docker-compose
- Simple evaluation script

## Dataset
This project uses MedQuAD from Hugging Face (`lavita/MedQuAD`).

## Recommended free model
Default setup uses Ollama with:
- `llama3.1:8b`

You can switch to smaller/faster models by editing `.env`, for example:
- `qwen2.5:7b`
- `gemma2:9b`
- `phi3:mini`

## Setup

### 1) Install Ollama and pull a free model

```bash
ollama pull llama3.1:8b
```

### 2) Prepare the project

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
uvicorn app.main:app --reload
```

## Run with Docker for Postgres + Redis

```bash
cp .env.example .env
mkdir -p storage
python scripts/build_index.py
docker compose up postgres redis -d
uvicorn app.main:app --reload
```

Note: the API connects to Ollama on `http://localhost:11434`, so keep Ollama running on your machine.

## Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session-1",
    "question": "What are the symptoms of asthma?"
  }'
```

## Endpoints
- `GET /health`
- `POST /chat`

## Architecture

1. `scripts/build_index.py` downloads MedQuAD and builds FAISS.
2. `/chat` receives a question.
3. RAG retrieves top-k relevant chunks.
4. Local LLM answers only from retrieved context.
5. Redis caches repeated questions.
6. PostgreSQL stores question, answer, and context for observability.

## Interview talking points
- Why Ollama? Free local inference, easy demos, and no API cost.
- Why RAG? To ground answers in trusted data and reduce hallucination.
- Why FAISS? Lightweight local vector retrieval.
- Why Redis? Faster repeated queries and lower cost.
- Why PostgreSQL? Logging, analytics, and auditability.
- How to improve? Add guardrails, better eval, async workers, and model routing.
