from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical RAG API"
    environment: str = "dev"

    llm_provider: str = "ollama"  # ollama or openai
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_path: str = "./storage/faiss_medquad"
    top_k: int = 4

    postgres_url: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/medical_rag"
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 1800

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
