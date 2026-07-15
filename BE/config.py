"""Configuration manager loading properties from config.yaml."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv(Path(__file__).parent / ".env")

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class Settings:
    """Settings class parsed from config.yaml."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # Neo4j Config
        neo4j_dict = config_dict.get("neo4j", {})
        self.neo4j_uri = neo4j_dict.get("uri") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_dict.get("username") or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_dict.get("password") or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_dict.get("database") or os.getenv("NEO4J_DATABASE", "neo4j")

        # LLM Config
        llm_dict = config_dict.get("llm", {})
        self.llm_provider = llm_dict.get("provider") or os.getenv("LLM_PROVIDER", "ollama")
        self.llm_model = llm_dict.get("model") or os.getenv("LLM_MODEL", "llama3:8b")
        self.llm_base_url = llm_dict.get("base_url") or os.getenv("LLM_BASE_URL", "http://localhost:11434")
        self.llm_temperature = float(llm_dict.get("temperature", 0.0))
        self.openai_api_key = llm_dict.get("api_key") or os.getenv("OPENAI_API_KEY", "")

        # Embeddings Config
        emb_dict = config_dict.get("embeddings", {})
        self.embeddings_provider = emb_dict.get("provider") or os.getenv("EMBEDDINGS_PROVIDER", "ollama")
        self.embeddings_model = emb_dict.get("model") or os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
        self.embeddings_base_url = emb_dict.get("base_url") or os.getenv("EMBEDDINGS_BASE_URL", "http://localhost:11434")

        # Server Config
        server_dict = config_dict.get("server", {})
        self.server_host = server_dict.get("host", "0.0.0.0")
        self.server_port = int(server_dict.get("port", 8000))

        # Postgres Config
        pg_dict = config_dict.get("postgres", {})
        self.pg_host = pg_dict.get("host") or os.getenv("PG_HOST", "localhost")
        self.pg_port = int(pg_dict.get("port") or os.getenv("PG_PORT", 5432))
        self.pg_user = pg_dict.get("user") or os.getenv("PG_USER", "postgres")
        self.pg_password = pg_dict.get("password") or os.getenv("PG_PASSWORD", "password")
        self.pg_database = pg_dict.get("database") or os.getenv("PG_DATABASE", "graph_rag_db")

        # MinIO Config
        minio_dict = config_dict.get("minio", {})
        self.minio_endpoint = minio_dict.get("endpoint") or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key = minio_dict.get("access_key") or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = minio_dict.get("secret_key") or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.minio_secure = bool(minio_dict.get("secure", False))
        self.minio_bucket = minio_dict.get("bucket") or os.getenv("MINIO_BUCKET", "documents")

    def validate(self):
        """Validate crucial parameters."""
        if not self.neo4j_uri:
            raise ValueError("Neo4j URI is required")
        if not self.neo4j_username:
            raise ValueError("Neo4j username is required")
        if not self.neo4j_password:
            raise ValueError("Neo4j password is required")
        if self.llm_provider not in ["ollama", "openai"]:
            raise ValueError("LLM provider must be either 'ollama' or 'openai'")
        if self.embeddings_provider not in ["ollama", "openai"]:
            raise ValueError("Embeddings provider must be either 'ollama' or 'openai'")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings from config.yaml or return defaults from environment variables."""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
                return Settings(config_dict)
        except Exception as e:
            print(f"Warning: Failed to load config.yaml ({e}). Using env fallback.")
    
    return Settings({})


# Global settings instance
settings = load_config()
settings.validate()
