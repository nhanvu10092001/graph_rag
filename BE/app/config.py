"""Configuration manager loading properties from config.yaml."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load env variables from .env file (located in BE root)
load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class Settings:
    """Settings class parsed from config.yaml."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # Neo4j Config
        neo4j_dict = config_dict.get("neo4j", {})
        self.neo4j_uri = os.getenv("NEO4J_URI") or neo4j_dict.get("uri") or "bolt://localhost:7687"
        self.neo4j_username = os.getenv("NEO4J_USERNAME") or neo4j_dict.get("username") or "neo4j"
        self.neo4j_password = os.getenv("NEO4J_PASSWORD") or neo4j_dict.get("password") or "password"
        self.neo4j_database = os.getenv("NEO4J_DATABASE") or neo4j_dict.get("database") or "neo4j"

        # LLM Config
        llm_dict = config_dict.get("llm", {})
        self.llm_provider = os.getenv("LLM_PROVIDER") or llm_dict.get("provider") or "ollama"
        self.llm_model = os.getenv("LLM_MODEL") or llm_dict.get("model") or "llama3:8b"
        self.llm_base_url = os.getenv("LLM_BASE_URL") or llm_dict.get("base_url") or "http://localhost:11434"
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE") or llm_dict.get("temperature") or 0.0)
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or llm_dict.get("api_key") or ""

        # Embeddings Config
        emb_dict = config_dict.get("embeddings", {})
        self.embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER") or emb_dict.get("provider") or "ollama"
        self.embeddings_model = os.getenv("EMBEDDINGS_MODEL") or emb_dict.get("model") or "nomic-embed-text"
        self.embeddings_base_url = os.getenv("EMBEDDINGS_BASE_URL") or emb_dict.get("base_url") or "http://localhost:11434"

        # Server Config
        server_dict = config_dict.get("server", {})
        self.server_host = os.getenv("SERVER_HOST") or server_dict.get("host") or "0.0.0.0"
        self.server_port = int(os.getenv("SERVER_PORT") or server_dict.get("port") or 8000)

        # Postgres Config
        pg_dict = config_dict.get("postgres", {})
        self.pg_host = os.getenv("PG_HOST") or pg_dict.get("host") or "localhost"
        self.pg_port = int(os.getenv("PG_PORT") or pg_dict.get("port") or 5432)
        self.pg_user = os.getenv("PG_USER") or pg_dict.get("user") or "postgres"
        self.pg_password = os.getenv("PG_PASSWORD") or pg_dict.get("password") or "password"
        self.pg_database = os.getenv("PG_DATABASE") or pg_dict.get("database") or "graph_rag_db"

        # MinIO Config
        minio_dict = config_dict.get("minio", {})
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT") or minio_dict.get("endpoint") or "localhost:9000"
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY") or minio_dict.get("access_key") or "minioadmin"
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY") or minio_dict.get("secret_key") or "minioadmin"
        self.minio_secure = bool(os.getenv("MINIO_SECURE") or minio_dict.get("secure") or False)
        self.minio_bucket = os.getenv("MINIO_BUCKET") or minio_dict.get("bucket") or "documents"

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
