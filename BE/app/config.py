"""Unified configuration manager loading from .env and graph_rag_config.yaml."""

import copy
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent.parent / "graph_rag_config.yaml"


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


class Settings:
    """Unified settings from .env + graph_rag_config.yaml.

    Resolution order: env var > YAML value > hardcoded default.
    """

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
        self.llm_provider = os.getenv("LLM_PROVIDER") or llm_dict.get("provider") or "openai"

        if self.llm_provider == "claude":
            claude_dict = llm_dict.get("claude", {})
            self.llm_model = os.getenv("LLM_MODEL") or claude_dict.get("model") or "claude-haiku-4-5-20251001"
            self.llm_temperature = float(os.getenv("LLM_TEMPERATURE") or claude_dict.get("temperature") or 0.7)
            self.llm_max_tokens = int(claude_dict.get("max_tokens") or 4096)
            self.anthropic_api_key = (
                os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("ANTHROPIC_AUTH_TOKEN")
                or claude_dict.get("api_key")
                or ""
            )
            self.anthropic_api_url = (
                os.getenv("LLM_ANTHROPIC_API_URL")
                or os.getenv("ANTHROPIC_BASE_URL")
                or claude_dict.get("base_url")
                or None
            )
            self.openai_api_key = os.getenv("OPENAI_API_KEY") or ""
            self.openai_api_base = None
        else:
            openai_dict = llm_dict.get("openai", {})
            self.llm_model = os.getenv("LLM_MODEL") or openai_dict.get("model") or "gpt-4o-mini"
            self.llm_temperature = float(os.getenv("LLM_TEMPERATURE") or openai_dict.get("temperature") or 0.7)
            self.llm_max_tokens = 4096
            self.openai_api_key = os.getenv("OPENAI_API_KEY") or openai_dict.get("api_key") or ""
            self.openai_api_base = (
                os.getenv("LLM_OPENAI_API_BASE")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or openai_dict.get("base_url")
                or None
            )
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or ""
            self.anthropic_api_url = None

        # Embeddings Config
        emb_dict = config_dict.get("embeddings", {})
        self.embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER") or emb_dict.get("provider") or "openai"
        self.embeddings_model = (
            os.getenv("EMBEDDINGS_MODEL")
            or emb_dict.get("openai", {}).get("model")
            or "text-embedding-3-small"
        )
        self._emb_openai_base = (
            os.getenv("EMBEDDINGS_OPENAI_API_BASE")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or emb_dict.get("openai", {}).get("base_url")
            or self.openai_api_base
        )

        # Server Config
        server_dict = config_dict.get("server", {})
        self.server_host = os.getenv("SERVER_HOST") or server_dict.get("host") or "0.0.0.0"
        self.server_port = int(os.getenv("SERVER_PORT") or server_dict.get("port") or 8000)

        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            self.cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        elif "cors_origins" in server_dict:
            raw_origins = server_dict["cors_origins"]
            if isinstance(raw_origins, list):
                self.cors_origins = [str(o).strip() for o in raw_origins if str(o).strip()]
            elif isinstance(raw_origins, str):
                self.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
            else:
                self.cors_origins = ["http://localhost:3000"]
        else:
            self.cors_origins = ["http://localhost:3000"]

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
        self.minio_secure = _parse_bool(os.getenv("MINIO_SECURE") or minio_dict.get("secure") or False)
        self.minio_bucket = os.getenv("MINIO_BUCKET") or minio_dict.get("bucket") or "documents"

    def validate(self):
        """Validate crucial parameters."""
        if not self.neo4j_uri:
            raise ValueError("Neo4j URI is required")
        if not self.neo4j_username:
            raise ValueError("Neo4j username is required")
        if not self.neo4j_password:
            raise ValueError("Neo4j password is required")
        if self.llm_provider == "claude":
            if not self.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
                raise ValueError("ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is required for claude provider")
        else:
            if not self.openai_api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required")

    def get_graph_rag_config(self) -> Dict[str, Any]:
        """Build the config dict consumed by Graph RAG services.

        Returns the exact dict shape expected by GraphRAGPlugin,
        GraphIndexingService, RerankerFactory, CommunityDetectionService,
        GlobalSearchService, and ArkQueryService.
        """
        openai_key = os.getenv("OPENAI_API_KEY") or self.openai_api_key

        llm_config = {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
        }
        if self.llm_provider == "claude":
            llm_config["anthropic_api_key"] = self.anthropic_api_key or None
            llm_config["anthropic_api_url"] = self.anthropic_api_url or None
            llm_config["max_tokens"] = self.llm_max_tokens
        else:
            llm_config["openai_api_key"] = self.openai_api_key or None
            llm_config["openai_api_base"] = self.openai_api_base or None

        # Extraction / OCR config overrides
        extraction_cfg = copy.deepcopy(self.raw.get("extraction", {}))
        paddle_url = os.getenv("PADDLE_OCR_API_URL") or os.getenv("OCR_PADDLE_API_URL")
        if paddle_url:
            ocr_cfg = extraction_cfg.setdefault("ocr", {})
            paddle_cfg = ocr_cfg.setdefault("paddle", {})
            paddle_cfg["api_url"] = paddle_url
            paddle_cfg["base_url"] = paddle_url

        return {
            "neo4j": {
                "uri": self.neo4j_uri,
                "username": self.neo4j_username,
                "password": self.neo4j_password,
                "database": self.neo4j_database,
            },
            "llm": llm_config,
            "embeddings": {
                "provider": "openai",
                "openai": {
                    "model": self.embeddings_model,
                    "openai_api_key": openai_key or None,
                    "openai_api_base": self._emb_openai_base or None,
                    "check_embedding_ctx_length": False,
                    "tiktoken_enabled": False,
                },
            },
            "reranking": self.raw.get(
                "reranking", {"enabled": False, "provider": "flashrank", "top_k": 5}
            ),
            "community_detection": self.raw.get(
                "community_detection",
                {
                    "enabled": True,
                    "algorithm": "leiden",
                    "resolution": 1.0,
                    "max_levels": 3,
                    "auto_rebuild": False,
                },
            ),
            "query": self.raw.get(
                "query",
                {
                    "search_mode": "auto",
                    "global_search": {"max_communities": 10, "default_level": 0},
                },
            ),
            "extraction": extraction_cfg,
            "subagents": self.raw.get("subagents", {"enabled": True, "agents": []}),
            "ark": self.raw.get("ark", {}),
            "chunking": self.raw.get("chunking", {}),
        }


def load_config(config_path: Path = CONFIG_PATH) -> Settings:
    """Load settings from graph_rag_config.yaml or return defaults from environment variables."""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
                return Settings(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load {config_path.name} ({e}). Using env fallback.")

    return Settings({})


# Global settings instance
settings = load_config()
try:
    settings.validate()
except Exception as e:
    logger.warning(f"Settings validation warning on startup: {e}")
