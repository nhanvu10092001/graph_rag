"""Factory for creating Embeddings instances based on provider configuration."""

from typing import Any, Dict
import os
from langchain_core.embeddings import Embeddings

from llm_utils_llm.logger import logger


class EmbeddingsFactory:
    """Factory for creating Embeddings instances based on provider configuration."""

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, factory_func):
        """Register a new embeddings provider factory function."""
        logger.info(f"Registering embeddings provider: {name}")
        cls._providers[name.lower()] = factory_func
        logger.info(f"Successfully registered embeddings provider: {name}")

    @classmethod
    def create_embeddings(cls, config: Dict[str, Any]) -> Embeddings:
        """Create Embeddings instance based on configuration.

        Args:
            config: Embeddings configuration containing provider and settings

        Returns:
            Embeddings: Configured embeddings instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.get("provider", "openai").lower()
        logger.info(f"Creating embeddings instance with provider: {provider}")

        if provider not in cls._providers:
            logger.info(f"Provider {provider} not registered, trying to register common providers")
            # Try to import common providers
            cls._try_register_common_providers()

        if provider in cls._providers:
            logger.info(f"Successfully creating embeddings instance for provider: {provider}")
            return cls._providers[provider](config)
        else:
            logger.error(f"Unsupported embeddings provider: {provider}")
            raise ValueError(f"Unsupported embeddings provider: {provider}")

    @classmethod
    def _try_register_common_providers(cls):
        """Try to register common providers if available."""
        # Try OpenAI
        try:
            from langchain_openai import OpenAIEmbeddings

            def create_openai(config):
                openai_config = config.get("openai", {})
                api_base = (
                    openai_config.get("openai_api_base")
                    or openai_config.get("base_url")
                    or os.getenv("OPENAI_API_BASE")
                    or os.getenv("OPENAI_BASE_URL")
                )
                check_ctx = openai_config.get("check_embedding_ctx_length", False)
                tiktoken_en = openai_config.get("tiktoken_enabled", False)
                return OpenAIEmbeddings(
                    model=openai_config.get("model", "text-embedding-ada-002"),
                    api_key=openai_config.get("openai_api_key") or openai_config.get("api_key") or os.getenv("OPENAI_API_KEY", "test"),
                    openai_api_base=api_base or None,
                    check_embedding_ctx_length=check_ctx,
                    tiktoken_enabled=tiktoken_en,
                )

            cls.register_provider("openai", create_openai)
        except ImportError:
            pass
