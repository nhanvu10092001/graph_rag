"""Factory for creating LLM instances based on provider configuration."""
import os
from typing import Any, Dict

from langchain_core.language_models import BaseLLM

from llm_utils_llm.logger import logger


class LLMFactory:
    """Factory for creating LLM instances based on provider configuration."""

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, factory_func):
        """Register a new LLM provider factory function."""
        logger.info(f"Registering LLM provider: {name}")
        cls._providers[name.lower()] = factory_func
        logger.info(f"Successfully registered LLM provider: {name}")

    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> BaseLLM:
        """Create LLM instance based on configuration.

        Args:
            config: LLM configuration containing provider and settings

        Returns:
            BaseLLM: Configured LLM instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.get("provider", "openai").lower()
        logger.info(f"Creating LLM instance with provider: {provider}")

        if provider not in cls._providers:
            logger.info(f"Provider {provider} not registered, trying to register common providers")
            # Try to import common providers
            cls._try_register_common_providers()

        if provider in cls._providers:
            logger.info(f"Successfully creating LLM instance for provider: {provider}")
            return cls._providers[provider](config)
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def _try_register_common_providers(cls):
        """Try to register common providers if available."""
        # Try OpenAI
        try:
            from langchain_openai import ChatOpenAI

            def create_openai(config):
                openai_config = config.get("openai", {})
                api_base = openai_config.get("openai_api_base") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
                return ChatOpenAI(
                    model=openai_config.get("model", "gpt-3.5-turbo-instruct"),
                    temperature=openai_config.get("temperature", 0.7),
                    api_key=openai_config.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
                    openai_api_base=api_base or None,
                )

            cls.register_provider("openai", create_openai)
        except ImportError:
            pass

        # Try Anthropic/Claude
        try:
            from langchain_anthropic import ChatAnthropic

            def create_claude(config):
                claude_config = config.get("claude", {})
                api_key = claude_config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
                api_url = claude_config.get("anthropic_api_url") or os.getenv("ANTHROPIC_BASE_URL")
                return ChatAnthropic(
                    model=claude_config.get("model", "claude-haiku-4-5-20251001"),
                    temperature=claude_config.get("temperature", 0.7),
                    max_tokens=int(claude_config.get("max_tokens", 4096)),
                    anthropic_api_key=api_key or None,
                    anthropic_api_url=api_url or None,
                )

            cls.register_provider("claude", create_claude)
        except ImportError:
            pass
