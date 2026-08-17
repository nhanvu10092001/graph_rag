"""LLM factory with provider-based initialization."""

import logging

logger = logging.getLogger(__name__)


def create_llm(config: dict):
    """Create LLM instance from config dict.

    Args:
        config: Dict with keys: provider, model, temperature,
                plus provider-specific keys (anthropic_api_key, openai_api_key, etc.)
    """
    provider = config["provider"]
    model = config["model"]
    temperature = config["temperature"]

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        anthropic_key = config.get("anthropic_api_key")
        anthropic_url = config.get("anthropic_api_url")
        max_tokens = config.get("max_tokens", 4096)
        logger.info(f"Using ChatAnthropic: {model} (Base URL: {anthropic_url})")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=anthropic_key or None,
            anthropic_api_url=anthropic_url or None,
        )
    else:
        from langchain_openai import ChatOpenAI

        openai_key = config.get("openai_api_key")
        openai_base = config.get("openai_api_base")
        logger.info(f"Using ChatOpenAI: {model} (Base URL: {openai_base})")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=openai_key or None,
            openai_api_base=openai_base or None,
        )
