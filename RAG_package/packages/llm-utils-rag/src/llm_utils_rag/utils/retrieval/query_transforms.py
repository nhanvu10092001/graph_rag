"""Query transformation pipeline for improved retrieval recall.

Provides composable query transformations that run BEFORE retrieval
to generate better or multiple query variants for improved recall.

Strategies:
- StepBackTransform: Creates a broader/abstract question
- DecompositionTransform: Splits complex query into sub-queries
- ExpansionTransform: Adds synonyms and related terms
- ContextualTransform: Uses conversation history to enrich query
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryDecompositionResult(BaseModel):
    sub_queries: List[str] = Field(
        default_factory=list,
        description="List of simple, self-contained sub-questions derived from the original question"
    )


# ── Base Interface ───────────────────────────────────────────────────────────


class BaseQueryTransform(ABC):
    """Abstract base for query transformations."""

    @abstractmethod
    async def transform(self, query: str, **kwargs) -> List[str]:
        """Transform a query into one or more query variants.

        Args:
            query: Original user query.

        Returns:
            List of transformed queries (may include the original).
        """
        ...


# ── Step-Back Prompting ──────────────────────────────────────────────────────


class StepBackTransform(BaseQueryTransform):
    """Creates a broader, more abstract version of the query.

    Example:
        "What was Apple's revenue in Q3 2024?" →
        "What are Apple's recent financial performance metrics?"
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def transform(self, query: str, **kwargs) -> List[str]:
        prompt = f"""Given the following specific question, generate a broader, more general version 
that would help retrieve background information needed to answer the original question.

Original question: {query}

Respond with ONLY the broader question, nothing else."""

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = self._clean_response(text.strip())

            if text and text != query:
                logger.info("StepBack: '%s' → '%s'", query[:50], text[:50])
                return [query, text]  # Return both original and step-back
        except Exception as e:
            logger.warning("StepBack transform failed: %s", e)

        return [query]

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean LLM response of thinking tags and markdown."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip().strip('"').strip("'")
        return text


# ── Query Decomposition ─────────────────────────────────────────────────────


class DecompositionTransform(BaseQueryTransform):
    """Decomposes a complex query into simpler sub-queries.

    Example:
        "Compare Python and Java for web development and which has better ML support?" →
        ["Python web development features",
         "Java web development features",
         "Python machine learning support",
         "Java machine learning support"]
    """

    def __init__(self, llm: BaseLanguageModel, max_sub_queries: int = 3):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    async def transform(self, query: str, **kwargs) -> List[str]:
        prompt = f"""Break down the following complex question into {self.max_sub_queries} or fewer simpler sub-questions.
Each sub-question should be self-contained and searchable independently.

Question: {query}"""

        if hasattr(self.llm, "with_structured_output"):
            try:
                structured_llm = self.llm.with_structured_output(QueryDecompositionResult)
                res = await structured_llm.ainvoke(prompt)
                sub_queries = getattr(res, "sub_queries", None) or (res.get("sub_queries") if isinstance(res, dict) else None)
                if sub_queries and isinstance(sub_queries, list):
                    return [str(q) for q in sub_queries[: self.max_sub_queries]]
            except Exception as e:
                logger.warning("Structured decomposition transform failed: %s, using fallback", e)

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = self._clean_response(text)

            sub_queries = self._parse_queries(text, query)
            logger.info(
                "Decomposition: '%s' → %d sub-queries", query[:50], len(sub_queries)
            )
            return sub_queries

        except Exception as e:
            logger.warning("Decomposition transform failed: %s", e)
            return [query]

    def _parse_queries(self, text: str, original: str) -> List[str]:
        """Parse sub-queries from LLM response."""
        # Try JSON parse
        try:
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[: self.max_sub_queries]
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from text
        match = re.search(r"\[([^\]]+)\]", text)
        if match:
            try:
                queries = json.loads(f"[{match.group(1)}]")
                if isinstance(queries, list):
                    return [str(q) for q in queries[: self.max_sub_queries]]
            except json.JSONDecodeError:
                pass

        # Fallback: return original
        return [original]

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean thinking tags and code fences."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        return text.strip()


# ── Query Expansion ──────────────────────────────────────────────────────────


class ExpansionTransform(BaseQueryTransform):
    """Expands the query with synonyms and related terms.

    Example:
        "ML model accuracy" →
        "machine learning model accuracy precision recall performance metrics"
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def transform(self, query: str, **kwargs) -> List[str]:
        prompt = f"""Given this search query, create an expanded version that includes 
synonyms, related terms, and alternative phrasings to improve search recall.
Keep it as a single search query (not multiple queries).

Original: {query}

Respond with ONLY the expanded query, nothing else."""

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = text.strip().strip('"').strip("'")

            if text and text != query:
                logger.info("Expansion: '%s' → '%s'", query[:50], text[:50])
                return [query, text]
        except Exception as e:
            logger.warning("Expansion transform failed: %s", e)

        return [query]


# ── Contextual Transform ────────────────────────────────────────────────────


class ContextualTransform(BaseQueryTransform):
    """Uses conversation history to contextualize the query.

    Example:
        History: "Tell me about Python"
        Query: "What about its performance?" →
        "What is Python's performance characteristics?"
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def transform(self, query: str, **kwargs) -> List[str]:
        history = kwargs.get("chat_history", [])
        if not history:
            return [query]

        # Build history context (last 3 messages)
        history_text = "\n".join(
            f"{'User' if i % 2 == 0 else 'AI'}: {msg}"
            for i, msg in enumerate(history[-6:])
        )

        prompt = f"""Given the conversation history and the latest query, 
rewrite the query to be self-contained (resolve pronouns and references).

Conversation:
{history_text}

Latest query: {query}

Respond with ONLY the rewritten query, nothing else."""

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = text.strip().strip('"').strip("'")

            if text and text != query:
                logger.info("Contextual: '%s' → '%s'", query[:50], text[:50])
                return [text]  # Replace original with contextualized
        except Exception as e:
            logger.warning("Contextual transform failed: %s", e)

        return [query]


# ── Pipeline ─────────────────────────────────────────────────────────────────


class QueryTransformPipeline:
    """Composable pipeline that chains multiple query transformations.

    Usage:
        pipeline = QueryTransformPipeline.from_config(config, llm)
        transformed_queries = await pipeline.transform("my query")
        # Use each query variant to retrieve docs, then merge results
    """

    def __init__(self, transforms: List[BaseQueryTransform]):
        self.transforms = transforms

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        llm: BaseLanguageModel,
    ) -> Optional["QueryTransformPipeline"]:
        """Create pipeline from config dict.

        Config structure:
            query_transform:
                enabled: true
                strategies: ["step_back", "decomposition", "expansion", "contextual"]
                max_sub_queries: 3
        """
        transform_config = config.get("query_transform", {})

        if not transform_config.get("enabled", False):
            return None

        strategies = transform_config.get("strategies", ["step_back"])
        max_sub_queries = transform_config.get("max_sub_queries", 3)

        transforms = []
        for strategy in strategies:
            if strategy == "step_back":
                transforms.append(StepBackTransform(llm))
            elif strategy == "decomposition":
                transforms.append(DecompositionTransform(llm, max_sub_queries))
            elif strategy == "expansion":
                transforms.append(ExpansionTransform(llm))
            elif strategy == "contextual":
                transforms.append(ContextualTransform(llm))
            else:
                logger.warning("Unknown query transform strategy: %s", strategy)

        if not transforms:
            return None

        logger.info(
            "QueryTransformPipeline created with %d strategies: %s",
            len(transforms),
            strategies,
        )
        return cls(transforms)

    async def transform(self, query: str, **kwargs) -> List[str]:
        """Apply all transforms and collect unique query variants.

        Returns:
            Deduplicated list of query variants (always includes original).
        """
        all_queries = {query}  # Always include original

        for transform in self.transforms:
            try:
                variants = await transform.transform(query, **kwargs)
                all_queries.update(variants)
            except Exception as e:
                logger.warning(
                    "%s failed: %s", transform.__class__.__name__, e
                )

        result = list(all_queries)
        logger.info(
            "Query transform pipeline: '%s' → %d variants",
            query[:50],
            len(result),
        )
        return result
