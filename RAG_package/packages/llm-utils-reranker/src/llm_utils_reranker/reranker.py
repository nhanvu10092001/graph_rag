"""Reranking module for RAG and Graph RAG pipelines.

Provides a pluggable reranking layer supporting FlashRank, CrossEncoder, and LLM-based strategies.
Each reranker supports both async (rerank) and sync (rerank_sync) execution paths.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMRerankResult(BaseModel):
    ranking: List[int] = Field(
        default_factory=list,
        description="List of candidate document indices (integers) ranked from most to least relevant"
    )


# ── Base Interface ───────────────────────────────────────────────────────────


class BaseReranker(ABC):
    """Abstract base class for document rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """Rerank documents asynchronously by relevance to query, returning top_k."""
        ...

    @abstractmethod
    def rerank_sync(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """Rerank documents synchronously by relevance to query, returning top_k."""
        ...

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on page_content."""
        seen = set()
        unique = []
        for doc in documents:
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        return unique


# ── FlashRank Reranker ───────────────────────────────────────────────────────


class FlashRankReranker(BaseReranker):
    """Lightweight reranker using FlashRank (no GPU required, ~30ms/query)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", **kwargs):
        self.model_name = model_name
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def _perform_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not documents:
            return []

        documents = self._deduplicate(documents)
        if len(documents) <= 1:
            return documents[:top_k]

        try:
            from flashrank import RerankRequest

            ranker = self._get_ranker()

            # Convert to FlashRank format
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            # Map back to Documents, sorted by score desc
            reranked = []
            for result in results[:top_k]:
                idx = result["id"]
                doc = documents[idx]
                doc.metadata["rerank_score"] = result["score"]
                doc.metadata["rerank_provider"] = "flashrank"
                reranked.append(doc)

            logger.info(
                f"FlashRank reranked {len(documents)} → {len(reranked)} docs "
                f"(scores: {[round(r['score'], 3) for r in results[:top_k]]})"
            )
            return reranked

        except ImportError:
            logger.warning("flashrank not installed, returning docs without reranking")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"FlashRank reranking failed: {e}")
            return documents[:top_k]

    async def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        return self._perform_rerank(query, documents, top_k)

    def rerank_sync(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        return self._perform_rerank(query, documents, top_k)


# ── Cross-Encoder Reranker ───────────────────────────────────────────────────


class CrossEncoderReranker(BaseReranker):
    """High-quality reranker using HuggingFace CrossEncoder models."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", **kwargs):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _perform_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not documents:
            return []

        documents = self._deduplicate(documents)
        if len(documents) <= 1:
            return documents[:top_k]

        try:
            model = self._get_model()

            # Prepare pairs: (query, doc_text)
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)

            # Sort documents by score desc
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_provider"] = "cross_encoder"
                reranked.append(doc)

            logger.info(
                f"CrossEncoder reranked {len(documents)} → {len(reranked)} docs "
                f"(top score: {round(scored_docs[0][1], 3) if scored_docs else 0.0})"
            )
            return reranked

        except ImportError:
            logger.warning("sentence-transformers not installed, returning docs without reranking")
            return documents[:top_k]
        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}")
            return documents[:top_k]

    async def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        return self._perform_rerank(query, documents, top_k)

    def rerank_sync(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        return self._perform_rerank(query, documents, top_k)


# ── LLM Reranker ─────────────────────────────────────────────────────────────


class LLMReranker(BaseReranker):
    """Reranker using any standard LangChain LLM (GPT, Claude, etc.)."""

    def __init__(self, llm: Any, max_chars_per_doc: int = 1200, **kwargs):
        self.llm = llm
        self.max_chars_per_doc = max_chars_per_doc

    def _build_prompt(self, query: str, documents: List[Document]) -> str:
        doc_list_str = ""
        for i, doc in enumerate(documents):
            truncated_content = doc.page_content[: self.max_chars_per_doc]
            doc_list_str += f"----- Document [{i}] -----\n{truncated_content}\n\n"

        return f"""You are an advanced information retrieval assistant. Your task is to rank the following candidate documents based on their relevance to the user's query.

Query: {query}

Candidates:
{doc_list_str}
Task: Evaluate the candidates. Rank them from most relevant to least relevant.
Response format: Return ONLY a JSON object containing a "ranking" key, which points to a list of document indices (integers) in order of decreasing relevance.
Do not include any thinking tags, preambles, or markdown format blocks. Just return the JSON object.

Example Output:
{{
  "ranking": [2, 0, 1]
}}
"""

    def _parse_ranking(self, raw: str, num_docs: int) -> List[int]:
        """Parse ranking indices from LLM response."""
        try:
            data = json.loads(raw)
            ranking = data.get("ranking", [])
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[^}]*\"ranking\"\s*:\s*\[([^\]]*)\][^}]*\}", raw)
            if match:
                try:
                    data = json.loads(match.group(0))
                    ranking = data.get("ranking", [])
                except json.JSONDecodeError:
                    ranking = []
            else:
                ranking = []

        seen = set()
        valid = []
        for i in ranking:
            if isinstance(i, int) and 0 <= i < num_docs and i not in seen:
                seen.add(i)
                valid.append(i)

        # Append missing indices to preserve safety
        for i in range(num_docs):
            if i not in seen:
                valid.append(i)

        return valid

    async def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        if not documents:
            return []

        documents = self._deduplicate(documents)
        if len(documents) <= 1:
            return documents[:top_k]

        try:
            prompt = self._build_prompt(query, documents)
            ranking = None
            if hasattr(self.llm, "with_structured_output"):
                try:
                    structured_llm = self.llm.with_structured_output(LLMRerankResult)
                    res = await structured_llm.ainvoke(prompt)
                    raw_ranking = getattr(res, "ranking", None) or (res.get("ranking") if isinstance(res, dict) else None)
                    if raw_ranking is not None:
                        ranking = self._parse_ranking(json.dumps({"ranking": raw_ranking}), len(documents))
                except Exception as e:
                    logger.warning(f"Structured output LLM reranking failed: {e}, falling back to text parsing.")

            if ranking is None:
                res = await self.llm.ainvoke(prompt)
                raw_text = res.content if hasattr(res, "content") else str(res)
                import re
                raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                raw_text = re.sub(r"```json|```", "", raw_text).strip()
                ranking = self._parse_ranking(raw_text, len(documents))

            reranked = []
            for position, idx in enumerate(ranking[:top_k]):
                doc = documents[idx]
                doc.metadata["rerank_score"] = float(1.0 - (position / len(ranking)))
                doc.metadata["rerank_provider"] = "llm"
                reranked.append(doc)

            logger.info(f"LLM reranked {len(documents)} → {len(reranked)} docs.")
            return reranked

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return documents[:top_k]

    def rerank_sync(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        if not documents:
            return []

        documents = self._deduplicate(documents)
        if len(documents) <= 1:
            return documents[:top_k]

        try:
            prompt = self._build_prompt(query, documents)
            ranking = None
            if hasattr(self.llm, "with_structured_output"):
                try:
                    structured_llm = self.llm.with_structured_output(LLMRerankResult)
                    res = structured_llm.invoke(prompt)
                    raw_ranking = getattr(res, "ranking", None) or (res.get("ranking") if isinstance(res, dict) else None)
                    if raw_ranking is not None:
                        ranking = self._parse_ranking(json.dumps({"ranking": raw_ranking}), len(documents))
                except Exception as e:
                    logger.warning(f"Structured output LLM sync reranking failed: {e}, falling back to text parsing.")

            if ranking is None:
                res = self.llm.invoke(prompt)
                raw_text = res.content if hasattr(res, "content") else str(res)
                import re
                raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                raw_text = re.sub(r"```json|```", "", raw_text).strip()
                ranking = self._parse_ranking(raw_text, len(documents))

            reranked = []
            for position, idx in enumerate(ranking[:top_k]):
                doc = documents[idx]
                doc.metadata["rerank_score"] = float(1.0 - (position / len(ranking)))
                doc.metadata["rerank_provider"] = "llm"
                reranked.append(doc)

            logger.info(f"LLM sync reranked {len(documents)} → {len(reranked)} docs.")
            return reranked

        except Exception as e:
            logger.error(f"LLM sync reranking failed: {e}")
            return documents[:top_k]


# ── Factory ──────────────────────────────────────────────────────────────────


class RerankerFactory:
    """Factory for creating reranker instances from config."""

    _registry: Dict[str, type] = {
        "flashrank": FlashRankReranker,
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
    }

    @classmethod
    def create_reranker(
        cls,
        config: Dict[str, Any],
        llm: Any = None,
    ) -> Optional[BaseReranker]:
        """Create a reranker from config dict.

        Args:
            config: The config dict (reads `reranking` section).
            llm: LLM instance for LLM-based reranker.

        Returns:
            BaseReranker instance, or None if reranking is disabled.
        """
        rerank_config = config.get("reranking", {})

        if not rerank_config.get("enabled", False):
            return None

        provider = rerank_config.get("provider", "flashrank").lower()
        provider_config = rerank_config.get(provider, {})

        if provider not in cls._registry:
            logger.warning(f"Unknown reranker provider '{provider}', using flashrank")
            provider = "flashrank"
            provider_config = rerank_config.get(provider, {})

        reranker_cls = cls._registry[provider]

        if provider == "llm":
            return reranker_cls(llm=llm, **provider_config)

        return reranker_cls(**provider_config)

    # Backward compat aliases
    @classmethod
    def create(cls, config: Dict[str, Any], llm: Any = None) -> Optional[BaseReranker]:
        return cls.create_reranker(config, llm=llm)

    @classmethod
    def register(cls, name: str, reranker_cls: type):
        cls._registry[name.lower()] = reranker_cls


async def rerank_documents(
    query: str,
    documents: List[Document],
    config: Dict[str, Any],
    llm: Any = None,
    top_k: Optional[int] = None,
) -> List[Document]:
    """Convenience function to rerank documents asynchronously using config."""
    reranker = RerankerFactory.create_reranker(config, llm=llm)
    if reranker is None:
        return documents

    rerank_config = config.get("reranking", {})
    k = top_k or rerank_config.get("top_k", 5)

    t0 = time.perf_counter()
    result = await reranker.rerank(query, documents, top_k=k)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(f"Reranking completed in {elapsed_ms}ms: {len(documents)} → {len(result)} docs")
    return result


def rerank_documents_sync(
    query: str,
    documents: List[Document],
    config: Dict[str, Any],
    llm: Any = None,
    top_k: Optional[int] = None,
) -> List[Document]:
    """Convenience function to rerank documents synchronously using config."""
    reranker = RerankerFactory.create_reranker(config, llm=llm)
    if reranker is None:
        return documents

    rerank_config = config.get("reranking", {})
    k = top_k or rerank_config.get("top_k", 5)

    t0 = time.perf_counter()
    result = reranker.rerank_sync(query, documents, top_k=k)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(f"Sync reranking completed in {elapsed_ms}ms: {len(documents)} → {len(result)} docs")
    return result
