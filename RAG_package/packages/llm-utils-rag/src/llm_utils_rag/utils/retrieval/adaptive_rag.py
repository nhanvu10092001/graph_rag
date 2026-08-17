"""Adaptive RAG — self-correcting retrieval with document grading and query rewriting.

Implements a feedback loop that:
1. Retrieves documents
2. Grades their relevance
3. If not relevant → rewrites the query and re-retrieves
4. Grades the final answer for hallucination

This significantly improves answer quality at the cost of ~2-3x latency.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentGradingResult(BaseModel):
    is_relevant: bool = Field(description="True if document contains information relevant to the question, False otherwise")


class HallucinationCheckResult(BaseModel):
    grounded: bool = Field(description="True if answer is supported by context, False otherwise")
    confidence: float = Field(default=0.5, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(default="", description="Brief explanation")


class DocumentGrader:
    """Grades whether a retrieved document is relevant to the query."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def grade(self, query: str, document: Document) -> bool:
        """Grade a single document for relevance.

        Returns:
            True if the document is relevant to the query.
        """
        prompt = f"""You are a relevance grader. Given a user question and a retrieved document,
determine if the document contains information relevant to answering the question.

Question: {query}
Document: {document.page_content[:1500]}"""

        if hasattr(self.llm, "with_structured_output"):
            try:
                structured_llm = self.llm.with_structured_output(DocumentGradingResult)
                res = await structured_llm.ainvoke(prompt)
                return getattr(res, "is_relevant", True) if not isinstance(res, dict) else res.get("is_relevant", True)
            except Exception as e:
                logger.warning("Structured document grading failed: %s, using fallback", e)

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = text.strip().lower()
            return "yes" in text
        except Exception as e:
            logger.warning("Document grading failed: %s, assuming relevant", e)
            return True  # Fail-open: assume relevant if grading fails

    async def grade_batch(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """Grade multiple documents, returning only relevant ones."""
        if not documents:
            return []

        relevant = []
        for doc in documents:
            is_relevant = await self.grade(query, doc)
            if is_relevant:
                doc.metadata["relevance_grade"] = "relevant"
                relevant.append(doc)
            else:
                logger.debug(
                    "Filtered irrelevant doc: %s", doc.page_content[:80]
                )

        logger.info(
            "Document grading: %d/%d docs passed relevance check",
            len(relevant),
            len(documents),
        )
        return relevant


class QueryRewriter:
    """Rewrites a query to improve retrieval when initial results are poor."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def rewrite(self, query: str, failed_docs: List[Document] = None) -> str:
        """Rewrite the query for better retrieval.

        Args:
            query: Original query that produced poor results.
            failed_docs: Documents that were graded as irrelevant (for context).

        Returns:
            Rewritten query.
        """
        context = ""
        if failed_docs:
            context = f"""
The following documents were retrieved but found irrelevant:
{chr(10).join(d.page_content[:200] for d in failed_docs[:3])}

Rewrite the query to avoid these irrelevant results and better target the needed information."""

        prompt = f"""You are a query optimizer. The following search query did not produce relevant results.
Rewrite it to be more specific and likely to find relevant documents.
{context}

Original query: {query}

Respond with ONLY the rewritten query, nothing else."""

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            rewritten = text.strip().strip('"').strip("'")

            if rewritten and rewritten != query:
                logger.info("Query rewritten: '%s' → '%s'", query[:50], rewritten[:50])
                return rewritten
        except Exception as e:
            logger.warning("Query rewriting failed: %s", e)

        return query


class HallucinationChecker:
    """Checks if the generated answer is grounded in the retrieved documents."""

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

    async def check(
        self, answer: str, documents: List[Document]
    ) -> Dict[str, Any]:
        """Check if the answer is supported by the documents.

        Returns:
            Dict with 'is_grounded' (bool), 'confidence' (float), 'reasoning' (str)
        """
        context = "\n\n".join(d.page_content[:800] for d in documents[:5])

        prompt = f"""You are a fact-checker. Determine if the following answer is supported by the provided context.

Context:
{context}

Answer: {answer}"""

        if hasattr(self.llm, "with_structured_output"):
            try:
                structured_llm = self.llm.with_structured_output(HallucinationCheckResult)
                res = await structured_llm.ainvoke(prompt)
                grounded = getattr(res, "grounded", True) if not isinstance(res, dict) else res.get("grounded", True)
                confidence = getattr(res, "confidence", 0.5) if not isinstance(res, dict) else res.get("confidence", 0.5)
                reasoning = getattr(res, "reasoning", "") if not isinstance(res, dict) else res.get("reasoning", "")
                return {
                    "is_grounded": grounded,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
            except Exception as e:
                logger.warning("Structured hallucination check failed: %s, using fallback", e)

        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

            # Try parse JSON
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)

            result = json.loads(text.strip())
            return {
                "is_grounded": result.get("grounded", True),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            logger.warning("Hallucination check failed: %s", e)
            return {"is_grounded": True, "confidence": 0.5, "reasoning": "Check failed"}


class AdaptiveRAGPipeline:
    """Self-correcting RAG pipeline with document grading and query rewriting.

    Flow:
        Query → Retrieve → Grade Docs → ┐
                                          ├─ Relevant? → Return docs
                                          └─ Not relevant? → Rewrite → Re-retrieve (max N retries)

    Usage:
        pipeline = AdaptiveRAGPipeline(
            retriever=my_retriever,
            llm=my_llm,
            max_retries=2,
            relevance_threshold=0.5,
        )
        docs = await pipeline.retrieve(query)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        max_retries: int = 2,
        relevance_threshold: float = 0.5,
        enable_hallucination_check: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_retries = max_retries
        self.relevance_threshold = relevance_threshold
        self.enable_hallucination_check = enable_hallucination_check

        self._grader = DocumentGrader(llm)
        self._rewriter = QueryRewriter(llm)
        self._hallucination_checker = HallucinationChecker(llm)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Execute adaptive retrieval with self-correction.

        Returns:
            Dict with 'documents', 'query_history', 'retries', 'metadata'.
        """
        t0 = time.perf_counter()
        query_history = [query]
        current_query = query
        all_filtered_docs = []

        for attempt in range(1 + self.max_retries):
            # Step 1: Retrieve
            try:
                raw_docs = await self.retriever.ainvoke(current_query)
            except Exception as e:
                logger.error("Retrieval failed on attempt %d: %s", attempt + 1, e)
                raw_docs = []

            if not raw_docs:
                logger.warning(
                    "No documents retrieved on attempt %d", attempt + 1
                )
                if attempt < self.max_retries:
                    current_query = await self._rewriter.rewrite(
                        current_query, all_filtered_docs
                    )
                    query_history.append(current_query)
                    continue
                break

            # Step 2: Grade documents
            relevant_docs = await self._grader.grade_batch(current_query, raw_docs)
            irrelevant_docs = [d for d in raw_docs if d not in relevant_docs]
            all_filtered_docs.extend(irrelevant_docs)

            relevance_ratio = len(relevant_docs) / len(raw_docs) if raw_docs else 0

            logger.info(
                "Attempt %d: %d/%d docs relevant (ratio=%.2f, threshold=%.2f)",
                attempt + 1,
                len(relevant_docs),
                len(raw_docs),
                relevance_ratio,
                self.relevance_threshold,
            )

            # Step 3: Check if we have enough relevant docs
            if relevance_ratio >= self.relevance_threshold and relevant_docs:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                return {
                    "documents": relevant_docs[:k],
                    "query_history": query_history,
                    "retries": attempt,
                    "metadata": {
                        "total_retrieved": len(raw_docs),
                        "total_relevant": len(relevant_docs),
                        "relevance_ratio": round(relevance_ratio, 3),
                        "total_time_ms": elapsed_ms,
                        "final_query": current_query,
                        "adaptive": True,
                    },
                }

            # Step 4: Rewrite and retry
            if attempt < self.max_retries:
                current_query = await self._rewriter.rewrite(
                    current_query, irrelevant_docs
                )
                query_history.append(current_query)
                logger.info(
                    "Retrying with rewritten query (attempt %d/%d)",
                    attempt + 2,
                    1 + self.max_retries,
                )

        # Exhausted retries — return best available
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Gather whatever relevant docs we found across attempts
        final_docs = relevant_docs[:k] if relevant_docs else raw_docs[:k]

        return {
            "documents": final_docs,
            "query_history": query_history,
            "retries": self.max_retries,
            "metadata": {
                "total_retrieved": len(raw_docs) if raw_docs else 0,
                "total_relevant": len(relevant_docs) if relevant_docs else 0,
                "relevance_ratio": round(
                    len(relevant_docs) / max(len(raw_docs), 1), 3
                )
                if relevant_docs
                else 0,
                "total_time_ms": elapsed_ms,
                "final_query": current_query,
                "adaptive": True,
                "exhausted_retries": True,
            },
        }

    async def check_answer(
        self,
        answer: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """Check if a generated answer is grounded in documents.

        Returns hallucination check result.
        """
        if not self.enable_hallucination_check:
            return {"is_grounded": True, "confidence": 1.0, "reasoning": "Check disabled"}

        return await self._hallucination_checker.check(answer, documents)
