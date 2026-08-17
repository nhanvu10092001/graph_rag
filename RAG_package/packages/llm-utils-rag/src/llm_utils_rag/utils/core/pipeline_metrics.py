"""Pipeline instrumentation — timing and metrics for RAG operations.

Provides a PipelineTimer context manager for tracking latency at each
pipeline stage, and a MetricsCollector for aggregating query-level stats.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineTimer:
    """Context manager for timing pipeline stages.

    Usage:
        timer = PipelineTimer()
        with timer.stage("extraction"):
            content = extract_text(file)
        with timer.stage("chunking"):
            chunks = chunker.split(docs)

        timer.timings  # {"extraction": 0.34, "chunking": 0.12}
    """

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._start_time: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        """Time a named pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            self.stages[name] = elapsed
            logger.info("pipeline.stage.%s completed in %.1fms", name, elapsed)

    @property
    def total_ms(self) -> float:
        """Total elapsed time since timer creation."""
        return round((time.perf_counter() - self._start_time) * 1000, 1)

    @property
    def timings(self) -> Dict[str, float]:
        """All stage timings in milliseconds."""
        return {**self.stages, "total_ms": self.total_ms}


class MetricsCollector:
    """Collects and exposes aggregate RAG metrics.

    Usage:
        metrics = MetricsCollector()
        metrics.record_query(query, collection, num_docs, latency_ms)
        metrics.record_index(collection, num_chunks, latency_ms)

        stats = metrics.get_stats()
    """

    def __init__(self):
        self._queries: List[Dict[str, Any]] = []
        self._indexes: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []

    def record_query(
        self,
        query: str,
        collection: str,
        num_docs: int,
        latency_ms: float,
        sources: Optional[list] = None,
    ):
        """Record a query execution."""
        entry = {
            "timestamp": time.time(),
            "query_length": len(query),
            "collection": collection,
            "docs_returned": num_docs,
            "latency_ms": latency_ms,
            "source_count": len(sources) if sources else 0,
        }
        self._queries.append(entry)
        logger.info(
            "metrics.query collection=%s docs=%d latency=%.1fms",
            collection, num_docs, latency_ms,
        )

    def record_index(
        self,
        collection: str,
        num_chunks: int,
        latency_ms: float,
        provider: str = "unknown",
    ):
        """Record an indexing execution."""
        entry = {
            "timestamp": time.time(),
            "collection": collection,
            "chunks_indexed": num_chunks,
            "latency_ms": latency_ms,
            "provider": provider,
        }
        self._indexes.append(entry)
        logger.info(
            "metrics.index collection=%s chunks=%d latency=%.1fms provider=%s",
            collection, num_chunks, latency_ms, provider,
        )

    def record_error(self, operation: str, error: str, collection: str = ""):
        """Record an error."""
        self._errors.append({
            "timestamp": time.time(),
            "operation": operation,
            "error": error,
            "collection": collection,
        })
        logger.error(
            "metrics.error operation=%s collection=%s error=%s",
            operation, collection, error,
        )

    def record_rerank(
        self,
        provider: str,
        input_docs: int,
        output_docs: int,
        latency_ms: float,
        scores: Optional[list] = None,
    ):
        """Record a reranking execution."""
        score_stats = {}
        if scores:
            score_stats = {
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                "avg_score": round(sum(scores) / len(scores), 4),
            }
        logger.info(
            "metrics.rerank provider=%s docs=%d→%d latency=%.1fms %s",
            provider, input_docs, output_docs, latency_ms,
            f"scores={score_stats}" if score_stats else "",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        query_latencies = [q["latency_ms"] for q in self._queries]
        index_latencies = [i["latency_ms"] for i in self._indexes]

        return {
            "queries": {
                "total": len(self._queries),
                "avg_latency_ms": round(
                    sum(query_latencies) / len(query_latencies), 1
                ) if query_latencies else 0,
                "max_latency_ms": max(query_latencies) if query_latencies else 0,
                "avg_docs_returned": round(
                    sum(q["docs_returned"] for q in self._queries) / len(self._queries), 1
                ) if self._queries else 0,
            },
            "indexing": {
                "total": len(self._indexes),
                "avg_latency_ms": round(
                    sum(index_latencies) / len(index_latencies), 1
                ) if index_latencies else 0,
                "total_chunks_indexed": sum(
                    i["chunks_indexed"] for i in self._indexes
                ),
            },
            "errors": {
                "total": len(self._errors),
                "recent": self._errors[-5:] if self._errors else [],
            },
        }

    def reset(self):
        """Clear all collected metrics."""
        self._queries.clear()
        self._indexes.clear()
        self._errors.clear()
