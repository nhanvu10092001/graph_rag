from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import asyncio
from dataclasses import dataclass

# RAGAS imports with fallback
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


@dataclass
class RAGEvaluationResult:
    """Container for RAG evaluation results."""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    context_precision: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None


class SemanticRAGMetrics:
    """
    Metrics and evaluation for semantic RAG performance.
    """
    
    def __init__(self, enable_ragas: bool = True):
        """
        Initialize semantic RAG metrics.
        
        Args:
            enable_ragas: Whether to enable RAGAS evaluation metrics
        """
        self.query_categories = {}
        self.retrieval_stats = {}
        self.response_quality = {}
        self.ragas_results = []
        self.enable_ragas = enable_ragas and RAGAS_AVAILABLE
        
        if self.enable_ragas:
            print("✅ RAGAS metrics enabled")
        elif enable_ragas and not RAGAS_AVAILABLE:
            print("⚠️  RAGAS not available. Install with: pip install ragas")
    
    def log_query(self, query: str, category: str, retrieved_docs: List[Document], response: str, ground_truth: Optional[str] = None):
        """
        Log query metrics for analysis.
        
        Args:
            query: The input query
            category: Expected category of the query
            retrieved_docs: Documents retrieved by the RAG system
            response: Generated response from the RAG system
            ground_truth: Optional ground truth answer for RAGAS evaluation
        """
        self.query_categories[query] = category
        
        # Analyze retrieved documents
        doc_categories = [doc.metadata.get("category", "unknown") for doc in retrieved_docs]
        category_match_rate = sum(1 for cat in doc_categories if cat == category) / len(doc_categories) if doc_categories else 0
        
        semantic_chunks = sum(1 for doc in retrieved_docs if doc.metadata.get("chunk_method") == "semantic")
        semantic_rate = semantic_chunks / len(retrieved_docs) if retrieved_docs else 0
        
        # Extract context strings from retrieved documents
        contexts = [doc.page_content for doc in retrieved_docs]
        
        self.retrieval_stats[query] = {
            "category_match_rate": category_match_rate,
            "semantic_chunk_rate": semantic_rate,
            "total_docs": len(retrieved_docs),
            "avg_similarity": sum(doc.metadata.get("similarity_score", 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        }
        
        # Store data for RAGAS evaluation if enabled
        if self.enable_ragas:
            rag_result = RAGEvaluationResult(
                query=query,
                answer=response,
                contexts=contexts,
                ground_truth=ground_truth
            )
            self.ragas_results.append(rag_result)
    
    async def evaluate_with_ragas(self, llm=None, embeddings=None) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            llm: Optional LLM for RAGAS evaluation (uses default if None)
            embeddings: Optional embeddings for RAGAS evaluation (uses default if None)
            
        Returns:
            Dictionary with RAGAS metric scores
        """
        if not self.enable_ragas:
            return {"error": "RAGAS not enabled or available"}
        
        if not self.ragas_results:
            return {"error": "No RAGAS evaluation data available"}
        
        # Filter results that have ground truth for recall calculation
        results_with_ground_truth = [r for r in self.ragas_results if r.ground_truth is not None]
        all_results = self.ragas_results
        
        try:
            ragas_scores = {}
            
            # Evaluate metrics that don't require ground truth
            if all_results:
                no_gt_data = [
                    {
                        "question": r.query,
                        "answer": r.answer,
                        "contexts": r.contexts
                    }
                    for r in all_results
                ]
                
                no_gt_dataset = Dataset.from_list(no_gt_data)
                
                # Evaluate context_precision, faithfulness, and answer_relevancy
                no_gt_metrics = [context_precision, faithfulness, answer_relevancy]
                no_gt_result = evaluate(
                    dataset=no_gt_dataset,
                    metrics=no_gt_metrics,
                    llm=llm,
                    embeddings=embeddings
                )
                ragas_scores.update(no_gt_result)
            
            # Evaluate context_recall (requires ground truth)
            if results_with_ground_truth:
                gt_data = [
                    {
                        "question": r.query,
                        "answer": r.answer,
                        "contexts": r.contexts,
                        "ground_truth": r.ground_truth
                    }
                    for r in results_with_ground_truth
                ]
                
                gt_dataset = Dataset.from_list(gt_data)
                
                # Evaluate context_recall
                recall_result = evaluate(
                    dataset=gt_dataset,
                    metrics=[context_recall],
                    llm=llm,
                    embeddings=embeddings
                )
                ragas_scores.update(recall_result)
            
            # Update individual results with scores
            for i, result in enumerate(self.ragas_results):
                if i < len(all_results):
                    result.context_precision = ragas_scores.get("context_precision", None)
                    result.faithfulness = ragas_scores.get("faithfulness", None)
                    result.answer_relevancy = ragas_scores.get("answer_relevancy", None)
                    
                    if result.ground_truth and i < len(results_with_ground_truth):
                        result.context_recall = ragas_scores.get("context_recall", None)
            
            return ragas_scores
            
        except Exception as e:
            return {"error": f"RAGAS evaluation failed: {str(e)}"}
    
    def get_ragas_summary(self) -> Dict[str, Any]:
        """Get RAGAS evaluation summary."""
        if not self.enable_ragas or not self.ragas_results:
            return {"message": "No RAGAS evaluation data available"}
        
        total_evaluations = len(self.ragas_results)
        with_ground_truth = len([r for r in self.ragas_results if r.ground_truth is not None])
        
        # Calculate average scores where available
        precision_scores = [r.context_precision for r in self.ragas_results if r.context_precision is not None]
        faithfulness_scores = [r.faithfulness for r in self.ragas_results if r.faithfulness is not None]
        relevancy_scores = [r.answer_relevancy for r in self.ragas_results if r.answer_relevancy is not None]
        recall_scores = [r.context_recall for r in self.ragas_results if r.context_recall is not None]
        
        summary = {
            "total_evaluations": total_evaluations,
            "evaluations_with_ground_truth": with_ground_truth,
            "ragas_metrics": {}
        }
        
        if precision_scores:
            summary["ragas_metrics"]["avg_context_precision"] = sum(precision_scores) / len(precision_scores)
        if faithfulness_scores:
            summary["ragas_metrics"]["avg_faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
        if relevancy_scores:
            summary["ragas_metrics"]["avg_answer_relevancy"] = sum(relevancy_scores) / len(relevancy_scores)
        if recall_scores:
            summary["ragas_metrics"]["avg_context_recall"] = sum(recall_scores) / len(recall_scores)
        
        return summary

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including RAGAS metrics."""
        if not self.retrieval_stats:
            return {"message": "No queries logged yet"}
        
        total_queries = len(self.retrieval_stats)
        avg_category_match = sum(stats["category_match_rate"] for stats in self.retrieval_stats.values()) / total_queries
        avg_semantic_rate = sum(stats["semantic_chunk_rate"] for stats in self.retrieval_stats.values()) / total_queries
        avg_similarity = sum(stats["avg_similarity"] for stats in self.retrieval_stats.values()) / total_queries
        
        summary = {
            "total_queries": total_queries,
            "avg_category_match_rate": avg_category_match,
            "avg_semantic_chunk_rate": avg_semantic_rate,
            "avg_similarity_score": avg_similarity,
            "query_categories": self.query_categories,
            "ragas_enabled": self.enable_ragas
        }
        
        # Add RAGAS summary if available
        if self.enable_ragas:
            ragas_summary = self.get_ragas_summary()
            summary["ragas_evaluation"] = ragas_summary
        
        return summary

    def export_ragas_data(self) -> List[Dict[str, Any]]:
        """Export RAGAS evaluation data for external analysis."""
        if not self.enable_ragas:
            return []
        
        return [
            {
                "question": r.query,
                "answer": r.answer,
                "contexts": r.contexts,
                "ground_truth": r.ground_truth,
                "context_precision": r.context_precision,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_recall": r.context_recall
            }
            for r in self.ragas_results
        ]

    def clear_results(self):
        """Clear all logged results."""
        self.query_categories.clear()
        self.retrieval_stats.clear()
        self.response_quality.clear()
        self.ragas_results.clear()