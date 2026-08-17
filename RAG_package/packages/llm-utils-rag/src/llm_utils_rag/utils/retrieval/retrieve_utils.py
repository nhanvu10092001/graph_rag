"""
Semantic RAG Chain for Category-Aware Retrieval

This module implements enhanced RAG chains that leverage semantic chunking metadata
and category information for improved retrieval and generation.
"""

from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from enum import Enum

from ...logger import logger


class QueryCategory(str, Enum):
    """Enumeration of query categories."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    GENERAL = "general"


class CategoryDetectionResult(BaseModel):
    """Structured result for category detection."""

    category: QueryCategory = Field(
        description="The detected category of the query"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 for the category detection",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for the category choice")


class CategoryAwareRetriever:
    """
    Enhanced retriever that uses category information to improve document retrieval.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        category_weights: Optional[Dict[str, float]] = None,
        enable_category_filtering: bool = True,
        enable_semantic_reranking: bool = True,
        llm: Optional[BaseLanguageModel] = None,
    ):
        """
        Initialize category-aware retriever.

        Args:
            base_retriever: Base vector store retriever
            category_weights: Weights for different categories during retrieval
            enable_category_filtering: Whether to filter by detected query category
            enable_semantic_reranking: Whether to rerank based on semantic metadata
            llm: Language model for category detection (optional, falls back to keyword matching)
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.category_weights = category_weights or {
            "technical": 1.2,
            "business": 1.0,
            "scientific": 1.1,
            "legal": 1.3,
            "medical": 1.1,
            "educational": 1.0,
            "general": 0.9,
        }
        self.enable_category_filtering = enable_category_filtering
        self.enable_semantic_reranking = enable_semantic_reranking

        # Cache for category detection to avoid repeated LLM calls
        self._category_cache = {}

    async def detect_query_category(self, query: str) -> str:
        """
        Detect the category of a query using LLM.
        """
        logger.info(f"Detecting category for query: {query[:100]}...")
        # Check cache first
        if query in self._category_cache:
            logger.info("Category found in cache")
            return self._category_cache[query]

        # Require LLM for category detection
        if self.llm is None:
            logger.error("LLM is required for category detection but not provided")
            raise ValueError(
                "LLM is required for category detection. Please provide an LLM instance."
            )

        # Use LLM for category detection
        category = await self._detect_category_with_llm(query)
        self._category_cache[query] = category
        logger.info(f"Detected category: {category}")
        return category

    async def _detect_category_with_llm(self, query: str) -> str:
        """Use LLM to detect query category, with fallback for unsupported providers."""

        prompt = f"""Analyze the following query and categorize it into exactly one of these categories:
technical, business, scientific, legal, medical, educational, general

Query: {query}

Respond with ONLY the category name (one word, lowercase)."""

        # Try structured output first (supported by OpenAI, Anthropic, Google, etc.)
        try:
            structured_llm = self.llm.with_structured_output(CategoryDetectionResult)
            result = await structured_llm.ainvoke(prompt)
            return result.category.value
        except (NotImplementedError, AttributeError):
            logger.info("with_structured_output not supported, falling back to plain text")
        except Exception as e:
            logger.warning(f"Structured output failed: {e}, falling back to plain text")

        # Fallback: plain text response + parse
        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip().lower()

            valid_categories = {c.value for c in QueryCategory}
            # Check if the response is directly a valid category
            if text in valid_categories:
                return text
            # Try to find a category keyword in the response
            for cat in valid_categories:
                if cat in text:
                    return cat
            logger.warning(f"Could not parse category from LLM response: '{text}', defaulting to 'general'")
            return "general"
        except Exception as e:
            logger.error(f"LLM category detection failed entirely: {e}")
            return "general"

    async def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents with category-aware filtering and reranking.
        """
        logger.info(f"Retrieving documents for query: {query[:100]}... (k={k})")
        # Detect query category
        query_category = await self.detect_query_category(query)
        logger.info(f"Query category: {query_category}")

        # Get base retrieval results (request more than needed for reranking)
        retrieval_k = min(k * 3, 20)  # Get 3x more for reranking
        logger.info(f"Requesting {retrieval_k} documents from base retriever")

        try:
            # Check if retriever supports k parameter
            if hasattr(self.base_retriever, "search_kwargs"):
                # Update k parameter for retrievers that support it
                original_k = self.base_retriever.search_kwargs.get("k")
                self.base_retriever.search_kwargs["k"] = retrieval_k
                base_docs = await self.base_retriever.ainvoke(query)
                # Restore original k
                if original_k is not None:
                    self.base_retriever.search_kwargs["k"] = original_k
                else:
                    self.base_retriever.search_kwargs.pop("k", None)
            else:
                base_docs = await self.base_retriever.ainvoke(query)
        except Exception as e:
            logger.warning(f"Base retriever failed: {e}")
            # If SelfQueryRetriever fails, try fallback to similarity search
            if hasattr(self.base_retriever, "vectorstore"):
                logger.info("Attempting fallback to direct vectorstore similarity search")
                try:
                    base_docs = (
                        await self.base_retriever.vectorstore.asimilarity_search(
                            query, k=retrieval_k
                        )
                    )
                    logger.info(f"Fallback successful, retrieved {len(base_docs)} documents")
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {fallback_e}")
                    return []
            else:
                logger.error("No fallback available, returning empty results")
                return []

        if not base_docs:
            logger.warning("No documents retrieved")
            return []

        logger.info(f"Retrieved {len(base_docs)} base documents")

        # Apply category-aware scoring
        if self.enable_category_filtering or self.enable_semantic_reranking:
            logger.info("Applying category-aware scoring and reranking")
            scored_docs = []

            for doc in base_docs:
                score = 1.0  # Base score
                metadata = doc.metadata

                # Category-based scoring
                doc_category = metadata.get("category", "general")
                if query_category == doc_category:
                    score *= self.category_weights.get(doc_category, 1.0)
                elif doc_category in self.category_weights:
                    # Apply reduced weight for different categories
                    score *= self.category_weights[doc_category] * 0.7

                # Semantic chunk quality scoring
                if self.enable_semantic_reranking:
                    chunk_method = metadata.get("chunk_method", "unknown")
                    if chunk_method == "semantic":
                        similarity_score = metadata.get("similarity_score", 0.5)
                        score *= (
                            1.0 + similarity_score * 0.3
                        )  # Boost for high-quality semantic chunks

                    # Boost for category confidence
                    category_confidence = metadata.get("category_confidence", 0.5)
                    score *= 1.0 + category_confidence * 0.2

                scored_docs.append((doc, score))

            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Reranking complete, returning top {k} documents")
            return [doc for doc, _ in scored_docs[:k]]

        logger.info(f"Returning top {k} documents without reranking")
        return base_docs[:k]


class SemanticRAGChain:
    """
    Enhanced RAG chain that leverages semantic chunking metadata and category awareness.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[BaseLanguageModel] = None,
        enable_category_aware_prompts: bool = True,
        enable_context_enhancement: bool = True,
        enable_llm_generation: bool = False,
    ):
        """
        Initialize semantic RAG chain.

        Args:
            retriever: Document retriever (will be wrapped in CategoryAwareRetriever)
            llm: Language model for generation (optional, only needed if enable_llm_generation=True)
            enable_category_aware_prompts: Use category-specific prompts
            enable_context_enhancement: Enhance context with metadata
            enable_llm_generation: Whether to generate answers using LLM (default: False)
        """
        self.category_aware_retriever = CategoryAwareRetriever(retriever, llm=llm)
        self.llm = llm
        self.enable_category_aware_prompts = enable_category_aware_prompts
        self.enable_context_enhancement = enable_context_enhancement
        self.enable_llm_generation = enable_llm_generation

        # Category-specific prompt templates
        self.category_prompts = {
            "technical": """You are a technical expert assistant. Use the provided technical documentation to answer the question accurately and precisely.

            Context from technical documentation:
            {context}

            Question: {question}

            Provide a detailed technical answer with specific implementation details, code examples if relevant, and step-by-step instructions where appropriate. Focus on accuracy and practical application.""",
            "business": """You are a business analyst assistant. Use the provided business information to answer the question with strategic insights and actionable recommendations.

            Context from business documents:
            {context}

            Question: {question}

            Provide a comprehensive business-focused answer with relevant metrics, strategic implications, and practical recommendations. Consider market impact and business value.""",
            "scientific": """You are a research assistant with expertise in scientific analysis. Use the provided research data to answer the question with scientific rigor.

            Context from scientific literature:
            {context}

            Question: {question}

            Provide a scientifically accurate answer with proper methodology, data interpretation, and evidence-based conclusions. Include statistical significance where relevant.""",
            "legal": """You are a legal research assistant. Use the provided legal documents to answer the question with precision and attention to legal nuances.

            Context from legal documents:
            {context}

            Question: {question}

            Provide a legally informed answer with attention to specific terms, conditions, and implications. Note any jurisdictional considerations and cite relevant provisions.""",
            "medical": """You are a medical information assistant. Use the provided medical literature to answer the question with clinical accuracy.

            Context from medical sources:
            {context}

            Question: {question}

            Provide a medically accurate answer based on the provided context. Focus on evidence-based information and clinical relevance. Note: This is for informational purposes only.""",
            "general": """You are a helpful assistant. Use the provided context to answer the question accurately and comprehensively.

            Context:
            {context}

            Question: {question}

            Provide a clear, accurate, and helpful answer based on the provided context. Be comprehensive yet concise.""",
        }

    def enhance_context(self, documents: List[Document]) -> str:
        """
        Enhance context by organizing documents by category and including metadata.
        """
        if not self.enable_context_enhancement:
            return "\n\n".join(doc.page_content for doc in documents)

        # Group documents by category
        categorized_docs = {}
        for doc in documents:
            category = doc.metadata.get("category", "general")
            if category not in categorized_docs:
                categorized_docs[category] = []
            categorized_docs[category].append(doc)

        # Build enhanced context
        context_parts = []

        for category, docs in categorized_docs.items():
            if (
                len(categorized_docs) > 1
            ):  # Only add category headers if multiple categories
                context_parts.append(f"=== {category.upper()} DOCUMENTS ===")

            for i, doc in enumerate(docs):
                metadata = doc.metadata

                # Add document header with metadata
                doc_info = []
                if metadata.get("filename"):
                    doc_info.append(f"Source: {metadata['filename']}")
                if metadata.get("chunk_method") == "semantic":
                    similarity = metadata.get("similarity_score", 0)
                    doc_info.append(f"Quality: {similarity:.2f}")

                if doc_info:
                    context_parts.append(f"[{' | '.join(doc_info)}]")

                context_parts.append(doc.page_content)
                context_parts.append("")  # Add spacing

        return "\n".join(context_parts)

    async def get_category_prompt(self, query: str, context: str) -> ChatPromptTemplate:
        """
        Get category-appropriate prompt template.
        """
        if not self.enable_category_aware_prompts:
            template = self.category_prompts["general"]
        else:
            query_category = await self.category_aware_retriever.detect_query_category(
                query
            )
            template = self.category_prompts.get(
                query_category, self.category_prompts["general"]
            )

        return ChatPromptTemplate.from_template(template)

    def create_chain(self):
        """
        Create the complete semantic RAG chain.
        """

        async def retrieve_and_enhance_context(input_dict):
            """Retrieve documents and enhance context."""
            query = input_dict["question"]

            # Retrieve documents using category-aware retrieval
            documents = await self.category_aware_retriever.retrieve_documents(query)

            # Enhance context with metadata
            enhanced_context = self.enhance_context(documents)

            return {
                "context": enhanced_context,
                "question": query,
                "documents": documents,
                "retrieved_count": len(documents),
            }

        async def apply_category_prompt(input_dict):
            """Apply category-specific prompt or return documents only."""
            query = input_dict["question"]
            context = input_dict["context"]

            result = {
                **input_dict,
                "category": await self.category_aware_retriever.detect_query_category(
                    query
                ),
            }

            # Only generate answer if LLM generation is enabled
            if self.enable_llm_generation and self.llm is not None:
                # Get category-appropriate prompt
                prompt = await self.get_category_prompt(query, context)

                # Generate response
                response = await self.llm.ainvoke(
                    prompt.format(context=context, question=query)
                )
                result["answer"] = (
                    response.content if hasattr(response, "content") else str(response)
                )

            return result

        # Create the chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(retrieve_and_enhance_context)
            | RunnableLambda(apply_category_prompt)
        )

        return chain


def create_semantic_rag_chain(
    retriever: BaseRetriever, llm: BaseLanguageModel, **kwargs
) -> SemanticRAGChain:
    """
    Factory function to create semantic RAG chains.

    Args:
        retriever: Document retriever
        llm: Language model
        **kwargs: Additional configuration options

    Returns:
        SemanticRAGChain instance
    """
    return SemanticRAGChain(retriever, llm, **kwargs)
