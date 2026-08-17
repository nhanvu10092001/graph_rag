"""
Semantic Chunking Pipeline for RAG

This module implements semantic chunking that creates chunks based on meaning similarity
rather than fixed sizes, with category-aware metadata for enhanced retrieval.

Based on LangChain's SemanticChunker with category-aware enhancements.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_experimental.text_splitter import (
    SemanticChunker as LangChainSemanticChunker,
)
from langchain_community.utils.math import (
    cosine_similarity,
)
from pydantic import BaseModel, Field
from enum import Enum
from typing import Tuple


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """Combine sentences based on buffer size.

    Args:
        sentences: List of sentences to combine.
        buffer_size: Number of sentences to combine. Defaults to 1.

    Returns:
        List of sentences with combined sentences.
    """

    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative
            # (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    """Calculate cosine distances between sentences.

    Args:
        sentences: List of sentences to calculate distances for.

    Returns:
        Tuple of distances and sentences.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


class DocumentCategory(str, Enum):
    """Enumeration of document categories."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    GENERAL = "general"


class CategoryDetectionResult(BaseModel):
    """Structured result for category detection."""

    category: DocumentCategory = Field(
        description="The detected category of the document"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 for the category detection",
        ge=0.0,
        le=1.0,
    )


class SemanticChunkerCustom(LangChainSemanticChunker):
    """
    Enhanced semantic chunker based on LangChain's SemanticChunker with category detection.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        enable_category_detection: bool = True,
        llm: Optional[BaseLLM] = None,
        embedding_batch_size: int = 20,
        **kwargs,
    ):
        """
        Initialize the semantic chunker.

        Args:
            embeddings: Embeddings model for computing sentence similarities
            breakpoint_threshold_type: Method for determining breakpoints ("percentile", "standard_deviation", "interquartile")
            breakpoint_threshold_amount: Threshold value for the breakpoint method
            number_of_chunks: If specified, will try to create approximately this many chunks
            enable_category_detection: Whether to detect document categories
            llm: LLM for structured category detection (required if enable_category_detection is True)
            embedding_batch_size: Batch size for embedding sentences (default: 20)
            **kwargs: Additional arguments passed to LangChain SemanticChunker
        """
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
            **kwargs,
        )
        self.enable_category_detection = enable_category_detection
        self.llm = llm
        self.embedding_batch_size = embedding_batch_size
        
        # Add caching for category detection
        self._category_cache = {}

        # Validate LLM is provided if category detection is enabled
        if self.enable_category_detection and self.llm is None:
            raise ValueError(
                "LLM is required when enable_category_detection is True. "
                "Please provide an LLM instance."
            )

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)

        # Embed all combined sentences concurrently using ThreadPoolExecutor
        def embed_single_sentence(text):
            """Embed a single sentence."""
            return self.embeddings.embed_query(text)

        embeddings = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all embedding tasks
            future_to_idx = {
                executor.submit(
                    embed_single_sentence, sentences[i]["combined_sentence"]
                ): i
                for i in range(len(sentences))
            }

            # Collect results in order
            results = [None] * len(sentences)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error embedding sentence {idx}: {e}")
                    # Fallback to zero vector or handle error
                    results[idx] = [0.0] * 768  # Assuming 768-dim embedding

            embeddings = results

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return calculate_cosine_distances(sentences)

    async def _calculate_sentence_distances_async(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components asynchronously."""
        import asyncio

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)

        # Embed combined sentences in batches using asyncio

        async def embed_batch(batch):
            """Embed a single sentence asynchronously."""
            if hasattr(self.embeddings, "aembed_query"):
                return await self.embeddings.aembed_documents(
                    [x["combined_sentence"] for x in batch]
                )
            else:
                return await asyncio.to_thread(
                    self.embeddings.embed_query, [x["combined_sentence"] for x in batch]
                )

        # Split sentences into batches
        batches = [
            sentences[i : i + self.embedding_batch_size]
            for i in range(0, len(sentences), self.embedding_batch_size)
        ]
        tasks = [embed_batch(batch) for batch in batches]
        batch_embeddings = await asyncio.gather(*tasks)

        # Flatten batch results
        all_embeddings = []
        for batch_emb in batch_embeddings:
            all_embeddings.extend(batch_emb)

        # Assign embeddings to sentences
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = all_embeddings[i]

        return calculate_cosine_distances(sentences)

    async def split_text_async(self, text: str) -> List[str]:
        """Split text asynchronously using async embedding."""
        import re

        # Splitting the text
        single_sentences_list = re.split(self.sentence_split_regex, text)

        if len(single_sentences_list) == 1:
            return single_sentences_list
        if (
            self.breakpoint_threshold_type == "gradient"
            and len(single_sentences_list) == 2
        ):
            return single_sentences_list

        # Use async version of calculate_sentence_distances
        distances, sentences = await self._calculate_sentence_distances_async(
            single_sentences_list
        )

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            if (
                self.min_chunk_size is not None
                and len(combined_text) < self.min_chunk_size
            ):
                continue
            chunks.append(combined_text)
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents asynchronously using async embedding."""

        import copy

        all_chunks = []

        for doc in documents:
            # Detect category if enabled
            category = None
            if self.enable_category_detection:
                category = self._detect_category(doc.page_content)

            # Use async text splitting
            chunks_text = await self.split_text_async(doc.page_content)

            # Create Document objects with enhanced metadata
            for i, chunk_text in enumerate(chunks_text):
                enhanced_metadata = copy.deepcopy(doc.metadata)
                enhanced_metadata.update(
                    {
                        "chunk_id": i,
                        "chunk_method": "semantic",
                        "chunk_size": len(chunk_text),
                    }
                )

                if category:
                    enhanced_metadata["category"] = category
                    enhanced_metadata["detection_method"] = getattr(
                        self, "_last_detection_method", "keyword"
                    )

                chunk_doc = Document(
                    page_content=chunk_text, metadata=enhanced_metadata
                )
                all_chunks.append(chunk_doc)

        return all_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using semantic chunking with category detection.

        Args:
            documents: List of documents to split

        Returns:
            List of Document objects representing semantic chunks
        """
        all_chunks = []

        for doc in documents:
            # Detect category if enabled
            category = None
            if self.enable_category_detection:
                category = self._detect_category(doc.page_content)

            # Use LangChain's semantic chunking
            chunks = LangChainSemanticChunker.split_documents(self, [doc])

            # Enhance chunks with category information
            for i, chunk in enumerate(chunks):
                enhanced_metadata = chunk.metadata.copy()
                enhanced_metadata.update(
                    {
                        "chunk_id": i,
                        "chunk_method": "semantic",
                        "chunk_size": len(chunk.page_content),
                    }
                )

                if category:
                    enhanced_metadata["category"] = category
                    enhanced_metadata["detection_method"] = getattr(
                        self, "_last_detection_method", "keyword"
                    )
                chunk.metadata = enhanced_metadata
                all_chunks.append(chunk)

        return all_chunks

    def _detect_category(self, text: str) -> Optional[str]:
        """
        Detect document category using structured LLM with caching.
        """
        # Create cache key from first 200 chars to avoid duplicate detection
        cache_key = hash(text[:200])
        
        # Check cache first
        if cache_key in self._category_cache:
            self._last_detection_method = "cached"
            return self._category_cache[cache_key]
        
        try:
            result = self._detect_category_with_llm(text)
            self._last_detection_method = "llm"
            
            # Cache the result
            self._category_cache[cache_key] = result
            return result
        except Exception as e:
            raise Exception(f"Category detection failed: {e}")

    def _detect_category_with_llm(self, text: str) -> str:
        """
        Detect document category using optimized structured LLM.
        """
        # Truncate text more aggressively for speed (keep first 300 characters)
        text_sample = text[:300] if len(text) > 300 else text

        try:
            # Create structured LLM with the schema
            structured_llm = self.llm.with_structured_output(CategoryDetectionResult)

            # Simplified, shorter prompt for faster processing
            prompt = f"""Categorize this text quickly:

{text_sample}

Choose: technical, business, scientific, legal, medical, educational, or general.
Return category and confidence only."""

            result = structured_llm.invoke(prompt)
            return result.category.value
        except NotImplementedError:
            # For models like OllamaLLM that don't support with_structured_output
            return "general"
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM category detection failed: {e}. Falling back to 'general'.")
            return "general"


