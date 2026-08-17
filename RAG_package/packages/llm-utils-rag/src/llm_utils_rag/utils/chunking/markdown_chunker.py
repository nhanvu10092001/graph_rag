"""Markdown header-aware chunker.

Splits Markdown documents by header hierarchy first, then applies
recursive splitting on large sections. Preserves header context in metadata.
"""

import copy
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)

# Default header hierarchy for splitting
DEFAULT_HEADERS = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]


class MarkdownChunkerWrapper(BaseChunker):
    """Structure-aware chunker for Markdown documents.

    Splits by headers first, then applies recursive splitting on large sections.

    Usage:
        chunker = MarkdownChunkerWrapper(chunk_size=1000)
        chunks = chunker.split_documents(docs)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_headers: bool = False,
        headers_to_split_on: Optional[list] = None,
    ):
        self._chunk_size = chunk_size
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on or DEFAULT_HEADERS,
            strip_headers=strip_headers,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split markdown documents by headers, then by size."""
        all_chunks = []

        for doc in documents:
            # Step 1: Split by headers
            header_splits = self._header_splitter.split_text(doc.page_content)

            # Step 2: Further split large sections
            for header_doc in header_splits:
                if len(header_doc.page_content) > self._chunk_size:
                    sub_chunks = self._recursive_splitter.split_documents([header_doc])
                else:
                    sub_chunks = [header_doc]

                # Build header breadcrumb for context
                header_path = " > ".join(
                    v for k, v in sorted(header_doc.metadata.items())
                    if k.startswith("header_") and v
                )

                for chunk in sub_chunks:
                    extra = {}
                    if header_path:
                        extra["header_path"] = header_path
                    # Merge header metadata (header_1, header_2, etc.)
                    extra.update({
                        k: v for k, v in header_doc.metadata.items()
                        if k.startswith("header_")
                    })

                    self._enrich_metadata(
                        chunk, index=len(all_chunks), method="markdown",
                        source_metadata=doc.metadata,
                        extra=extra,
                    )
                    all_chunks.append(chunk)

        return all_chunks

    async def asplit_documents(self, documents: List[Document]) -> List[Document]:
        """Async version — markdown splitting is CPU-bound, so same as sync."""
        return self.split_documents(documents)
