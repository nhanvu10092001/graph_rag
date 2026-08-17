from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import PrivateAttr


class LLMListwiseRerankCompressor(BaseDocumentCompressor):
    """Listwise reranker using any LLM (no structured-output requirement)."""

    top_n: int = 3
    max_chars_per_doc: int = 1200
    prompt: Optional[PromptTemplate] = None

    _llm: Any = PrivateAttr()
    _parser: StrOutputParser = PrivateAttr(default_factory=StrOutputParser)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: Any, **data: Any):
        """llm must support .invoke(str) -> str (or object with .content)."""

        default_prompt = PromptTemplate(
            template="""
                You are a ranking model. Rank the following documents by relevance to the query.
                Return ONLY JSON with the schema: {"ranking":[<int index starting at 0>...]}.

                Query: {{ query }}

                Documents:
                {% for i, d in docs %}
                [{{ i }}] {{ d }}
                {% endfor %}

                Return JSON only.""",
            input_variables=["query", "docs"],
            template_format="jinja2",
        )
        if "prompt" not in data or data["prompt"] is None:
            data["prompt"] = default_prompt

        super().__init__(**data)
        self._llm = llm

    def _rank_indices(self, docs: Sequence[Document], query: str) -> List[int]:
        rendered = self.prompt.format(
            query=query,
            docs=[
                (i, d.page_content[: self.max_chars_per_doc])
                for i, d in enumerate(docs)
            ],
        )
        raw = self._llm.invoke(rendered)

        if isinstance(raw, dict) and "content" in raw:
            raw = raw["content"]
        elif hasattr(raw, "content"):
            raw = raw.content

        try:
            data = json.loads(raw)
            ranking = data.get("ranking", [])
            seen = set()
            return [
                i
                for i in ranking
                if isinstance(i, int)
                and 0 <= i < len(docs)
                and (i not in seen and not seen.add(i))
            ]
        except Exception:
            return list(range(len(docs)))

    def compress_documents(
        self, docs: Sequence[Document], query: str, **kwargs: Any
    ) -> List[Document]:
        if not docs:
            return []
        order = self._rank_indices(docs, query)
        ordered = [docs[i] for i in order]
        return ordered[: self.top_n]

    async def acompress_documents(
        self, docs: Sequence[Document], query: str, **kwargs: Any
    ) -> List[Document]:
        return self.compress_documents(docs, query, **kwargs)
