"""KnowledgeBaseSearchTool — Search internal knowledge base via Elasticsearch.

Wraps the same Elasticsearch vector search used in:
    - mcp_sever/misc/elastic_search.py (similarity_search)
    - utils_packages/RAG_utils (ElasticsearchDB, AutoElasticSearchProcessor)
"""

import os
from typing import Any

from pydantic import BaseModel, Field

from ..abs_interface.utility_tool import AbstractUtilityTool
from ..core.registry import ServiceToolRegistry


class KBSearchInput(BaseModel):
    query: str = Field(description="Search query to find relevant documents in the knowledge base")
    k: int = Field(default=3, description="Number of results to return (1-10)")


@ServiceToolRegistry.register
class KnowledgeBaseSearchTool(AbstractUtilityTool):
    """Search internal knowledge base (uploaded documents, past estimations) via Elasticsearch."""

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Search the internal knowledge base for relevant documents. "
            "The knowledge base contains uploaded files (PDF, Excel), past project estimations, "
            "and other company data indexed in Elasticsearch. "
            "Use this when the user asks about internal data, past projects, "
            "or company-specific information. "
            "Input: search query and number of results. "
            "Returns: list of relevant document excerpts with metadata."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return KBSearchInput

    async def call_api(self, query: str, k: int = 3, **_) -> Any:
        es_url = os.getenv("ELASTICSEARCH_URL", "")
        if not es_url:
            return {"error": "ELASTICSEARCH_URL not configured"}

        try:
            from langchain_elasticsearch import ElasticsearchStore
            from langchain_openai import OpenAIEmbeddings

            embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
            embeddings = OpenAIEmbeddings(model=embedding_model)

            es_store = ElasticsearchStore(
                es_url=es_url,
                index_name=os.getenv("ELASTICSEARCH_INDEX", ""),
                embedding=embeddings,
                es_user=os.getenv("ELASTICSEARCH_USERNAME", ""),
                es_password=os.getenv("ELASTICSEARCH_PASSWORD", ""),
                es_params={"verify_certs": False},
            )

            results = es_store.similarity_search(query=query, k=k)
            return {
                "query": query,
                "results": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in results
                ],
            }
        except Exception as e:
            return {"error": f"Knowledge base search failed: {str(e)}"}
