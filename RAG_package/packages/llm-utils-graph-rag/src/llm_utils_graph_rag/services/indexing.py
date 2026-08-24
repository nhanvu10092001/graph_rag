"""Graph Indexing Service — Entity/Relationship extraction and ingestion to Neo4j."""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from ..graph_store import Neo4jGraphStore
from ..prompts import load_prompt_file

logger = logging.getLogger(__name__)

# Dynamically load default extraction prompt from markdown file
EXTRACTION_PROMPT = load_prompt_file("extraction_prompt.md")


class EntityModel(BaseModel):
    id: str
    type: str = "CONCEPT"
    description: str = ""


class RelationshipModel(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""


class ExtractionResult(BaseModel):
    entities: List[EntityModel] = Field(default_factory=list)
    relationships: List[RelationshipModel] = Field(default_factory=list)


class GraphIndexingService:
    """Extracts entities & relationships from texts and indexes them into Neo4j."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        prompt_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.graph_store = graph_store
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or {}
        if prompt_template:
            self.extraction_prompt = prompt_template
        elif prompt_path:
            self.extraction_prompt = load_prompt_file(custom_path=prompt_path)
        else:
            self.extraction_prompt = EXTRACTION_PROMPT

    def index_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from file bytes and index into the knowledge graph."""
        from llm_utils_parser import extract_text

        extraction_config = self.config.get("extraction")
        text = extract_text(filename, file_bytes, config=extraction_config)
        return self.index_text(text, source_doc=filename)

    def _setup_vector_index(self, dimension: int):
        """Configure Neo4j Vector Index for entity descriptions."""
        if getattr(self, "_index_initialized", False):
            return
        try:
            check = self.graph_store.query("SHOW INDEXES YIELD name WHERE name = 'entity_embeddings' RETURN name")
            if not check:
                create_index_query = """
                CREATE VECTOR INDEX `entity_embeddings` IF NOT EXISTS
                FOR (n:Entity) ON (n.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: $dimension,
                  `vector.similarity_function`: 'cosine'
                }}
                """
                self.graph_store.query(create_index_query, {"dimension": dimension})
                logger.info("Successfully set up Neo4j vector index 'entity_embeddings'.")
            self._index_initialized = True
        except Exception as e:
            logger.warning(f"Could not create vector index (might be older Neo4j version or already exists): {e}")

    def index_text(self, text: str, source_doc: Optional[str] = None) -> Dict[str, Any]:
        """Extract and insert graph nodes/edges from text passage."""
        self.graph_store.connect()
        source_doc_val = source_doc or "unknown"

        # 1. Run LLM Extraction with structured output
        prompt = self.extraction_prompt.format(text=text)
        try:
            structured_llm = self.llm.with_structured_output(ExtractionResult)
            result = structured_llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}", exc_info=True)
            return {"status": "error", "error": f"Extraction failure: {e}"}

        entities = result.entities
        relationships = result.relationships

        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships.")

        # 2. Setup index on first run
        if entities:
            test_emb = self.embeddings.embed_query("test")
            self._setup_vector_index(len(test_emb))

        # 3. Write Entities to Neo4j
        indexed_entities = 0
        for entity in entities:
            ent_id = entity.id.strip().upper()
            ent_type = entity.type.strip().upper()
            ent_desc = entity.description.strip()

            if not ent_id:
                continue

            embedding = self.embeddings.embed_query(f"{ent_id} ({ent_type}): {ent_desc}")

            cypher = """
            MERGE (e:Entity {id: $id})
            ON CREATE SET e.type = $type, e.description = $description, e.embedding = $embedding, e.source_documents = [$source_doc]
            ON MATCH SET e.description = COALESCE(e.description, $description), e.embedding = $embedding,
                         e.source_documents = CASE
                             WHEN e.source_documents IS NULL THEN [$source_doc]
                             WHEN $source_doc IN e.source_documents THEN e.source_documents
                             ELSE e.source_documents + [$source_doc]
                         END
            RETURN e.id
            """
            try:
                self.graph_store.query(cypher, {
                    "id": ent_id,
                    "type": ent_type,
                    "description": ent_desc,
                    "embedding": embedding,
                    "source_doc": source_doc_val
                })
                indexed_entities += 1
            except Exception as e:
                logger.error(f"Failed to write entity {ent_id} to Neo4j: {e}")

        # 4. Write Relationships to Neo4j
        indexed_relationships = 0
        for rel in relationships:
            source = rel.source.strip().upper()
            target = rel.target.strip().upper()
            rel_type = rel.type.strip().upper()
            rel_desc = rel.description.strip()

            if not source or not target:
                continue

            clean_rel_type = re.sub(r"[^A-Z0-9_]", "", rel_type)
            if not clean_rel_type:
                clean_rel_type = "RELATED_TO"

            cypher = f"""
            MATCH (source:Entity {{id: $source}})
            MATCH (target:Entity {{id: $target}})
            MERGE (source)-[r:{clean_rel_type}]->(target)
            ON CREATE SET r.description = $description, r.source_documents = [$source_doc]
            ON MATCH SET r.description = COALESCE(r.description, $description),
                         r.source_documents = CASE
                             WHEN r.source_documents IS NULL THEN [$source_doc]
                             WHEN $source_doc IN r.source_documents THEN r.source_documents
                             ELSE r.source_documents + [$source_doc]
                         END
            RETURN type(r)
            """
            try:
                self.graph_store.query(cypher, {
                    "source": source,
                    "target": target,
                    "description": rel_desc,
                    "source_doc": source_doc_val
                })
                indexed_relationships += 1
            except Exception as e:
                logger.error(f"Failed to write relationship {source} -> {target} to Neo4j: {e}")

        return {
            "status": "success",
            "extracted_entities": len(entities),
            "indexed_entities": indexed_entities,
            "extracted_relationships": len(relationships),
            "indexed_relationships": indexed_relationships
        }

    def delete_document_from_graph(self, source_doc: str) -> Dict[str, Any]:
        """Removes a document's association from entities/relationships, deleting them if orphan."""
        self.graph_store.connect()
        logger.info(f"Deleting document '{source_doc}' from Neo4j graph...")

        # 1. Update relationships: remove source_doc from source_documents, treat NULL as empty
        rel_cypher = """
        MATCH (s)-[r]->(t)
        WHERE r.source_documents IS NULL OR $source_doc IN r.source_documents
        SET r.source_documents = [x IN COALESCE(r.source_documents, []) WHERE x <> $source_doc]
        WITH r
        WHERE size(r.source_documents) = 0
        DELETE r
        """
        self.graph_store.query(rel_cypher, {"source_doc": source_doc})

        # 2. Update entities: remove source_doc from source_documents, treat NULL as empty
        ent_cypher = """
        MATCH (e:Entity)
        WHERE e.source_documents IS NULL OR $source_doc IN e.source_documents
        SET e.source_documents = [x IN COALESCE(e.source_documents, []) WHERE x <> $source_doc]
        WITH e
        WHERE size(e.source_documents) = 0
        DETACH DELETE e
        """
        self.graph_store.query(ent_cypher, {"source_doc": source_doc})

        return {"status": "success", "message": f"Deleted document '{source_doc}' from Neo4j graph."}
