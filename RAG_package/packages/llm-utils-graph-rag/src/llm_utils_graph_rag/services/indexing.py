"""Graph Indexing Service — Entity/Relationship/Claim extraction and ingestion to Neo4j."""

import hashlib
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


class ClaimModel(BaseModel):
    subject_id: str = Field(description="Entity ID of the claim's subject")
    object_id: Optional[str] = Field(default=None, description="Entity ID of the claim's object, if applicable")
    claim_type: str = Field(default="FACTUAL", description="Type: FACTUAL, TEMPORAL, CAUSAL, QUANTITATIVE, STATUS_CHANGE")
    claim_status: str = Field(default="STATED", description="Status: STATED, INFERRED, DISPUTED")
    claim_description: str = Field(description="The factual claim text")
    claim_date: Optional[str] = Field(default=None, description="ISO date if temporal")
    claim_source_text: str = Field(default="", description="Exact source text for this claim")


class ExtractionResultBase(BaseModel):
    """Schema without claims — used when extract_claims=False and for gleaning."""
    entities: List[EntityModel] = Field(default_factory=list)
    relationships: List[RelationshipModel] = Field(default_factory=list)


class ExtractionResult(ExtractionResultBase):
    """Full schema with claims — used when extract_claims=True."""
    claims: List[ClaimModel] = Field(default_factory=list)


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

        graph_extraction_cfg = self.config.get("graph_extraction", {})
        self.max_gleaning_rounds = int(graph_extraction_cfg.get("max_gleaning_rounds", 0))
        self.extract_claims = bool(graph_extraction_cfg.get("extract_claims", False))

        if self.max_gleaning_rounds > 0:
            self.gleaning_prompt = load_prompt_file("gleaning_prompt.md")
        if self.extract_claims:
            self.claims_addendum = load_prompt_file("claims_extraction_addendum.md")

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

    def _run_gleaning(
        self,
        text: str,
        entities: List[EntityModel],
        relationships: List[RelationshipModel],
    ) -> tuple:
        """Run gleaning loop to discover missed entities/relationships."""
        for round_num in range(self.max_gleaning_rounds):
            existing_entities_text = "\n".join(
                f"- {e.id} ({e.type}): {e.description}" for e in entities
            )
            existing_rels_text = "\n".join(
                f"- {r.source} -[{r.type}]-> {r.target}: {r.description}" for r in relationships
            )

            prompt = self.gleaning_prompt.format(
                text=text,
                existing_entities=existing_entities_text,
                existing_relationships=existing_rels_text,
            )

            try:
                structured_llm = self.llm.with_structured_output(ExtractionResultBase)
                gleaning_result = structured_llm.invoke(prompt)
            except Exception as e:
                logger.warning(f"Gleaning round {round_num + 1} failed: {e}")
                break

            new_entities = gleaning_result.entities or []
            new_rels = gleaning_result.relationships or []

            if not new_entities and not new_rels:
                logger.info(f"Gleaning round {round_num + 1}: nothing new. Stopping.")
                break

            existing_ids = {e.id.strip().upper() for e in entities}
            added_ents = 0
            for ne in new_entities:
                if ne.id.strip().upper() not in existing_ids:
                    entities.append(ne)
                    existing_ids.add(ne.id.strip().upper())
                    added_ents += 1

            existing_rel_keys = {
                (r.source.strip().upper(), r.target.strip().upper(), r.type.strip().upper())
                for r in relationships
            }
            added_rels = 0
            for nr in new_rels:
                key = (nr.source.strip().upper(), nr.target.strip().upper(), nr.type.strip().upper())
                if key not in existing_rel_keys:
                    relationships.append(nr)
                    existing_rel_keys.add(key)
                    added_rels += 1

            logger.info(f"Gleaning round {round_num + 1}: +{added_ents} entities, +{added_rels} relationships")

        return entities, relationships

    def index_text(self, text: str, source_doc: Optional[str] = None) -> Dict[str, Any]:
        """Extract and insert graph nodes/edges from text passage."""
        self.graph_store.connect()
        source_doc_val = source_doc or "unknown"

        # 1. Run LLM Extraction with structured output
        prompt = self.extraction_prompt.format(text=text)
        if self.extract_claims:
            prompt = prompt + "\n\n" + self.claims_addendum

        try:
            schema_class = ExtractionResult if self.extract_claims else ExtractionResultBase
            structured_llm = self.llm.with_structured_output(schema_class)
            result = structured_llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Structured extraction failed: {e}", exc_info=True)
            return {"status": "error", "error": f"Extraction failure: {e}"}

        entities = result.entities
        relationships = result.relationships

        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships.")

        # 1.5. Gleaning loop (if configured)
        if self.max_gleaning_rounds > 0:
            entities, relationships = self._run_gleaning(text, entities, relationships)
            logger.info(f"After gleaning: {len(entities)} entities, {len(relationships)} relationships")

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

        # 5. Write Claims to Neo4j (if enabled)
        indexed_claims = 0
        claims = getattr(result, 'claims', [])
        if self.extract_claims and claims:
            for claim in claims:
                subject = claim.subject_id.strip().upper()
                if not subject:
                    continue

                claim_hash = hashlib.sha256(
                    f"{subject}:{claim.claim_description.strip().lower()}".encode()
                ).hexdigest()[:16]
                claim_id = f"CLM_{claim_hash}"

                cypher = """
                MERGE (cl:Claim {id: $claim_id})
                ON CREATE SET cl.subject_id = $subject_id,
                              cl.object_id = $object_id,
                              cl.claim_type = $claim_type,
                              cl.claim_status = $claim_status,
                              cl.description = $description,
                              cl.claim_date = $claim_date,
                              cl.source_text = $source_text,
                              cl.source_documents = [$source_doc]
                ON MATCH SET cl.source_documents = CASE
                                 WHEN cl.source_documents IS NULL THEN [$source_doc]
                                 WHEN $source_doc IN cl.source_documents THEN cl.source_documents
                                 ELSE cl.source_documents + [$source_doc]
                             END
                """
                try:
                    self.graph_store.query(cypher, {
                        "claim_id": claim_id,
                        "subject_id": subject,
                        "object_id": (claim.object_id or "").strip().upper() or None,
                        "claim_type": claim.claim_type.strip().upper(),
                        "claim_status": claim.claim_status.strip().upper(),
                        "description": claim.claim_description.strip(),
                        "claim_date": claim.claim_date,
                        "source_text": claim.claim_source_text.strip(),
                        "source_doc": source_doc_val,
                    })

                    self.graph_store.query("""
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (cl:Claim {id: $claim_id})
                    MERGE (e)-[:HAS_CLAIM]->(cl)
                    """, {"entity_id": subject, "claim_id": claim_id})

                    if claim.object_id:
                        obj = claim.object_id.strip().upper()
                        self.graph_store.query("""
                        MATCH (e:Entity {id: $entity_id})
                        MATCH (cl:Claim {id: $claim_id})
                        MERGE (cl)-[:REFERENCES]->(e)
                        """, {"entity_id": obj, "claim_id": claim_id})

                    indexed_claims += 1
                except Exception as e:
                    logger.error(f"Failed to write claim for {subject}: {e}")

        return {
            "status": "success",
            "extracted_entities": len(entities),
            "indexed_entities": indexed_entities,
            "extracted_relationships": len(relationships),
            "indexed_relationships": indexed_relationships,
            "extracted_claims": len(claims) if self.extract_claims else 0,
            "indexed_claims": indexed_claims,
        }

    def delete_document_from_graph(self, source_doc: str) -> Dict[str, Any]:
        """Removes a document's association from entities/relationships/claims, deleting them if orphan."""
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

        # 2. Clean up orphan claims
        claim_cypher = """
        MATCH (cl:Claim)
        WHERE cl.source_documents IS NULL OR $source_doc IN cl.source_documents
        SET cl.source_documents = [x IN COALESCE(cl.source_documents, []) WHERE x <> $source_doc]
        WITH cl
        WHERE size(cl.source_documents) = 0
        DETACH DELETE cl
        """
        self.graph_store.query(claim_cypher, {"source_doc": source_doc})

        # 3. Update entities: remove source_doc from source_documents, treat NULL as empty
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
