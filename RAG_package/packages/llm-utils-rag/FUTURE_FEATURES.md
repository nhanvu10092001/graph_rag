# Future RAG Features — Phase 2+

> These features have been identified as valuable but are deferred to future phases.
> Reference: implementation_plan.md from 2026-05-16

---

## Feature 4: Streaming Citations & Source Attribution
**Priority**: 🟡 Medium | **Scope**: Small-Medium

**Problem**: `query_service._build_sources()` trả sources sau khi query xong nhưng:
- Không có inline citations trong answer text
- Không biết sentence nào đến từ chunk nào

**Solution**:
1. **Inline citations**: Prompt LLM generate `[1]`, `[2]` references trong answer
2. **Source mapping**: Map citation numbers → source chunks
3. **Confidence scores**: Mỗi citation có similarity score
4. **Streaming**: Stream answer tokens + emit citation events real-time

**Files to create**:
- `utils/retrieval/citation_chain.py` — citation-aware prompt + post-processing
- `services/query_service.py` — add citation mode
- Frontend: Render clickable `[1]` badges linking to source preview

---

## Feature 6: Knowledge Graph Retrieval
**Priority**: 🟢 Phase 2 | **Scope**: Large

**Problem**: Documents are flat vectors with no relationship awareness.

**Solution**: Lightweight knowledge graph:
1. Entity extraction during indexing (LLM-based)
2. Graph storage (Neo4j or in-memory NetworkX)
3. Graph-enhanced retrieval: vector search + graph traversal

**Prerequisites**: Neo4j setup or decision on in-memory graph

---

## Feature 7: Multi-Modal RAG
**Priority**: 🟢 Phase 2 | **Scope**: Large

**Problem**: OCR extracts text from images but doesn't understand image content.

**Solution**:
- Index images with CLIP embeddings
- Multi-modal retrieval (text + image search)
- Return relevant images alongside text answers

**Prerequisites**: CLIP model deployment, image storage solution

---

## Feature 8: Semantic Cache
**Priority**: 🟢 Phase 2 | **Scope**: Small

**Problem**: Every query runs the full pipeline (embed → retrieve → rerank → generate).

**Solution**: Semantic cache layer:
- Cache query embeddings + results
- If new query has cosine similarity > 0.95 with cached query → return cached
- TTL-based invalidation when documents change

**Files to create**:
- `services/cache_service.py` — Redis-backed semantic cache
- Wrap into `query_service` as middleware

**Prerequisites**: Redis availability or in-memory fallback decision
