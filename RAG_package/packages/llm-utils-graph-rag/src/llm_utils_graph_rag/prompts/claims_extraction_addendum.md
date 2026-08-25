### 3. Claims / Covariates
In addition to entities and relationships, extract factual claims about entities — important statements such as dates, events, status changes, quantitative facts, and interactions.
For each claim, provide:
- `subject_id`: The entity ID this claim is about (must match an entity `id`).
- `object_id`: (Optional) Another entity ID involved in this claim.
- `claim_type`: One of `FACTUAL`, `TEMPORAL`, `CAUSAL`, `QUANTITATIVE`, `STATUS_CHANGE`.
- `claim_status`: One of `STATED` (directly in text), `INFERRED` (implied), `DISPUTED` (contradicted).
- `claim_description`: The claim text itself — a single, self-contained factual statement.
- `claim_date`: (Optional) ISO date string if temporal.
- `claim_source_text`: The exact sentence or phrase from the text supporting this claim.
