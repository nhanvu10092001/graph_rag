# Entity & Relationship Gleaning Prompt

You previously extracted entities and relationships from the text below. Review your extraction and identify any entities or relationships you may have missed.

## Previously Extracted Entities
{existing_entities}

## Previously Extracted Relationships
{existing_relationships}

## Original Text
{text}

## Instructions
- Carefully re-read the text and identify any entities or relationships that were missed.
- Only return NEW entities and relationships NOT already in the lists above.
- Follow the same format: each entity needs id (UPPERCASE), type, description; each relationship needs source, target, type, description.
- If the previous extraction is already complete, return empty lists for both entities and relationships.
