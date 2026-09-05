from src.core.node import Node

import re
import tantivy
import polars as pl
from pathlib import Path
from tqdm import tqdm


def clean_query(query: str) -> str:

    # Remove newlines and replace with spaces
    query = query.replace("\n", " ")

    # Remove leading/trailing dashes and spaces
    query = re.sub(r"^[-\s]+", "", query)
    query = re.sub(r"[-\s]+$", "", query)

    # Remove bullet point markers (-, •, *, etc.) at start of segments
    query = re.sub(r"(?:^|\s)[-•*]\s*", " ", query)

    # Remove special characters that can interfere with search
    query = query.replace(":", "")
    query = query.replace('"', "")
    query = query.replace("'", "")
    query = query.replace("(", "")
    query = query.replace(")", "")
    query = query.replace("[", "")
    query = query.replace("]", "")
    query = query.replace("{", "")
    query = query.replace("}", "")
    query = query.replace("<", "")
    query = query.replace(">", "")
    query = query.replace("+", " ")
    query = query.replace("^", "")
    query = query.replace("-", " ")
    query = query.replace("`", " ")

    # Normalize whitespace (multiple spaces -> single space)
    query = re.sub(r"\s+", " ", query)

    # Strip leading/trailing whitespace
    query = query.strip()

    # Escape problematic terms instead of lowercasing
    query = re.sub(r"\b(AND|OR|NOT|DE|IN|THE)\b", r'"\1"', query)

    return query


class GraphIndex:

    def __init__(self, path: Path):
        self.index = tantivy.Index.open(str(path))

    @classmethod
    def from_nodes_df(cls, path: Path, nodes: pl.DataFrame):
        path.mkdir()

        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_integer_field("index", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_text_field("summary", stored=True)
        schema = schema_builder.build()

        index = tantivy.Index(schema, path=str(path))
        writer = index.writer()
        for row in tqdm(nodes.iter_rows(named=True), total=nodes.height, desc="Populating index"):
            node = Node.from_df_row(row)
            doc = tantivy.Document()
            doc.add_text("name", clean_query(node.name))
            doc.add_integer("index", node.index)
            doc.add_text("type", node.type)
            doc.add_text("summary", clean_query(node.summary))
            writer.add_document(doc)
        writer.commit()

        return cls(path=path)

    def search(self, query: str, k: int = 10):
        query = clean_query(query)
        parsed_query = self.index.parse_query(query, ["name", "summary"])

        searcher = self.index.searcher()
        search_result = searcher.search(parsed_query, k)

        nodes = []
        scores = []
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)

            scores.append(score)
            nodes.append(Node.from_doc(doc))

        return nodes, scores
