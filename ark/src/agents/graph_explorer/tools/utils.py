import polars as pl

STOPWORDS = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "must",
    "shall",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
]


def filter_by_node_type(candidates: pl.DataFrame, node_type: str) -> pl.DataFrame:
    """Filter candidates by node type."""
    return candidates.filter(pl.col("type") == node_type)


def filter_by_query(candidates: pl.DataFrame, query: str) -> pl.DataFrame:
    """Filter candidates by query, matching in name or summary."""
    # NOTE: Removed fuzzy match by now
    lower_query = query.lower()
    candidates_w_query_in_name = candidates.filter(
        pl.col("name").is_not_null()
        & pl.col("name").str.to_lowercase().str.contains(lower_query, literal=True)
    )

    query_words = query.replace("-", " ").split()
    query_words = [word.lower() for word in query_words if word not in STOPWORDS]
    candidates_w_query_in_details = (
        candidates.filter(~pl.col("index").is_in(candidates_w_query_in_name["index"]))
        .with_columns(
            pl.col("summary")
            .fill_null("")
            .map_elements(
                lambda summary: [word for word in query_words if word in str(summary).lower()],
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("appearing_words")
        )
        .with_columns(pl.col("appearing_words").list.len().alias("relevance"))
        .filter(pl.col("relevance") > 0)
        .sort("relevance", descending=True)
    )

    def return_matching_segments(words: list[str], summary: str):
        if not summary:
            return ""
        summary_lower = summary.lower()
        matching_segments: list[str] = []
        for word in words:
            word_lower = str(word).lower()
            idx = summary_lower.find(word_lower)
            if idx == -1:
                continue
            matching_segments.append(summary[max(0, idx - 30) : min(len(summary), idx + 30)])
        return " ... ".join(matching_segments)

    def build_explanation(row: dict) -> str:
        words = row.get("appearing_words") or []
        summary = row.get("summary") or ""
        if not words or not summary:
            return ""
        return (
            f"{len(words)} match{'es' if len(words) != 1 else ''}: "
            + return_matching_segments(words, summary)
        )

    candidates_w_query_in_details = candidates_w_query_in_details.with_columns(
        pl.struct(["appearing_words", "summary"])
        .map_elements(build_explanation)
        .alias("match_explanation")
    )

    # Ensure both DataFrames have the same schema and compatible dtypes before concatenation
    candidates_w_query_in_name = candidates_w_query_in_name.with_columns(
        pl.lit([]).cast(pl.List(pl.Utf8)).alias("appearing_words"),
        pl.lit(0).cast(pl.Int64).alias("relevance"),
        pl.lit("").cast(pl.Utf8).alias("match_explanation"),
    )
    candidates_w_query_in_details = candidates_w_query_in_details.with_columns(
        pl.col("appearing_words").cast(pl.List(pl.Utf8)),
        pl.col("relevance").cast(pl.Int64),
        pl.col("match_explanation").cast(pl.Utf8),
    )

    return pl.concat([candidates_w_query_in_name, candidates_w_query_in_details])
