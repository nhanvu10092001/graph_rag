"""Embedding index on disk: parallel embedding API calls, Parquet chunks, single final merge."""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from litellm import embedding

from src.core.logger import logger
from src.core.node import Node

DEFAULT_EMBEDDING_MODEL = "azure/text-embedding-3-large"

BATCH_SIZE = 100
FLUSH_EVERY_ROWS = 10_000
PROGRESS_LOG_EVERY = 1000
MAX_WORKERS = max(1, int(os.environ.get("GRAPHSEARCH_EMBEDDING_MAX_WORKERS", "8")))
MAX_CHARS = 20_000
API_RETRIES = 5

_CHUNK_RE = re.compile(r"^(\d{6})\.parquet$")


def _stem(model: str) -> str:
    return model.split("/")[-1]


def _model_dir(base: Path, model: str) -> Path:
    return base / _stem(model)


def _chunks_dir(base: Path, model: str) -> Path:
    return _model_dir(base, model) / "chunks"


def _final_path(base: Path, model: str) -> Path:
    return _model_dir(base, model) / f"{_stem(model)}.parquet"


def _legacy_flat(base: Path, model: str) -> Path:
    return base / f"{_stem(model)}.parquet"


def _load_parquet_path(base: Path, model: str) -> Path:
    new_p, leg = _final_path(base, model), _legacy_flat(base, model)
    if new_p.exists():
        return new_p
    return leg if leg.exists() else new_p


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    """Atomically write a PyArrow table to Parquet via a temp file."""
    path = path.resolve()
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _indices_from_parquet(p: Path) -> pl.Series:
    return pl.read_parquet(p, columns=["index"])["index"]


def _existing_indices(path: Path, model: str) -> pl.Series:
    """Collect all already-embedded indices as a Polars Series (stays in Arrow)."""
    parts: list[pl.Series] = []
    final_p, leg = _final_path(path, model), _legacy_flat(path, model)
    cd = _chunks_dir(path, model)

    def add(p: Path) -> None:
        try:
            parts.append(_indices_from_parquet(p))
        except Exception as e:
            bad = p.with_name(f"{p.stem}.bad.{int(time.time())}.parquet")
            logger.error(f"Unreadable {p} ({e!r}) → {bad.name}")
            os.replace(p, bad)

    if final_p.exists():
        add(final_p)
    elif leg.exists():
        add(leg)
    if cd.exists():
        for p in sorted(cd.glob("*.parquet")):
            add(p)
    if not parts:
        return pl.Series("index", [], dtype=pl.Int64)
    return pl.concat(parts).unique()


def _next_chunk_id(chunks_dir: Path) -> int:
    if not chunks_dir.exists():
        return 1
    nums = [int(m.group(1)) for p in chunks_dir.glob("*.parquet") if (m := _CHUNK_RE.match(p.name))]
    return max(nums, default=0) + 1


def _merge_stream(dest: Path, sources: list[Path]) -> int:
    """Append Parquet files into ``dest`` without loading whole files into pandas."""
    if not sources:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".merge.{os.getpid()}.tmp.parquet"
    writer: pq.ParquetWriter | None = None
    n = 0
    try:
        for src in sources:
            pf = pq.ParquetFile(src, memory_map=True)
            for batch in pf.iter_batches(batch_size=65536):
                t = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(tmp, t.schema, compression="snappy")
                writer.write_table(t)
                n += t.num_rows
        if writer:
            writer.close()
            writer = None
        os.replace(tmp, dest)
    except Exception:
        if writer:
            try:
                writer.close()
            except Exception:
                pass
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    return n


def _merge_all(path: Path, model: str) -> None:
    model_dir = _model_dir(path, model)
    chunks_dir = _chunks_dir(path, model)
    final_p = _final_path(path, model)
    leg = _legacy_flat(path, model)

    chunks = sorted(
        chunks_dir.glob("*.parquet"),
        key=lambda p: (int(m.group(1)) if (m := _CHUNK_RE.match(p.name)) else 0, p.name),
    )
    has_final = final_p.exists()
    has_leg = leg.exists()

    if not chunks:
        if has_final:
            return
        if has_leg:
            model_dir.mkdir(parents=True, exist_ok=True)
            _merge_stream(final_p, [leg])
            leg.unlink()
        return

    srcs: list[Path] = []
    if has_final:
        srcs.append(final_p)
    elif has_leg:
        srcs.append(leg)
    srcs.extend(chunks)

    logger.info(f"Merging {len(chunks)} chunk(s) → {final_p.name}")
    rows = _merge_stream(final_p, srcs)
    for c in chunks:
        c.unlink()
    if has_leg and leg.exists():
        leg.unlink()
    logger.info(f"Merge done: {rows} rows → {final_p}")


def _truncate(s: str) -> str:
    return s[:MAX_CHARS] if len(s) > MAX_CHARS else s


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    for attempt in range(API_RETRIES):
        try:
            r = embedding(model=model, input=texts)
            return [x["embedding"] for x in r.data]
        except Exception as e:
            if attempt < API_RETRIES - 1 and any(
                x in str(e).lower() for x in ("429", "rate", "throttl", "limit")
            ):
                time.sleep(min(60.0, 2.0**attempt))
                continue
            raise


class EmbeddingIndex:
    def __init__(self, path: Path, nodes_df: pl.DataFrame, model: str = DEFAULT_EMBEDDING_MODEL):
        self._model = model
        p = _load_parquet_path(path, model)
        if not p.exists():
            raise FileNotFoundError(f"No embeddings at {p}")

        pf = pq.ParquetFile(p)
        n_rows = pf.metadata.num_rows

        idx_table = pq.read_table(p, columns=["index"])
        self._node_indices = idx_table.column("index").combine_chunks().to_numpy(
            zero_copy_only=False
        ).astype(np.int64)
        del idx_table

        if n_rows == 0:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            dim = len(next(pf.iter_batches(batch_size=1, columns=["embedding"])).column("embedding")[0])
            self._embeddings = np.empty((n_rows, dim), dtype=np.float32)
            offset = 0
            for batch in pf.iter_batches(batch_size=10000, columns=["embedding"]):
                col = batch.column("embedding")
                batch_rows = len(col)
                chunk_flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32)
                self._embeddings[offset:offset + batch_rows] = chunk_flat.reshape(batch_rows, dim)
                offset += batch_rows

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings /= norms

        order = np.argsort(self._node_indices)
        self._node_indices = self._node_indices[order]
        self._embeddings = self._embeddings[order]

        self._nodes_df = nodes_df.sort("index")
        self._nodes_idx = self._nodes_df["index"].to_numpy()

    @classmethod
    def from_nodes_df(
        cls, path: Path, nodes_df: pl.DataFrame, model: str = DEFAULT_EMBEDDING_MODEL
    ) -> "EmbeddingIndex":
        path.mkdir(parents=True, exist_ok=True)
        model_dir = _model_dir(path, model)
        chunks_dir = _chunks_dir(path, model)
        model_dir.mkdir(parents=True, exist_ok=True)

        have = _existing_indices(path, model)
        have_count = len(have)
        if have_count > 0:
            pend = nodes_df.join(
                have.cast(pl.Int64).to_frame(),
                on="index",
                how="anti",
            )
        else:
            pend = nodes_df

        n = pend.height
        total_nodes = nodes_df.height

        if n == 0 and chunks_dir.exists() and list(chunks_dir.glob("*.parquet")):
            _merge_all(path, model)

        if n > 0:
            chunks_dir.mkdir(parents=True, exist_ok=True)
            cid = _next_chunk_id(chunks_dir)
            buf: list[pa.Table] = []
            buf_rows = 0
            done = 0
            run_embedded = 0
            last_log_at = have_count

            logger.info(
                f"Embedding {n} nodes ({have_count} already); "
                f"batch={BATCH_SIZE} workers={MAX_WORKERS}; chunks → {chunks_dir}"
            )

            def flush() -> None:
                nonlocal buf, buf_rows, cid, done
                if not buf:
                    return
                part = pa.concat_tables(buf)
                buf, buf_rows = [], 0
                p = chunks_dir / f"{cid:06d}.parquet"
                cid += 1
                _atomic_write_table(part, p)
                rows_written = part.num_rows
                done += rows_written
                logger.info(f"Chunk {p.name} ({rows_written} rows); total embedded this run: {done}/{n}")

            def job(start: int) -> pa.Table | None:
                b = pend[start : start + BATCH_SIZE]
                try:
                    texts = [_truncate(str(x) if x is not None else "") for x in b["summary"].to_list()]
                    embs = _embed_batch(texts, model)
                    return pa.table({
                        "index": pa.array(b["index"].to_list(), type=pa.int64()),
                        "embedding": pa.array(embs, type=pa.list_(pa.float64())),
                    })
                except Exception as e:
                    logger.error(f"Batch @ {start}: {e}")
                    return None

            starts = list(range(0, n, BATCH_SIZE))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(job, s) for s in starts]
                for fut in as_completed(futures):
                    try:
                        part = fut.result()
                    except Exception as e:
                        logger.error(f"Embedding worker failed: {e}")
                        continue
                    if part is None:
                        continue
                    buf.append(part)
                    part_rows = part.num_rows
                    buf_rows += part_rows
                    run_embedded += part_rows
                    done_global = have_count + run_embedded
                    if (
                        done_global - last_log_at >= PROGRESS_LOG_EVERY
                        or run_embedded >= n
                    ):
                        logger.info(
                            f"Embedded {done_global} / {total_nodes} nodes "
                            f"(this run {run_embedded}/{n})"
                        )
                        last_log_at = done_global
                    if buf_rows >= FLUSH_EVERY_ROWS:
                        flush()
            flush()
            _merge_all(path, model)
        else:
            logger.info(f"Embeddings complete ({have_count} nodes on disk).")

        return cls(path, nodes_df, model)

    def _embed_query(self, query: str) -> np.ndarray:
        r = embedding(model=self._model, input=[query])
        v = np.array(r.data[0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _resolve_nodes(self, indices: np.ndarray) -> list[Node]:
        positions = np.searchsorted(self._nodes_idx, indices)
        rows = self._nodes_df[positions.tolist()].to_dicts()
        return [Node.from_df_row(r) for r in rows]

    def _top_k(self, sim: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(sim))
        if k >= len(sim):
            return np.argsort(-sim)
        t = np.argpartition(-sim, k)[:k]
        return t[np.argsort(-sim[t])]

    def search(self, query: str, k: int = 10) -> tuple[list[Node], list[float]]:
        q = self._embed_query(query)
        s = self._embeddings @ q
        t = self._top_k(s, k)
        idx = self._node_indices[t]
        return self._resolve_nodes(idx), s[t].tolist()

    def search_in_subset(
        self, query: str, candidate_indices: list[int], k: int = 10
    ) -> tuple[list[int], list[float]]:
        q = self._embed_query(query)
        if not candidate_indices or len(self._node_indices) == 0:
            return [], []
        candidates = np.array(candidate_indices, dtype=np.int64)
        pos = np.searchsorted(self._node_indices, candidates)
        pos_clipped = np.minimum(pos, len(self._node_indices) - 1)
        valid = (pos < len(self._node_indices)) & (
            self._node_indices[pos_clipped] == candidates
        )
        pos = pos[valid]
        ids = candidates[valid]
        if len(pos) == 0:
            return [], []
        s = self._embeddings[pos] @ q
        t = self._top_k(s, k)
        return [int(ids[i]) for i in t], s[t].tolist()
