"""
Similarity search.

Two backends over the same feature matrix:

  exact  numpy matrix-vector product. O(n*d) per query, zero build cost,
         100% recall by definition. The correctness baseline.

  hnsw   Hierarchical Navigable Small World graph (Malkov & Yashunin 2016,
         arXiv:1603.09320) via the `hnswlib` bindings. Sub-linear query time
         at the cost of a build step and approximate recall.

Why HNSW rather than IVF or LSH
-------------------------------
HNSW builds a multi-layer proximity graph: upper layers are sparse and act
as express lanes, lower layers are dense and refine locally. A search greedily
descends from an entry point, so query cost grows roughly logarithmically with
dataset size rather than linearly.

Against the alternatives:

  IVF (inverted file / coarse quantisation) needs a training pass to learn
  centroids and its recall is sensitive to how many cells you probe. It wins
  on memory at much larger scale, but at 30k vectors the training step is
  pure overhead.

  LSH needs many hash tables to reach comparable recall, and its guarantees
  are asymptotic - it underperforms graph methods at this scale in practice.

  Exact search is genuinely fine at 30k. HNSW is included to demonstrate the
  scaling path: the crossover where the graph pays for itself is around
  10^5-10^6 vectors, and the benchmark in the README reports both so the
  claim is measured rather than asserted.

Inner product vs cosine
-----------------------
Feature rows are unit-norm by construction (see features.py), so inner
product, cosine similarity and Euclidean distance induce the same ranking.
The 'ip' space is used directly, which avoids hnswlib re-normalising.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from app.config import (
    DATASET_PATH,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    WEIGHTS,
)
from app.data import load_and_normalise
from app.features import build_feature_matrix

logger = logging.getLogger(__name__)

Backend = Literal["exact", "hnsw"]


class ProductNotFoundError(KeyError):
    """Raised when a product_id is not present in the dataset."""


@dataclass
class SimilarProduct:
    """A single search result."""

    product_id: str
    score: float
    product_name: str
    brand: Optional[str]
    child_category: Optional[str]
    sales_price: Optional[float]
    rating: Optional[float]
    image_url: Optional[str]


class SimilarityIndex:
    """
    Holds the dataset, the feature matrix and (optionally) an ANN graph.

    Built once at process start. Queries are read-only and therefore
    thread-safe, which matters because uvicorn runs sync endpoint handlers
    in a thread pool.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        matrix: np.ndarray,
        blocks=None,
        build_ann: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.blocks = blocks or []

        # product_id -> row index. Dict lookup is O(1); a pandas boolean mask
        # would be O(n) per query and dominate the cost of the search itself.
        self._index_of: dict[str, int] = {
            pid: i for i, pid in enumerate(self.df["product_id"])
        }

        self._ann = None
        self._ann_lock = threading.Lock()
        self.build_stats: dict[str, float] = {}

        if build_ann:
            self._build_ann()

    # ---------------------------------------------------------------- build

    @classmethod
    def from_dataset(
        cls,
        path: str | Path = DATASET_PATH,
        weights: Optional[dict[str, float]] = None,
        build_ann: bool = True,
    ) -> "SimilarityIndex":
        started = time.perf_counter()
        df = load_and_normalise(path)
        loaded = time.perf_counter()

        matrix, blocks = build_feature_matrix(df, weights or WEIGHTS)
        featured = time.perf_counter()

        index = cls(df, matrix, blocks, build_ann=build_ann)
        index.build_stats.update(
            {
                "load_seconds": loaded - started,
                "feature_seconds": featured - loaded,
                "n_products": float(len(df)),
                "n_dimensions": float(matrix.shape[1]),
                "matrix_mb": matrix.nbytes / 1e6,
            }
        )
        logger.info(
            "Index ready: %d products, %d dims, %.0f MB",
            len(df),
            matrix.shape[1],
            matrix.nbytes / 1e6,
        )
        return index

    def _build_ann(self) -> None:
        """Build the HNSW graph. Degrades to exact search if unavailable."""
        try:
            import hnswlib
        except ImportError:
            logger.warning("hnswlib not installed - falling back to exact search")
            return

        started = time.perf_counter()
        n, dim = self.matrix.shape

        ann = hnswlib.Index(space="ip", dim=dim)
        ann.init_index(max_elements=n, ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        ann.add_items(self.matrix, np.arange(n))
        ann.set_ef(HNSW_EF_SEARCH)

        self._ann = ann
        self.build_stats["ann_build_seconds"] = time.perf_counter() - started
        logger.info("HNSW index built in %.1fs", self.build_stats["ann_build_seconds"])

    # ---------------------------------------------------------------- query

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._index_of

    def row_of(self, product_id: str) -> int:
        try:
            return self._index_of[product_id]
        except KeyError as exc:
            raise ProductNotFoundError(product_id) from exc

    def _search_exact(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.matrix @ vector

        # argpartition finds the top k without fully sorting all 30k scores,
        # then only that slice is sorted. O(n + k log k) rather than O(n log n).
        k = min(k, scores.shape[0])
        candidates = np.argpartition(-scores, k - 1)[:k]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        return ordered, scores[ordered]

    def _search_ann(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        with self._ann_lock:
            labels, distances = self._ann.knn_query(vector, k=min(k, len(self.df)))
        # hnswlib returns 1 - inner_product for the 'ip' space.
        return labels[0], 1.0 - distances[0]

    def search(
        self,
        product_id: str,
        num_similar: int = 10,
        backend: Backend = "exact",
    ) -> list[SimilarProduct]:
        """
        Return the `num_similar` products most similar to `product_id`.

        The query product is always excluded. One extra candidate is
        requested so removing it never leaves the caller short.
        """
        if num_similar <= 0:
            return []

        row = self.row_of(product_id)
        vector = self.matrix[row]

        use_ann = backend == "hnsw" and self._ann is not None
        fetch = num_similar + 1
        indices, scores = (
            self._search_ann(vector, fetch) if use_ann else self._search_exact(vector, fetch)
        )

        results: list[tuple[int, float]] = [
            (int(i), float(s)) for i, s in zip(indices, scores) if int(i) != row
        ]

        # Tie-break deterministically: score desc, then rating desc, then
        # price asc, then product_id. Without this, equal-scoring products
        # (common - the dataset contains near-duplicate listings) would come
        # back in whatever order the backend happened to produce, and the
        # same query could return different results across runs.
        def sort_key(item: tuple[int, float]):
            idx, score = item
            product = self.df.iloc[idx]
            rating = product["rating"]
            price = product["sales_price"]
            return (
                -round(score, 6),
                -(rating if pd.notna(rating) else -1.0),
                price if pd.notna(price) else float("inf"),
                product["product_id"],
            )

        results.sort(key=sort_key)
        return [self._to_result(idx, score) for idx, score in results[:num_similar]]

    def _to_result(self, idx: int, score: float) -> SimilarProduct:
        product = self.df.iloc[idx]

        def optional(value):
            return None if pd.isna(value) else value

        return SimilarProduct(
            product_id=str(product["product_id"]),
            score=round(float(score), 6),
            product_name=str(product["product_name"]),
            brand=optional(product["brand"]),
            child_category=optional(product["child_category"]),
            sales_price=optional(product["sales_price"]),
            rating=optional(product["rating"]),
            image_url=optional(product["image_url"]),
        )

    # ------------------------------------------------------------ benchmark

    def recall_at_k(self, k: int = 10, sample_size: int = 200, seed: int = 42) -> float:
        """
        Measure ANN recall against exact search.

        This is the number that justifies using HNSW at all - without it,
        "approximate" is an unquantified claim.
        """
        if self._ann is None:
            return 1.0

        rng = np.random.default_rng(seed)
        rows = rng.choice(len(self.df), size=min(sample_size, len(self.df)), replace=False)

        hits = 0
        for row in rows:
            pid = self.df.iloc[row]["product_id"]
            exact = {r.product_id for r in self.search(pid, k, backend="exact")}
            approx = {r.product_id for r in self.search(pid, k, backend="hnsw")}
            hits += len(exact & approx)

        return hits / (len(rows) * k)


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------

_index: Optional[SimilarityIndex] = None
_index_lock = threading.Lock()


def get_index() -> SimilarityIndex:
    """
    Return the process-wide index, building it on first use.

    Double-checked locking so concurrent first requests cannot trigger two
    expensive builds. In the API this is warmed during startup so no request
    ever pays the build cost.
    """
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = SimilarityIndex.from_dataset()
    return _index


def set_index(index: Optional[SimilarityIndex]) -> None:
    """Inject an index. Used by tests to avoid loading the full dataset."""
    global _index
    _index = index
    find_similar_products.cache_clear()


# --------------------------------------------------------------------------
# Part 1 of the brief: the required function signature
# --------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def find_similar_products(product_id: str, num_similar: int = 10) -> list[str]:
    """
    Return the IDs of the `num_similar` products most similar to `product_id`.

    This is the exact signature the exercise asks for. It delegates to
    SimilarityIndex so the richer result objects stay available to the API.

    Caching: product catalogues are read-heavy and the feature matrix is
    immutable for the lifetime of the process, so results are perfectly
    cacheable. The LRU holds the 4,096 hottest queries; a real deployment
    would use Redis so the cache survives restarts and is shared across pods.
    """
    results = get_index().search(product_id, num_similar)
    return [r.product_id for r in results]


def calculate_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Cosine similarity between two feature vectors.

    Provided because the brief's stub includes it. The production path never
    calls this - scoring one pair at a time would mean 30,000 Python-level
    calls per query. The vectorised matrix product in `_search_exact` does
    the same arithmetic in a single BLAS call, roughly three orders of
    magnitude faster.
    """
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)
