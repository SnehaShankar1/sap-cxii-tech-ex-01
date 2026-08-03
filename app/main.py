"""
FastAPI microservice (Part 2).

Endpoints
---------
GET /find_similar_products   the required endpoint
GET /products/{product_id}   inspect a single product (useful for testing)
GET /sample_ids              a few valid IDs, so the API is explorable
GET /health                  liveness probe
GET /ready                   readiness probe - only 200 once the index is warm
GET /stats                   build timings and index shape

Operational design
------------------
The feature matrix takes ~8s to build and the HNSW graph ~22s. That work
happens once in the lifespan startup hook, not on first request, so no user
ever pays for it and Kubernetes will not route traffic until /ready passes.

Liveness and readiness are deliberately separate. Liveness answers "is the
process alive" and must succeed immediately, or the kubelet will kill the pod
mid-build and the container will crash-loop forever. Readiness answers "can
this pod serve traffic" and stays 503 until the index exists.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Path, Query, Response, status
from pydantic import BaseModel, Field

from app.config import MAX_NUM_SIMILAR
from app.similarity import ProductNotFoundError, SimilarityIndex, get_index

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Response models - explicit schemas give correct OpenAPI docs at /docs
# --------------------------------------------------------------------------


class SimilarProductOut(BaseModel):
    product_id: str
    score: float = Field(..., description="Weighted cosine similarity in [0, 1]")
    product_name: str
    brand: Optional[str] = None
    child_category: Optional[str] = None
    sales_price: Optional[float] = Field(None, description="INR")
    rating: Optional[float] = None
    image_url: Optional[str] = None


class SimilarProductsResponse(BaseModel):
    product_id: str
    num_similar: int
    backend: str
    took_ms: float
    results: list[SimilarProductOut]


class ProductIdsResponse(BaseModel):
    """The literal List[str] shape the brief specifies."""

    product_ids: list[str]


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building similarity index...")
    try:
        index = get_index()
        app.state.index = index
        logger.info("Index ready with %d products", len(index))
    except FileNotFoundError:
        # Let the container start so /health reports the problem clearly,
        # rather than crash-looping with the reason buried in kubectl logs.
        app.state.index = None
        logger.error("Dataset not found - service will report unready")
    yield
    app.state.index = None


app = FastAPI(
    title="Product Similarity Search",
    description=(
        "Similarity search over the Amazon India fashion catalogue (30k "
        "products). Scores combine category, text, numeric, brand and colour "
        "features as a weighted sum of per-block cosine similarities."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_index() -> SimilarityIndex:
    index = getattr(app.state, "index", None)
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index is not ready. Check that the dataset is present.",
        )
    return index


# --------------------------------------------------------------------------
# Required endpoint
# --------------------------------------------------------------------------


@app.get(
    "/find_similar_products",
    response_model=SimilarProductsResponse,
    summary="Find products similar to a given product",
    responses={
        404: {"description": "product_id not found in the dataset"},
        422: {"description": "Invalid parameters"},
        503: {"description": "Index not ready"},
    },
)
def find_similar_products_endpoint(
    product_id: str = Query(..., min_length=1, description="uniq_id of the product"),
    num_similar: int = Query(
        10, ge=1, le=MAX_NUM_SIMILAR, description=f"How many to return (1-{MAX_NUM_SIMILAR})"
    ),
    backend: Literal["exact", "hnsw"] = Query(
        "exact", description="exact = brute force, hnsw = approximate nearest neighbour"
    ),
) -> SimilarProductsResponse:
    import time

    index = _require_index()
    started = time.perf_counter()

    try:
        results = index.search(product_id.strip(), num_similar, backend=backend)
    except ProductNotFoundError:
        # 404 rather than 400: the request was well-formed, the resource is
        # simply absent.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found. Try GET /sample_ids for valid IDs.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Search failed for %s", product_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during similarity search",
        ) from exc

    return SimilarProductsResponse(
        product_id=product_id,
        num_similar=num_similar,
        backend=backend,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=[SimilarProductOut(**vars(r)) for r in results],
    )


@app.get(
    "/find_similar_product_ids",
    response_model=ProductIdsResponse,
    summary="Same search, returning only IDs (the brief's List[str] contract)",
)
def find_similar_product_ids(
    product_id: str = Query(..., min_length=1),
    num_similar: int = Query(10, ge=1, le=MAX_NUM_SIMILAR),
) -> ProductIdsResponse:
    index = _require_index()
    try:
        results = index.search(product_id.strip(), num_similar)
    except ProductNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found",
        )
    return ProductIdsResponse(product_ids=[r.product_id for r in results])


# --------------------------------------------------------------------------
# Supporting endpoints
# --------------------------------------------------------------------------


@app.get("/products/{product_id}", summary="Fetch one product's attributes")
def get_product(product_id: str = Path(..., min_length=1)):
    index = _require_index()
    try:
        row = index.row_of(product_id.strip())
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

    product = index.df.iloc[row]
    import pandas as pd

    return {
        key: (None if pd.isna(value) else value)
        for key, value in product.items()
        if key != "colour_tokens" and not isinstance(value, (list, dict))
    }


@app.get("/sample_ids", summary="Valid product IDs for exploring the API")
def sample_ids(n: int = Query(5, ge=1, le=50)):
    index = _require_index()
    sample = index.df.sample(min(n, len(index.df)), random_state=None)
    return {
        "sample": [
            {"product_id": r["product_id"], "product_name": r["product_name"][:80]}
            for _, r in sample.iterrows()
        ]
    }


@app.get("/health", summary="Liveness probe")
def health():
    """Always 200 while the process is alive, even mid-build."""
    return {"status": "ok"}


@app.get("/ready", summary="Readiness probe")
def ready(response: Response):
    index = getattr(app.state, "index", None)
    if index is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready"}
    return {"status": "ready", "n_products": len(index)}


@app.get("/stats", summary="Index build statistics")
def stats():
    index = _require_index()
    return {
        "n_products": len(index),
        "build": {k: round(v, 3) for k, v in index.build_stats.items()},
        "blocks": [
            {"name": b.name, "dims": b.n_dims, "weight": round(b.weight, 3),
             "coverage_pct": round(float(b.present.mean()) * 100, 1)}
            for b in index.blocks
        ],
        "ann_enabled": index._ann is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
