"""
Feature engineering.

Design
------
Attributes are grouped into five blocks - category, text, numeric, brand,
colour - rather than concatenated into one flat vector. Each block is
L2-normalised on its own, then scaled by sqrt(weight) before concatenation.

Why sqrt(weight):

    Given unit-norm blocks a_i, b_i and weights w_i summing to 1, define
        A = [sqrt(w_1)*a_1, ..., sqrt(w_k)*a_k]
    and B likewise. Then

        <A, B> = sum_i w_i * <a_i, b_i> = sum_i w_i * cos(a_i, b_i)

    and ||A||^2 = sum_i w_i * ||a_i||^2 = sum_i w_i = 1.

So a plain inner product over the concatenated vector *is* the weighted sum
of per-block cosine similarities, and the result is already unit-norm. One
matrix, one index, one dot product - no per-block bookkeeping at query time.

Trade-off: weights are baked in at build time, so changing them means a
rebuild (~20s on 30k rows). The alternative - one index per block, fused at
query time - allows per-request weights but multiplies query cost by the
number of blocks. For a read-heavy recommendation service the build-time
choice is the right one; see README.

Missing data
------------
A block that is entirely absent for a product produces a zero sub-vector.
That would silently deflate its score against every other product, so
`renormalise_rows` redistributes the weight across the blocks that *are*
present, per row. This is the fallback mechanism the brief asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from app.config import (
    SVD_COMPONENTS,
    SVD_RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TOP_N_BRANDS,
    TOP_N_COLOURS,
    WEIGHTS,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. All-zero rows are left as zeros."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _log1p_safe(values: np.ndarray) -> np.ndarray:
    """log1p on non-negative values; negatives clamped to 0 first."""
    return np.log1p(np.clip(values, 0.0, None))


@dataclass
class FeatureBlock:
    """One named group of columns, already L2-normalised row-wise."""

    name: str
    matrix: np.ndarray                  # (n_products, n_dims), float32
    weight: float
    present: np.ndarray = field(repr=False)  # (n_products,) bool

    @property
    def n_dims(self) -> int:
        return self.matrix.shape[1]


# --------------------------------------------------------------------------
# Block builders
# --------------------------------------------------------------------------


def build_category_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    One-hot over browsenode (302 values, 98% coverage) and child_category
    (207 values, 83% coverage), plus parent_category as a coarse backstop.

    Rationale: this is the strongest available signal. Two kurtas resemble
    each other more than a kurta resembles a wallet at the same price, and
    no combination of price/rating/weight expresses that. The brief does not
    list category as an attribute because it is not obvious from the column
    names - it has to be pulled out of a stringified dict.
    """
    parts = []
    for col in ("browsenode", "child_category", "parent_category"):
        dummies = pd.get_dummies(df[col].fillna("__missing__"), prefix=col)
        dummies = dummies.drop(
            columns=[c for c in dummies.columns if c.endswith("__missing__")],
            errors="ignore",
        )
        parts.append(dummies.to_numpy(dtype=np.float32))

    mat = np.hstack(parts) if parts else np.zeros((len(df), 0), np.float32)
    present = mat.any(axis=1)
    return mat, present


def build_text_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    TF-IDF over product_name + meta_keywords (100% coverage), reduced with
    truncated SVD.

    TF-IDF rather than transformer embeddings: titles here are ~8 words of
    dense keywords ("Cotton Kalamkari Handblock Saree Blouse"), not prose.
    Lexical overlap is close to semantic overlap in that regime, and TF-IDF
    costs seconds to fit rather than minutes of GPU-less inference. Bigrams
    are included so "slim fit" and "navy blue" survive as units.

    SVD (latent semantic analysis) then projects 4,096 sparse dims down to
    256 dense ones. Three reasons:

      1. Memory. 30k x 4,096 float32 dense is ~500 MB, which a modest pod
         cannot hold alongside the rest of the index. 256 dims is ~31 MB.
      2. Synonymy. Raw TF-IDF scores "kurta" and "kurti" as unrelated tokens.
         SVD places co-occurring terms on shared axes, so lexical variants
         partially collapse - a cheap approximation of semantic similarity.
      3. ANN performance. HNSW degrades in very high dimensions; 256 is a
         far friendlier space to build a proximity graph in.

    The cost is interpretability: individual SVD components have no
    human-readable meaning, so you can no longer point at a term and explain
    a match. For a recommendation endpoint that trade is worth taking.
    """
    vectoriser = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,          # damp the effect of repeated tokens
        dtype=np.float32,
    )
    sparse_mat = vectoriser.fit_transform(df["text"].fillna(""))
    present = np.asarray(sparse_mat.sum(axis=1)).ravel() > 0

    n_components = min(SVD_COMPONENTS, sparse_mat.shape[1] - 1)
    if SVD_COMPONENTS <= 0 or n_components <= 0:
        return sparse_mat.toarray().astype(np.float32), present, vectoriser

    svd = TruncatedSVD(n_components=n_components, random_state=SVD_RANDOM_STATE)
    reduced = svd.fit_transform(sparse_mat).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    reduced_meta = {"explained_variance": explained, "n_components": n_components}
    build_text_block.last_svd_info = reduced_meta  # surfaced in the README benchmark

    return reduced, present, vectoriser


def build_numeric_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardised numeric features with explicit missingness indicators.

    Price, list price, weight and sales rank are log-transformed first: all
    are strongly right-skewed (price spans 39 to 9,988 INR), and without the
    log a handful of expensive items dominate the Euclidean geometry.

    Median imputation fills gaps, but every imputed column also gets a binary
    "was missing" companion. Without it, two products that are both missing a
    price would look identical on that dimension - similarity manufactured
    out of absence.

    weight_grams is included despite only 20.8% coverage, precisely because
    the indicator makes its absence explicit rather than silently imputed.
    """
    numeric_specs = [
        ("sales_price", True),
        ("list_price", True),
        ("rating", False),
        ("discount_percentage", False),
        ("weight_grams", True),
        ("sales_rank", True),
    ]

    columns, indicators = [], []
    for col, use_log in numeric_specs:
        raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        missing = np.isnan(raw)

        if use_log:
            with np.errstate(invalid="ignore"):
                raw = np.where(missing, np.nan, _log1p_safe(raw))

        median = np.nanmedian(raw) if not np.all(missing) else 0.0
        columns.append(np.where(missing, median, raw))
        indicators.append(missing.astype(np.float64))

    # Binary flags need no imputation or scaling.
    flags = [
        df["is_prime"].fillna(0.0).to_numpy(dtype=np.float64),
        df["is_best_seller"].fillna(0.0).to_numpy(dtype=np.float64),
    ]

    scaled = StandardScaler().fit_transform(np.column_stack(columns))
    mat = np.column_stack([scaled] + indicators + flags).astype(np.float32)

    # Numeric features exist for every row (imputation guarantees it), but a
    # row where *everything* was missing carries no real information.
    all_missing = np.column_stack(indicators).all(axis=1)
    return mat, ~all_missing


def build_brand_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    One-hot over the most common brands only.

    6,337 distinct brands, 55% of which appear exactly once, and only 73% of
    rows have one at all. Full one-hot would add thousands of near-empty
    dimensions for almost no discriminative gain.

    Rare brands map to an all-zero row rather than a shared "other" bucket:
    an "other" bucket would assert that two unrelated obscure brands match,
    which is worse than asserting nothing. Cosine similarity between one-hot
    vectors is 1 for an exact brand match and 0 otherwise, so this block acts
    as a pure same-brand bonus.
    """
    brands = df["brand"].fillna("")
    top = brands[brands != ""].value_counts().head(TOP_N_BRANDS).index
    lookup = {b: i for i, b in enumerate(top)}

    mat = np.zeros((len(df), len(lookup)), dtype=np.float32)
    for row, brand in enumerate(brands):
        idx = lookup.get(brand)
        if idx is not None:
            mat[row, idx] = 1.0

    return mat, mat.any(axis=1)


def build_colour_block(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-hot over the most common colour tokens.

    colour is present for only 20% of rows and is pipe-delimited
    ("black|white", median 3 tokens, up to 80). It is multi-label, so
    multi-hot over split tokens rather than one-hot over raw strings -
    "black|white" and "white|black" must not be treated as unrelated.

    Low coverage is why this block carries the smallest weight: for 80% of
    products it contributes nothing and its weight is redistributed.
    """
    token_counts: dict[str, int] = {}
    for tokens in df["colour_tokens"]:
        for token in tokens or []:
            token_counts[token] = token_counts.get(token, 0) + 1

    top = sorted(token_counts, key=token_counts.get, reverse=True)[:TOP_N_COLOURS]
    lookup = {t: i for i, t in enumerate(top)}

    mat = np.zeros((len(df), len(lookup)), dtype=np.float32)
    for row, tokens in enumerate(df["colour_tokens"]):
        for token in tokens or []:
            idx = lookup.get(token)
            if idx is not None:
                mat[row, idx] = 1.0

    return mat, mat.any(axis=1)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def renormalise_rows(blocks: list[FeatureBlock]) -> np.ndarray:
    """
    Concatenate blocks with sqrt(weight) scaling, redistributing the weight
    of any block that is absent for a given row across that row's present
    blocks.

    Returns an (n_products, total_dims) float32 matrix whose rows are unit
    norm, so a plain dot product yields the weighted cosine score directly.
    """
    n_rows = blocks[0].matrix.shape[0]
    presence = np.column_stack([b.present for b in blocks])            # (n, k)
    weights = np.array([b.weight for b in blocks], dtype=np.float64)   # (k,)

    # Per-row weight mass that is actually available.
    available = presence @ weights                                     # (n,)
    # A row with nothing present cannot be scored; avoid divide-by-zero and
    # let it fall out of results naturally as an all-zero vector.
    available[available == 0.0] = 1.0

    effective = (presence * weights) / available[:, None]               # (n, k)
    scale = np.sqrt(effective).astype(np.float32)

    return np.hstack(
        [b.matrix * scale[:, i : i + 1] for i, b in enumerate(blocks)]
    ).astype(np.float32)


def build_feature_matrix(
    df: pd.DataFrame, weights: Optional[dict[str, float]] = None
) -> tuple[np.ndarray, list[FeatureBlock]]:
    """
    Build the full feature matrix.

    Returns the matrix plus the individual blocks, so tests and the README
    benchmark can inspect per-block contributions.
    """
    weights = dict(weights or WEIGHTS)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Feature weights must sum to a positive value")
    weights = {k: v / total for k, v in weights.items()}

    category_mat, category_present = build_category_block(df)
    text_mat, text_present, _ = build_text_block(df)
    numeric_mat, numeric_present = build_numeric_block(df)
    brand_mat, brand_present = build_brand_block(df)
    colour_mat, colour_present = build_colour_block(df)

    blocks = [
        FeatureBlock("category", l2_normalise(category_mat), weights["category"], category_present),
        FeatureBlock("text", l2_normalise(text_mat), weights["text"], text_present),
        FeatureBlock("numeric", l2_normalise(numeric_mat), weights["numeric"], numeric_present),
        FeatureBlock("brand", l2_normalise(brand_mat), weights["brand"], brand_present),
        FeatureBlock("colour", l2_normalise(colour_mat), weights["colour"], colour_present),
    ]

    return renormalise_rows(blocks), blocks
