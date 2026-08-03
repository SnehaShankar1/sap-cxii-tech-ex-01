"""
Central configuration.

Every tunable lives here so the similarity behaviour can be changed without
touching pipeline code, and so the README can point at one file when
explaining the weighting scheme.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_DATASET = (
    "marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson"
)

# Overridable so the Docker image can mount the data elsewhere.
DATASET_PATH = Path(os.getenv("DATASET_PATH", DATA_DIR / DEFAULT_DATASET))

# --------------------------------------------------------------------------
# Feature block weights
# --------------------------------------------------------------------------
# Rationale (see README):
#   category  - browsenode + child-category key. 100% / 83% coverage and the
#               single most discriminative signal: a kurta is more like another
#               kurta than like a wallet at the same price point.
#   text      - TF-IDF over product_name + meta_keywords. 100% coverage.
#               Implicitly carries brand, garment type, fabric, gender.
#   numeric   - price / rating / sales rank / discount. Cheap, always present,
#               but weakly discriminative on its own (many unrelated products
#               share a price).
#   brand     - 6,338 distinct values, 55% of them appearing exactly once.
#               Kept at low weight as an exact-match bonus only.
#   colour    - only 20% coverage, so it can never carry much of the score.
#
# Weights are relative; they are renormalised to sum to 1.0 at build time.
WEIGHTS = {
    "category": 0.35,
    "text": 0.30,
    "numeric": 0.20,
    "brand": 0.10,
    "colour": 0.05,
}

# --------------------------------------------------------------------------
# Feature engineering tunables
# --------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 4096
TFIDF_MIN_DF = 3          # ignore terms in fewer than 3 documents
TFIDF_NGRAM_RANGE = (1, 2)

# Truncated SVD (LSA) applied to the TF-IDF block.
# 4,096 raw TF-IDF dims x 30k rows is ~500 MB dense - too heavy for a
# container. 256 components retain the dominant semantic structure at ~6% of
# the memory, and denoise rare-term noise as a side effect. Set to 0 to
# disable and keep the raw TF-IDF space.
SVD_COMPONENTS = 256
SVD_RANDOM_STATE = 42

TOP_N_BRANDS = 300        # long tail beyond this maps to an all-zero vector
TOP_N_COLOURS = 60        # multi-hot over the most common colour tokens

# Weight values above this are the dataset's missing-data sentinel.
WEIGHT_SENTINEL_THRESHOLD = 1_000_000.0

# --------------------------------------------------------------------------
# ANN index (Part 3)
# --------------------------------------------------------------------------
HNSW_M = 32               # graph degree; higher = better recall, more memory
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128      # runtime candidate list; raise for better recall

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
MAX_NUM_SIMILAR = 100
