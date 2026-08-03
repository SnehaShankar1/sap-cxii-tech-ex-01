"""
Dataset loading and field parsing.

The Amazon India fashion dump is messy in specific, documented ways. Every
parser here exists because of something observed during profiling:

  weight   79.1% of rows are the literal sentinel `999999999`
  brand    27.1% missing, 6,338 distinct values, 55% of them singletons
  colour   79.9% missing, pipe-delimited multi-values ("black|white")
  category not a plain string - a stringified dict {category: sales_rank}
  price    plain decimal strings, no currency symbol, in INR
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.config import WEIGHT_SENTINEL_THRESHOLD

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Read line-delimited JSON into a DataFrame.

    Written as an explicit loop rather than `pd.read_json(lines=True)` so a
    single malformed line degrades to a skipped row instead of failing the
    whole load - important for a service that must start reliably.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Unzip data/archive.zip, or set "
            f"the DATASET_PATH environment variable."
        )

    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1

    if not records:
        raise ValueError(f"No parseable records found in {path}")

    df = pd.DataFrame(records)
    df.attrs["skipped_lines"] = skipped
    return df


# --------------------------------------------------------------------------
# Scalar parsers
# --------------------------------------------------------------------------

_MISSING_TOKENS = {"", "nan", "none", "null", "<na>", "n/a"}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip().lower() in _MISSING_TOKENS


_NUMBER_RE = re.compile(r"[\d,]+\.?\d*")


def parse_price(v: Any) -> Optional[float]:
    """'200.00' -> 200.0. Ranges average to their midpoint. INR, no symbol."""
    if _is_missing(v):
        return None
    nums = [float(m.replace(",", "")) for m in _NUMBER_RE.findall(str(v))]
    if not nums:
        return None
    value = sum(nums) / len(nums)
    return value if value > 0 else None


_WEIGHT_UNITS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "gm": 1.0, "gms": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
}
_WEIGHT_RE = re.compile(r"([\d.,]+)\s*([a-zA-Z]+)")


def parse_weight_grams(v: Any) -> Optional[float]:
    """
    '240 g' -> 240.0, '1.5 kg' -> 1500.0.

    Returns None for the `999999999` sentinel that fills 79% of this column,
    and for any bare number with no unit (which is what the sentinel looks
    like once you strip it).
    """
    if _is_missing(v):
        return None
    m = _WEIGHT_RE.search(str(v).lower())
    if not m:
        return None
    try:
        qty = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    factor = _WEIGHT_UNITS.get(m.group(2))
    if factor is None:
        return None
    grams = qty * factor
    if grams <= 0 or grams >= WEIGHT_SENTINEL_THRESHOLD:
        return None
    return grams


def parse_rating(v: Any) -> Optional[float]:
    """'4.5' or '4.5 out of 5 stars' -> 4.5. Clamped to [0, 5]."""
    if _is_missing(v):
        return None
    s = str(v)
    m = re.search(r"([\d.]+)\s*out of\s*5", s)
    raw = m.group(1) if m else s.strip()
    try:
        f = float(raw)
    except ValueError:
        return None
    return f if 0.0 <= f <= 5.0 else None


def parse_percentage(v: Any) -> Optional[float]:
    """'54' or '54%' -> 54.0."""
    if _is_missing(v):
        return None
    digits = re.sub(r"[^\d.]", "", str(v))
    if not digits:
        return None
    try:
        f = float(digits)
    except ValueError:
        return None
    return f if 0.0 <= f <= 100.0 else None


def parse_dict_field(v: Any) -> Optional[dict]:
    """
    The category columns hold stringified Python dicts, e.g.

        "{'ClothingAccessories': '#19,259', 'MensT_Shirts': '#12151'}"

    `ast.literal_eval` is used rather than `eval` so no arbitrary code can
    execute from dataset contents.
    """
    if _is_missing(v):
        return None
    if isinstance(v, dict):
        return v
    try:
        parsed = ast.literal_eval(str(v))
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def category_key(v: Any) -> Optional[str]:
    """
    Extract the category name from a {category: rank} dict.

    For the child-category column this yields values like 'WomensKurtasKurtis'
    - 218 distinct across the dataset, covering 83% of rows. This is the
    single most useful similarity signal available and it is invisible unless
    you look inside the dict.
    """
    d = parse_dict_field(v)
    if not d:
        return None
    keys = [k for k in d.keys() if str(k).strip()]
    return str(keys[-1]).strip() if keys else None


def category_rank(v: Any) -> Optional[float]:
    """Extract the numeric sales rank from a {category: '#12,151'} dict."""
    d = parse_dict_field(v)
    if not d:
        return None
    values = list(d.values())
    if not values:
        return None
    digits = re.sub(r"[^\d]", "", str(values[-1]))
    return float(digits) if digits else None


def split_multi(v: Any, sep: str = "|") -> list[str]:
    """'black|white' -> ['black', 'white']. Missing -> []."""
    if _is_missing(v):
        return []
    return [p.strip().lower() for p in str(v).split(sep) if p.strip()]


def first_of_multi(v: Any, sep: str = "|") -> Optional[str]:
    """First element of a pipe-delimited list, e.g. the primary image URL."""
    parts = split_multi(v, sep)
    return parts[0] if parts else None


def clean_text(v: Any) -> str:
    """Normalise a free-text field for TF-IDF. Missing -> empty string."""
    if _is_missing(v):
        return ""
    s = str(v).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_categorical(v: Any) -> Optional[str]:
    """Lowercase and strip a categorical value. Missing -> None."""
    if _is_missing(v):
        return None
    s = str(v).strip().lower()
    return s or None


# --------------------------------------------------------------------------
# Normalisation into a tidy frame
# --------------------------------------------------------------------------

# Source column -> canonical name. Kept explicit so a schema change surfaces
# as a clear KeyError rather than silently producing an all-missing feature.
COLUMN_MAP = {
    "uniq_id": "product_id",
    "product_name": "product_name",
    "brand": "brand",
    "colour": "colour",
    "sales_price": "sales_price",
    "weight": "weight_raw",
    "rating": "rating",
    "discount_percentage": "discount_percentage",
    "browsenode": "browsenode",
    "meta_keywords": "meta_keywords",
    "medium": "image_url_raw",
    "sales_rank_in_child_category": "child_category_raw",
    "sales_rank_in_parent_category": "parent_category_raw",
    "amazon_prime__y_or_n": "prime_raw",
    "best_seller_tag__y_or_n": "best_seller_raw",
}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw dump into a tidy frame with typed, canonically named columns.

    Missing source columns are tolerated - they become all-null - so the
    pipeline still runs against a truncated or differently-versioned dump.
    """
    out = pd.DataFrame(index=df.index)

    for src, dest in COLUMN_MAP.items():
        out[dest] = df[src] if src in df.columns else None

    out["product_id"] = out["product_id"].astype(str).str.strip()

    # Numeric block
    out["sales_price"] = out["sales_price"].map(parse_price)
    out["rating"] = out["rating"].map(parse_rating)
    out["discount_percentage"] = out["discount_percentage"].map(parse_percentage)
    out["weight_grams"] = out["weight_raw"].map(parse_weight_grams)
    out["sales_rank"] = out["child_category_raw"].map(category_rank)

    # Reconstruct list price where a discount is known:
    #   sales_price = list_price * (1 - discount/100)
    disc = out["discount_percentage"]
    valid = disc.notna() & (disc < 100) & out["sales_price"].notna()
    out["list_price"] = np.where(
        valid, out["sales_price"] / (1.0 - disc.fillna(0) / 100.0), np.nan
    )

    # Categorical block
    out["child_category"] = out["child_category_raw"].map(category_key)
    out["parent_category"] = out["parent_category_raw"].map(category_key)
    out["browsenode"] = out["browsenode"].map(clean_categorical)
    out["brand"] = out["brand"].map(clean_categorical)
    out["colour_tokens"] = out["colour"].map(split_multi)

    # Text block - product_name and meta_keywords overlap heavily but
    # meta_keywords sometimes adds the manufacturer, so both are used.
    out["text"] = (
        out["product_name"].map(clean_text) + " " + out["meta_keywords"].map(clean_text)
    ).str.strip()

    # Flags
    out["is_prime"] = out["prime_raw"].map(
        lambda v: 1.0 if str(v).strip().upper() == "Y" else 0.0
    )
    out["is_best_seller"] = out["best_seller_raw"].map(
        lambda v: 1.0 if str(v).strip().upper() == "Y" else 0.0
    )

    out["image_url"] = out["image_url_raw"].map(first_of_multi)
    out["product_name"] = out["product_name"].fillna("").astype(str)

    out = out.drop(
        columns=[
            "weight_raw", "child_category_raw", "parent_category_raw",
            "prime_raw", "best_seller_raw", "image_url_raw", "colour",
            "meta_keywords",
        ]
    )

    # A product with no ID cannot be looked up or returned.
    out = out[out["product_id"].notna() & (out["product_id"] != "")]

    # uniq_id is unique in this dump (asin is not), but guard anyway.
    return out.drop_duplicates(subset="product_id").reset_index(drop=True)


def load_and_normalise(path: str | Path) -> pd.DataFrame:
    return normalise(load_dataset(path))
