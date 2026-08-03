"""
Data exploration for the SAP similarity-search exercise.

Run this FIRST. Its output determines every feature-engineering decision
downstream, and the numbers it prints belong in your README as evidence.

Usage:
    python explore.py data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(path: str) -> pd.DataFrame:
    """Line-delimited JSON loader that survives malformed rows."""
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    print(f"Loaded {len(rows):,} rows  ({bad} unparseable lines skipped)\n")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Parsers for the messy fields
# --------------------------------------------------------------------------
_MONEY = re.compile(r"[\d,]+\.?\d*")


def parse_price(v):
    """'$29.99' -> 29.99 ; '$14.99 - $19.99' -> 17.49 (midpoint) ; junk -> None"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v)
    nums = [float(m.replace(",", "")) for m in _MONEY.findall(s)]
    if not nums:
        return None
    return sum(nums) / len(nums) if len(nums) > 1 else nums[0]


_WEIGHT_UNITS = {
    "ounce": 28.3495, "ounces": 28.3495, "oz": 28.3495,
    "pound": 453.592, "pounds": 453.592, "lb": 453.592, "lbs": 453.592,
    "gram": 1.0, "grams": 1.0, "g": 1.0,
    "kilogram": 1000.0, "kilograms": 1000.0, "kg": 1000.0,
    "milligram": 0.001, "mg": 0.001,
}
_WEIGHT_RE = re.compile(r"([\d.,]+)\s*([a-zA-Z]+)")


def parse_weight_grams(v):
    """'3.2 ounces' -> 90.7 ; '1.5 pounds' -> 680.4 ; unknown unit -> None"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    m = _WEIGHT_RE.search(str(v).lower())
    if not m:
        return None
    try:
        qty = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = m.group(2)
    factor = _WEIGHT_UNITS.get(unit)
    return qty * factor if factor else None


_RATING_RE = re.compile(r"([\d.]+)\s*out of\s*5")


def parse_rating(v):
    """'4.5 out of 5 stars' -> 4.5 ; bare '4.5' -> 4.5 ; junk -> None"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v)
    m = _RATING_RE.search(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    try:
        f = float(s.strip())
        return f if 0 <= f <= 5 else None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def null_report(df: pd.DataFrame):
    section("1. COLUMNS + NULL RATES  (paste this table into your README)")
    stats = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null_pct": (df.isna().mean() * 100).round(1),
    })
    # Treat empty strings as missing too - very common in this dump.
    blanks = df.apply(
        lambda c: (c.astype(str).str.strip().isin(["", "nan", "None"])).mean() * 100
    ).round(1)
    stats["blank_or_null_pct"] = blanks
    print(stats.sort_values("blank_or_null_pct").to_string())


def guess_columns(df: pd.DataFrame):
    """Find the likely candidates for each attribute the brief asks for."""
    section("2. CANDIDATE COLUMNS FOR THE REQUIRED ATTRIBUTES")
    wanted = {
        "id":     ["uniq_id", "asin", "sku"],
        "name":   ["product_name", "title"],
        "brand":  ["brand", "manufacturer"],
        "color":  ["color", "colour"],
        "price":  ["list_price", "retail_price", "price", "mrp"],
        "sale":   ["selling_price", "sales_price", "sale_price", "discounted_price"],
        "weight": ["shipping_weight", "weight", "item_weight"],
        "rating": ["rating", "average_rating", "stars", "product_rating"],
        "image":  ["image", "image_url", "images"],
        "text":   ["about_product", "product_description", "description",
                   "product_specification", "technical_details", "product_details"],
    }
    cols_lower = {c.lower(): c for c in df.columns}
    found = {}
    for role, options in wanted.items():
        hits = [cols_lower[o] for o in options if o in cols_lower]
        found[role] = hits
        status = ", ".join(hits) if hits else "!! NOT FOUND - inspect manually"
        print(f"  {role:<8} -> {status}")
    return found


def parse_quality(df: pd.DataFrame, found: dict):
    section("3. PARSE SUCCESS RATES  (can we actually USE these fields?)")
    checks = [
        ("price",  found["price"],  parse_price),
        ("sale",   found["sale"],   parse_price),
        ("weight", found["weight"], parse_weight_grams),
        ("rating", found["rating"], parse_rating),
    ]
    for role, cols, fn in checks:
        for col in cols:
            parsed = df[col].map(fn)
            ok = parsed.notna().mean() * 100
            print(f"\n  {col}  ({role})")
            print(f"    parsed OK : {ok:.1f}%")
            if parsed.notna().any():
                d = parsed.describe()
                print(f"    min/median/max : {d['min']:.2f} / "
                      f"{parsed.median():.2f} / {d['max']:.2f}")
            raw = df[col].dropna().astype(str)
            raw = raw[raw.str.strip() != ""]
            print(f"    raw samples: {list(raw.head(4))}")


def cardinality(df: pd.DataFrame, found: dict):
    section("4. CATEGORICAL CARDINALITY  (drives your one-hot strategy)")
    for role in ("brand", "color"):
        for col in found[role]:
            s = df[col].dropna().astype(str).str.strip().str.lower()
            s = s[s != ""]
            counts = s.value_counts()
            top10 = counts.head(10)
            coverage = counts.head(100).sum() / len(s) * 100 if len(s) else 0
            print(f"\n  {col}: {counts.nunique()} distinct values over {len(s):,} rows")
            print(f"    top-100 values cover {coverage:.1f}% of non-blank rows")
            print(f"    top 10: {dict(top10)}")


def text_fields(df: pd.DataFrame, found: dict):
    section("5. TEXT FIELD LENGTHS  (drives TF-IDF vs embedding choice)")
    for col in found["name"] + found["text"]:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        s = s[s.str.strip() != ""]
        if s.empty:
            print(f"  {col}: entirely empty")
            continue
        lens = s.str.split().str.len()
        print(f"  {col:<24} non-blank {len(s):>6,}  "
              f"median {int(lens.median()):>4} words  p95 {int(lens.quantile(.95)):>5}")


def id_integrity(df: pd.DataFrame, found: dict):
    section("6. ID INTEGRITY")
    for col in found["id"]:
        n, uniq = len(df), df[col].nunique()
        print(f"  {col}: {uniq:,} unique / {n:,} rows  "
              f"({'OK - unique' if uniq == n else 'WARNING: duplicates present'})")
        if df[col].isna().any():
            print(f"    !! {df[col].isna().sum()} null IDs")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    df = load(sys.argv[1])
    null_report(df)
    found = guess_columns(df)
    id_integrity(df, found)
    parse_quality(df, found)
    cardinality(df, found)
    text_fields(df, found)

    section("NEXT STEP")
    print("""
  Read section 3 carefully. Any field parsing below ~50% is a field you
  should either drop or treat as mostly-missing with an indicator flag -
  and you should say so explicitly in your README. That justification is
  worth more marks than silently imputing it.
""")


if __name__ == "__main__":
    main()