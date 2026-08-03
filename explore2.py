"""
Round 2 profiling - targets the specific unknowns from round 1.

Fixes the cardinality bug in explore.py and answers:
  - how bad is `weight` really (sentinels, units)?
  - true brand / colour cardinality
  - is `rating` trustworthy given missing review counts?
  - what is inside meta_keywords and product_details__k_v_pairs?

Usage:
    python explore2.py data/marketing_sample_...ldjson
"""

import json
import re
import sys
from collections import Counter

import pandas as pd

pd.set_option("display.width", 160)


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows)


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def blank(s):
    """Series of booleans: is this value missing OR empty-ish?"""
    return s.isna() | s.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])


# --------------------------------------------------------------------------
def raw_records(df):
    section("A. TWO RAW RECORDS  (what does a row actually look like?)")
    for i in (0, 1):
        print(f"\n--- record {i} ---")
        for k, v in df.iloc[i].items():
            s = str(v)
            if len(s) > 110:
                s = s[:110] + " ...[truncated]"
            print(f"  {k:<32} {s}")


def true_cardinality(df):
    section("B. TRUE CARDINALITY  (round-1 numbers were wrong)")
    for col in ["brand", "colour"]:
        if col not in df.columns:
            continue
        s = df[col][~blank(df[col])].astype(str).str.strip().str.lower()
        counts = s.value_counts()
        n = len(counts)
        print(f"\n  {col}: {n:,} distinct values over {len(s):,} non-blank rows "
              f"({len(s)/len(df)*100:.1f}% coverage)")
        for k in (20, 50, 100, 200, 500):
            if k <= n:
                print(f"    top-{k:<4} covers {counts.head(k).sum()/len(s)*100:5.1f}% of non-blank rows")
        print(f"    singletons (appear once): {(counts == 1).sum():,} "
              f"({(counts == 1).sum()/n*100:.1f}% of distinct values)")


def colour_tokens(df):
    section("C. COLOUR IS PIPE-DELIMITED  (multi-label, not single-category)")
    if "colour" not in df.columns:
        return
    s = df["colour"][~blank(df["colour"])].astype(str).str.lower()
    tokens = Counter()
    per_row = []
    for v in s:
        parts = [p.strip() for p in v.split("|") if p.strip()]
        per_row.append(len(parts))
        tokens.update(parts)
    print(f"  rows with colour      : {len(s):,}")
    print(f"  distinct raw strings  : {s.nunique():,}")
    print(f"  distinct TOKENS       : {len(tokens):,}   <-- encode on these")
    print(f"  tokens per row        : median {pd.Series(per_row).median():.0f}, "
          f"max {max(per_row)}")
    print(f"  top 25 tokens         : {dict(tokens.most_common(25))}")


def weight_reality(df):
    section("D. WEIGHT - HOW BAD IS IT?")
    if "weight" not in df.columns:
        return
    s = df["weight"].astype(str).str.strip().str.lower()
    n = len(s)
    sentinel = s.str.fullmatch(r"9{6,}(\.0+)?").sum()
    print(f"  total rows                  : {n:,}")
    print(f"  '999999999'-style sentinels : {sentinel:,} ({sentinel/n*100:.1f}%)")
    print(f"  blank/empty                 : {blank(df['weight']).sum():,}")

    units = Counter()
    for v in s:
        m = re.search(r"[\d.,]+\s*([a-z]+)", v)
        units[m.group(1) if m else "(no unit)"] += 1
    print(f"  unit tokens found           : {dict(units.most_common(12))}")

    print("\n  20 sample values that are NOT sentinels:")
    good = s[~s.str.fullmatch(r"9{6,}(\.0+)?")].head(20).tolist()
    print(f"    {good}")


def rating_trust(df):
    section("E. IS RATING TRUSTWORTHY?")
    if "rating" not in df.columns:
        return
    r = pd.to_numeric(df["rating"], errors="coerce")
    print(f"  parsed              : {r.notna().sum():,}")
    print(f"  distribution        :")
    print(r.describe().to_string().replace("\n", "\n      "))
    print(f"\n  value counts (top 12):")
    print(f"    {dict(r.value_counts().head(12))}")

    if "no__of_reviews" in df.columns:
        nr = pd.to_numeric(
            df["no__of_reviews"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce",
        )
        has = nr.notna() & (nr > 0)
        print(f"\n  rows with a usable review count : {has.sum():,} "
              f"({has.sum()/len(df)*100:.1f}%)")
        if has.any():
            print(f"  review count median/max         : {nr[has].median():.0f} / {nr[has].max():.0f}")
            print(f"  mean rating WHERE reviews known : {r[has].mean():.2f}")
            print(f"  mean rating WHERE reviews UNknown: {r[~has].mean():.2f}")
            print("\n  ^ if these two differ a lot, unreviewed ratings are noise.")


def text_signals(df):
    section("F. TEXT FIELDS AVAILABLE FOR TF-IDF")
    for col in ["product_name", "meta_keywords", "browsenode",
                "parent___child_category__all", "product_details__k_v_pairs"]:
        if col not in df.columns:
            continue
        s = df[col][~blank(df[col])].astype(str)
        if s.empty:
            print(f"  {col:<32} EMPTY")
            continue
        lens = s.str.split().str.len()
        print(f"  {col:<32} {len(s):>6,} rows  median {int(lens.median()):>4} words  "
              f"p95 {int(lens.quantile(.95)):>5}")
        print(f"      sample: {s.iloc[0][:150]}")


def category_field(df):
    section("G. CATEGORY - THE STRONGEST SIMILARITY SIGNAL YOU HAVE")
    col = "parent___child_category__all"
    if col not in df.columns:
        print("  not present")
        return
    s = df[col][~blank(df[col])].astype(str)
    print(f"  coverage        : {len(s):,} / {len(df):,} ({len(s)/len(df)*100:.1f}%)")
    print(f"  distinct values : {s.nunique():,}")
    print(f"  top 15          :")
    for k, v in s.value_counts().head(15).items():
        print(f"      {v:>6,}  {k[:100]}")


def images(df):
    section("H. IMAGE URLS (for the optional multimodal extension)")
    for col in ["image_urls__small", "medium", "large"]:
        if col not in df.columns:
            continue
        s = df[col][~blank(df[col])].astype(str)
        print(f"  {col:<22} {len(s):>6,} rows ({len(s)/len(df)*100:.1f}%)")
        if not s.empty:
            print(f"      sample: {s.iloc[0][:130]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    df = load(sys.argv[1])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    raw_records(df)
    true_cardinality(df)
    colour_tokens(df)
    weight_reality(df)
    rating_trust(df)
    text_signals(df)
    category_field(df)
    images(df)


if __name__ == "__main__":
    main()