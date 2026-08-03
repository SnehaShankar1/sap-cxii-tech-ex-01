"""
Benchmark exact vs HNSW search, and report recall.

Every performance number quoted in the README comes from running this. An
approximate index without a measured recall figure is an unverified claim.

Usage:
    python benchmark.py
    python benchmark.py --queries 500 --k 10
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from app.similarity import SimilarityIndex


def time_queries(index: SimilarityIndex, ids: list[str], k: int, backend: str) -> dict:
    latencies = []
    for pid in ids:
        started = time.perf_counter()
        index.search(pid, k, backend=backend)
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()
    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--recall-sample", type=int, default=200)
    args = parser.parse_args()

    print("Building index...")
    started = time.perf_counter()
    index = SimilarityIndex.from_dataset()
    build_seconds = time.perf_counter() - started

    print("\n" + "=" * 68)
    print("INDEX")
    print("=" * 68)
    for key, value in index.build_stats.items():
        print(f"  {key:<22} {value:,.2f}")
    print(f"  {'total_build_seconds':<22} {build_seconds:,.2f}")

    print("\n  feature blocks:")
    for block in index.blocks:
        print(
            f"    {block.name:<10} dims={block.n_dims:<5} "
            f"weight={block.weight:.2f}  coverage={block.present.mean()*100:5.1f}%"
        )

    rng = np.random.default_rng(0)
    rows = rng.choice(len(index), size=min(args.queries, len(index)), replace=False)
    ids = [index.df.iloc[r]["product_id"] for r in rows]

    print("\n" + "=" * 68)
    print(f"LATENCY  ({len(ids)} queries, k={args.k})")
    print("=" * 68)

    results = {}
    for backend in ("exact", "hnsw"):
        if backend == "hnsw" and index._ann is None:
            print("  hnsw: unavailable (hnswlib not installed)")
            continue
        # Warm caches so the first query does not skew the mean.
        index.search(ids[0], args.k, backend=backend)
        stats = time_queries(index, ids, args.k, backend)
        results[backend] = stats
        print(
            f"  {backend:<6} mean {stats['mean_ms']:6.2f} ms   "
            f"p50 {stats['p50_ms']:6.2f}   p95 {stats['p95_ms']:6.2f}   "
            f"p99 {stats['p99_ms']:6.2f}"
        )

    if "hnsw" in results:
        speedup = results["exact"]["mean_ms"] / results["hnsw"]["mean_ms"]
        print(f"\n  speedup: {speedup:.1f}x")

        print("\n" + "=" * 68)
        print(f"RECALL@{args.k}  (HNSW vs exact, {args.recall_sample} queries)")
        print("=" * 68)
        recall = index.recall_at_k(args.k, args.recall_sample)
        print(f"  recall@{args.k}: {recall:.4f}")
        print(
            "\n  At 30k vectors exact search is already fast enough; HNSW is\n"
            "  included to demonstrate the scaling path. The graph pays for\n"
            "  itself from roughly 10^5-10^6 vectors, where exact search grows\n"
            "  linearly and HNSW grows logarithmically."
        )


if __name__ == "__main__":
    main()
