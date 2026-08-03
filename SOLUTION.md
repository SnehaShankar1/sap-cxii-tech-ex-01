# Solution and Implementation

## 1. Data profiling

I profiled the dataset before writing any modelling code, in order to verify
that the attributes named in the brief were suitable for a similarity measure.
Three of them proved unreliable.

The `weight` field contains the literal string `999999999` in 23,745 of the
30,000 rows. This is a missing-data placeholder rather than a measurement,
leaving only 20.8% of the column usable.

The `colour` field is populated for 20.1% of rows, and the values are
pipe-delimited (for example `black|white`), making the field multi-valued
rather than a single category.

The `brand` field contains 6,337 distinct values, of which 55% appear exactly
once, and it is absent for 27% of rows.

Given that three of the four attributes emphasised in the brief were weak, I
carried out a second profiling pass to identify stronger signals.

## 2. Identifying a stronger similarity signal

The `sales_rank_in_child_category` column appears to hold a sales rank. Its
actual contents are a stringified dictionary:

```
"{'WomensKurtasKurtis': '#1793'}"
```

The dictionary key is a clean category label, with 207 distinct values across
83% of rows. The `browsenode` column provides comparable information at 98%
coverage.

Category is the most discriminative attribute available in this dataset, and
it is not referenced in the brief. No combination of price, rating and weight
establishes that a kurta belongs alongside another kurta rather than alongside
an accessory at a similar price point. Since the brief permits the use of
additional attributes, I extracted this field and assigned it the highest
weight.

To verify the effect, I queried a pair of men's shorts. All ten results were
men's shorts; socks and headwear at comparable price points were correctly
excluded.

## 3. Treatment of the unreliable fields

**Weight.** I retained the field but paired it with an explicit
missing-indicator column. Imputing a median across 79% of the column would
cause products to appear similar on weight purely because neither has a
recorded value, which would generate similarity from absence rather than
evidence. The indicator allows the model to distinguish a recorded weight from
an unknown one. The same treatment is applied to every numeric field with
gaps.

**Colour.** I split the values on the delimiter and encoded the resulting
tokens as multi-hot, so that `black|white` and `white|black` are not treated as
unrelated categories. The block carries the lowest weight, since at 20%
coverage it cannot contribute meaningfully to most comparisons.

**Brand.** I capped the encoding at the 300 most frequent brands and mapped the
remainder to an all-zero row rather than a shared "other" category. A shared
category would assert that two unrelated infrequent brands match one another,
which is less accurate than asserting nothing.

## 4. Feature block structure

Features are grouped into five blocks: category (weight 0.35), text (0.30),
numeric (0.20), brand (0.10) and colour (0.05). Each block is normalised
independently before the blocks are combined.

Independent normalisation is necessary because a single normalisation across
the concatenated vector would allow whichever block has the most dimensions or
the largest raw magnitudes to dominate the distance, irrespective of its
importance. The TF-IDF block alone would otherwise overwhelm the six numeric
columns.

The weights are derived from coverage and discriminative power. They are not
learned, as the dataset contains no interaction data from which to learn them.

For the text block I selected TF-IDF rather than transformer embeddings.
Product titles average eight words of dense keywords rather than prose, so
lexical overlap approximates semantic overlap adequately, and TF-IDF fits in
seconds on CPU. The limitation is synonymy: `kurta` and `kurti` are treated as
unrelated tokens. A sentence-transformer would handle this better and is the
first change I would make given more time.

The raw TF-IDF matrix was 30,000 × 4,096, approximately 500 MB when dense,
which is impractical for a container. Truncated SVD to 256 components reduced
the complete feature matrix from approximately 600 MB to 139 MB and reduced
build time by roughly a factor of three. The cost is interpretability: SVD
components have no nameable meaning, so individual matches cannot be explained
in terms of specific tokens. For a recommendation endpoint I considered this an
acceptable trade.

## 5. Scoring method

Rather than scoring each block separately and summing the results, I scale each
normalised block by the square root of its weight before concatenation. The
arithmetic resolves such that a single dot product over the combined vector
yields the weighted sum of per-block cosine similarities, and the resulting
rows remain unit length.

This reduces scoring to one matrix operation with no per-block bookkeeping at
query time, and allows Part 3 to use a single approximate index rather than one
index per block. A unit test asserts the unit-norm property, since the
correctness of the scores depends on it.

The trade-off is that weights are fixed at index build time; altering them
requires a rebuild. Maintaining separate indexes fused at query time would
permit per-request weighting but would multiply query cost by the number of
blocks. For a service that is read considerably more often than it is
reconfigured, fixing the weights at build time is the appropriate choice.

## 6. Handling absent feature blocks

Where a block is entirely absent for a product — for example a product with no
recorded colour — its weight is redistributed across the blocks that are
present for that row. The product is therefore scored on the evidence
available rather than penalised for a gap in the source data.

## 7. Service implementation

The function is exposed through FastAPI as `GET /find_similar_products`. The
endpoint returns 404 when the product ID is absent, 422 for invalid
parameters, and 503 when the index is unavailable.

The index is constructed during application startup rather than on first
request, so no request incurs the build cost.

I implemented liveness and readiness as separate endpoints. `/health` returns
200 throughout the build; had liveness failed during construction, the kubelet
would terminate the pod mid-build and the container would enter a permanent
crash-loop. `/ready` returns 503 until the index exists and is therefore the
endpoint that gates traffic.

I added `/sample_ids` because the dataset uses hashed identifiers, and without
it there is no practical way to obtain a valid ID for testing.

Results are cached using an LRU cache. The feature matrix is immutable for the
lifetime of the process, so results are fully cacheable. A production
deployment would use Redis so that the cache survives restarts and is shared
across replicas.

## 8. Container configuration

I made four changes to the supplied Dockerfile.

The original placed `COPY . /app` before `pip install`, which invalidates the
dependency layer on every source change and triggers a full reinstall. Copying
`requirements.txt` first resolves this.

`ENV NAME ProductSimilarityApp` uses the space-separated form, which Docker has
deprecated. This is now `ENV NAME=value`.

The container ran as root. Clusters enforcing Pod Security Standards with
`runAsNonRoot: true` will decline to schedule such a pod, so I added a
non-root user.

I converted the build to multi-stage so that the C++ toolchain required to
compile hnswlib remains in the builder stage and is excluded from the runtime
image.

The container runs a single worker deliberately: each worker would construct
its own 139 MB feature matrix and its own graph index, so scaling should be
achieved through replicas.

A deployment manifest with probes and resource requests is included in `k8s/`.
I wrote and reviewed both the Dockerfile and the manifest but was unable to
build or deploy them, as Docker Desktop could not be installed on the
development machine. The service itself is verified running under uvicorn
against the complete dataset, and the test suite passes.

## 9. Approximate nearest neighbour search

I implemented HNSW (Hierarchical Navigable Small World graphs; Malkov &
Yashunin, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320), 2016) using
`hnswlib`. The algorithm constructs a layered proximity graph in which sparse
upper layers provide long-range traversal and dense lower layers refine
locally, so query cost grows logarithmically rather than linearly with corpus
size.

I selected it over IVF, which requires a training pass to establish centroids
that constitutes pure overhead at 30,000 vectors, and over LSH, which requires
a substantial number of hash tables to achieve comparable recall.

FAISS was my initial choice, as suggested in the brief. It has no distributed
wheels for Python 3.13 and above on Windows and falls back to a source build
requiring SWIG, which I was unable to complete within the available time.
`hnswlib` implements the same algorithm with a considerably simpler
installation. At 10⁷ vectors and above I would return to FAISS for its
quantisation options and GPU support.

Measured across 300 queries at k=10:

| | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| exact | 17.98 ms | 15.03 | 42.62 | 57.65 |
| hnsw | 7.26 ms | 6.88 | 11.14 | 15.99 |

This is a 2.5× improvement with recall@10 of 1.0000, against a one-off build
cost of 17 seconds. Measurements were taken on a Windows development machine,
so absolute latencies are higher and the tail distribution wider than would be
expected in the container. The figures are reproducible via
`python benchmark.py`.

I should note that at 30,000 vectors approximate search is not required. Exact
search is already sufficiently fast, and 17 seconds of build time to save
approximately 10 ms per query is not self-evidently worthwhile. The
implementation demonstrates the scaling path: HNSW becomes advantageous in the
region of 10⁵ to 10⁶ vectors. The API defaults to exact search and exposes
`?backend=hnsw`, so both can be measured in deployment rather than assumed.

---

## Known limitations

**Rating reliability.** The `rating` field is fully populated, but
`no__of_reviews` is present for only 11.5% of rows. Mean rating is 4.27 where
review counts are known and 4.01 where they are not, indicating that
unreviewed ratings skew high: a 5.0 derived from a single review is treated
identically to a 4.2 derived from several hundred. Bayesian shrinkage toward
the global mean, weighted by review count, would be the appropriate correction.
I did not implement it.

**Absence of ground truth.** Recall@10 measures the fidelity with which HNSW
reproduces exact search. It does not measure recommendation quality. The only
evidence for the latter is qualitative inspection of results.

**Hand-set weights.** The block weights are informed by coverage and judgement
rather than learned from interaction data.

**No image features.** Image URLs are present for every row. The block
architecture would accommodate a sixth block with one additional builder
function and one additional weight, making this the most direct extension. CLIP
would be preferable to ResNet, as it places text and image representations in a
shared space.

**Near-duplicate listings** score close to 1.0 and can dominate the returned
set. A production system would apply maximal marginal relevance or a per-brand
diversity constraint.