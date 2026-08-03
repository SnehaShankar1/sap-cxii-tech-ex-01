# Product similarity search

SAP technical exercise. Given a product ID from the Amazon fashion catalogue
(30,000 items), return the most similar products.

The full write-up — why the features are what they are, what I found in the
data, what I'd change — is in **[SOLUTION.md](SOLUTION.md)**. The original
brief is preserved in [EXERCISE.md](EXERCISE.md).

## Running it

```bash
pip install -r requirements.txt
cd data && unzip archive.zip && cd ..
uvicorn app.main:app
```

Takes about a minute to start — it's building the feature matrix and the ANN
index. Once it's up, http://localhost:8000/docs gives you an interactive
interface. Call `/sample_ids` first to get a valid product ID, then paste it
into `/find_similar_products`.

Or from the command line:

```bash
curl "http://localhost:8000/sample_ids?n=3"
curl "http://localhost:8000/find_similar_products?product_id=<id>&num_similar=5"
```

## What you get back

Querying a pair of Jack & Jones men's shorts:

```
0.70  Checkersbay Men's Cotton Shorts (Pack of 3)     MensShorts   ₹898
0.67  Fitinc Combo Pack of 2 Lycra Shorts for Men     MensShorts   ₹749
0.66  BASICS Men's Cotton Shorts                      MensShorts   ₹599
0.66  Reebok Men's Shorts                             MensShorts  ₹1217
0.66  Diverse Men's Slim Fit Shorts                   MensShorts   ₹579
```

## Tests and benchmark

```bash
python -m pytest tests/ -q     # 46 tests
python benchmark.py            # exact vs HNSW
```

## Layout

```
app/
  config.py       weights and tunables — start here if you want to change behaviour
  data.py         loading and parsing the messy fields
  features.py     the five feature blocks
  similarity.py   search, exact and approximate
  main.py         FastAPI
tests/
k8s/              deployment manifest
explore.py        data profiling (run before building anything)
explore2.py       second profiling pass
benchmark.py
```

## Docker

```bash
docker build -t product-similarity:latest .
docker run -p 8000:8000 product-similarity:latest
```

The Dockerfile differs from the one supplied with the exercise in a few
places — layer caching, non-root user, multi-stage build. Reasons are in
SOLUTION.md.