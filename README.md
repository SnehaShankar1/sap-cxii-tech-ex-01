# Product Similarity Search

This project implements a product similarity search service for the **SAP technical exercise**. Given a product ID from the Amazon Fashion catalogue (approximately **30,000 products**), it returns the most similar products using a feature-based similarity model.

A detailed explanation of the approach, feature engineering decisions, data quality observations, assumptions, limitations, and possible improvements is available in **[SOLUTION.md](SOLUTION.md)**.

The original assessment brief is preserved in **[EXERCISE.md](EXERCISE.md)**.

---

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Extract the dataset

```bash
cd data
unzip archive.zip
cd ..
```

### Start the application

```bash
uvicorn app.main:app
```

> **Note:** The application takes approximately one minute to start because it builds the feature matrix and the Approximate Nearest Neighbor (HNSW) index during initialization.

Once the server is running, open:

```
http://localhost:8000/docs
```

to access the interactive Swagger UI.

---

## Using the API

Retrieve a few valid product IDs:

```bash
curl "http://localhost:8000/sample_ids?n=3"
```

Then use one of those IDs to search for similar products:

```bash
curl "http://localhost:8000/find_similar_products?product_id=<id>&num_similar=5"
```

---

## Example Response

Querying a pair of **Jack & Jones Men's Shorts** returns similar products such as:

```text
0.70  Checkersbay Men's Cotton Shorts (Pack of 3)     MensShorts   ₹898
0.67  Fitinc Combo Pack of 2 Lycra Shorts for Men     MensShorts   ₹749
0.66  BASICS Men's Cotton Shorts                      MensShorts   ₹599
0.66  Reebok Men's Shorts                             MensShorts  ₹1217
0.66  Diverse Men's Slim Fit Shorts                   MensShorts   ₹579
```

The score on the left represents the similarity between the query product and the returned product.

---

## Running Tests

Run the test suite:

```bash
python -m pytest tests/ -q
```

The project includes **46 automated tests** covering data loading, feature engineering, similarity search, and API functionality.

To compare the performance of exact search and the HNSW-based approximate search:

```bash
python benchmark.py
```

---

## Project Structure

```text
app/
├── config.py       # Feature weights and configurable parameters
├── data.py         # Data loading and preprocessing
├── features.py     # Feature engineering pipeline
├── similarity.py   # Exact and approximate similarity search
└── main.py         # FastAPI application

tests/              # Unit tests
k8s/                # Kubernetes deployment manifests

explore.py          # Initial data exploration
explore2.py         # Additional profiling and analysis
benchmark.py        # Exact vs. HNSW benchmark
```

---

## Docker

Build the Docker image:

```bash
docker build -t product-similarity:latest .
```

Run the container:

```bash
docker run -p 8000:8000 product-similarity:latest
```

Compared to the Dockerfile supplied with the assessment, this implementation includes several improvements, including:

- Multi-stage builds
- Better Docker layer caching
- Running the application as a non-root user

The reasoning behind these changes is documented in **SOLUTION.md**.
