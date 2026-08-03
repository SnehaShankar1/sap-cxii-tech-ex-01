"""
Tests for the similarity search and the API.

A small synthetic fixture is used instead of the 30k-row dataset so the suite
runs in seconds and stays deterministic. Two integration tests exercise the
real data and are skipped automatically when it is absent - CI should not
fail merely because the archive was not unzipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import DATASET_PATH
from app.data import (
    category_key,
    category_rank,
    normalise,
    parse_percentage,
    parse_price,
    parse_rating,
    parse_weight_grams,
    split_multi,
)
from app.features import build_feature_matrix, l2_normalise
from app.similarity import (
    ProductNotFoundError,
    SimilarityIndex,
    calculate_similarity,
    set_index,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _raw_row(i: int, **overrides) -> dict:
    row = {
        "uniq_id": f"id{i:04d}",
        "product_name": f"Test Cotton Kurta {i}",
        "brand": "biba",
        "colour": "black|white",
        "sales_price": "500.00",
        "weight": "999999999",
        "rating": "4.0",
        "discount_percentage": "50",
        "browsenode": "1968255031",
        "meta_keywords": "kurta cotton women ethnic",
        "medium": "https://example.com/a.jpg|https://example.com/b.jpg",
        "sales_rank_in_child_category": "{'WomensKurtasKurtis': '#1793'}",
        "sales_rank_in_parent_category": "{'ClothingAccessories': '#19,259'}",
        "amazon_prime__y_or_n": "N",
        "best_seller_tag__y_or_n": "N",
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="module")
def small_index() -> SimilarityIndex:
    """60 products across two categories, with deliberate edge cases."""
    rows = []
    for i in range(30):
        rows.append(_raw_row(i))
    for i in range(30, 60):
        rows.append(
            _raw_row(
                i,
                product_name=f"Test Mens Slim Fit T-Shirt {i}",
                brand="pantaloons",
                browsenode="1968123031",
                sales_rank_in_child_category="{'MensT_Shirts': '#12151'}",
                sales_price="250.00",
                meta_keywords="tshirt men slim fit cotton",
            )
        )
    # Edge cases: everything missing, and a real weight.
    rows.append(
        _raw_row(
            99,
            brand=None,
            colour=None,
            sales_price=None,
            discount_percentage=None,
            sales_rank_in_child_category=None,
            weight="240 g",
        )
    )

    df = normalise(pd.DataFrame(rows))
    matrix, blocks = build_feature_matrix(df)
    return SimilarityIndex(df, matrix, blocks, build_ann=True)


@pytest.fixture()
def client(small_index):
    from app.main import app

    set_index(small_index)
    with TestClient(app) as test_client:
        test_client.app.state.index = small_index
        yield test_client
    set_index(None)


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------


class TestParsers:
    def test_price_plain(self):
        assert parse_price("200.00") == 200.0

    def test_price_with_thousands_separator(self):
        assert parse_price("1,299.00") == 1299.0

    def test_price_range_averages(self):
        assert parse_price("$14.99 - $19.99") == pytest.approx(17.49)

    @pytest.mark.parametrize("value", [None, "", "nan", "abc", float("nan")])
    def test_price_missing(self, value):
        assert parse_price(value) is None

    def test_weight_sentinel_rejected(self):
        """79% of this column is the literal 999999999 placeholder."""
        assert parse_weight_grams("999999999") is None

    def test_weight_unit_conversion(self):
        assert parse_weight_grams("240 g") == 240.0
        assert parse_weight_grams("1.5 kg") == 1500.0
        assert parse_weight_grams("3.2 ounces") == pytest.approx(90.7184)

    def test_weight_without_unit_rejected(self):
        assert parse_weight_grams("500") is None

    def test_rating_variants(self):
        assert parse_rating("4.5") == 4.5
        assert parse_rating("4.5 out of 5 stars") == 4.5

    def test_rating_out_of_range_rejected(self):
        assert parse_rating("11.0") is None

    def test_percentage(self):
        assert parse_percentage("54") == 54.0
        assert parse_percentage("54%") == 54.0
        assert parse_percentage("150") is None

    def test_category_key_from_dict_string(self):
        assert category_key("{'WomensKurtasKurtis': '#1793'}") == "WomensKurtasKurtis"

    def test_category_rank_strips_formatting(self):
        assert category_rank("{'ClothingAccessories': '#19,259'}") == 19259.0

    def test_category_key_survives_malformed_input(self):
        assert category_key("not a dict") is None

    def test_split_multi_is_order_insensitive_set(self):
        assert set(split_multi("black|white")) == set(split_multi("white|black"))

    def test_split_multi_missing(self):
        assert split_multi(None) == []


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


class TestFeatures:
    def test_rows_are_unit_norm(self, small_index):
        """
        The sqrt(weight) construction only yields weighted cosine if every
        row has unit norm. This is the load-bearing invariant.
        """
        norms = np.linalg.norm(small_index.matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_l2_normalise_leaves_zero_rows_alone(self):
        mat = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        out = l2_normalise(mat)
        assert np.allclose(out[0], 0.0)
        assert np.isclose(np.linalg.norm(out[1]), 1.0)

    def test_missing_attributes_do_not_crash_the_build(self, small_index):
        assert "id0099" in small_index

    def test_blocks_have_expected_names(self, small_index):
        names = [b.name for b in small_index.blocks]
        assert names == ["category", "text", "numeric", "brand", "colour"]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class TestSearch:
    def test_returns_requested_count(self, small_index):
        assert len(small_index.search("id0000", 5)) == 5

    def test_excludes_the_query_product(self, small_index):
        results = small_index.search("id0000", 10)
        assert all(r.product_id != "id0000" for r in results)

    def test_unknown_id_raises(self, small_index):
        with pytest.raises(ProductNotFoundError):
            small_index.search("does-not-exist", 5)

    def test_scores_are_descending(self, small_index):
        scores = [r.score for r in small_index.search("id0000", 10)]
        assert scores == sorted(scores, reverse=True)

    def test_same_category_ranks_above_other_category(self, small_index):
        """The core quality claim: category dominates the ranking."""
        results = small_index.search("id0000", 10)
        assert all(r.child_category == "WomensKurtasKurtis" for r in results)

    def test_results_are_deterministic(self, small_index):
        """Near-duplicate listings make ties common; ordering must be stable."""
        first = [r.product_id for r in small_index.search("id0000", 10)]
        second = [r.product_id for r in small_index.search("id0000", 10)]
        assert first == second

    def test_zero_requested_returns_empty(self, small_index):
        assert small_index.search("id0000", 0) == []

    def test_num_similar_larger_than_corpus(self, small_index):
        results = small_index.search("id0000", 10_000)
        assert len(results) == len(small_index) - 1

    def test_ann_matches_exact_on_small_corpus(self, small_index):
        exact = {r.product_id for r in small_index.search("id0000", 10, backend="exact")}
        approx = {r.product_id for r in small_index.search("id0000", 10, backend="hnsw")}
        assert len(exact & approx) >= 9  # allow one miss from approximation

    def test_calculate_similarity_bounds(self):
        a = np.array([1.0, 0.0])
        assert calculate_similarity(a, a) == pytest.approx(1.0)
        assert calculate_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
        assert calculate_similarity(a, np.zeros(2)) == 0.0


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


class TestAPI:
    def test_health_is_always_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_ready_reports_product_count(self, client):
        body = client.get("/ready").json()
        assert body["status"] == "ready"
        assert body["n_products"] == 61

    def test_find_similar_products_happy_path(self, client):
        response = client.get(
            "/find_similar_products", params={"product_id": "id0000", "num_similar": 3}
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 3

    def test_unknown_product_returns_404(self, client):
        response = client.get(
            "/find_similar_products", params={"product_id": "nope", "num_similar": 3}
        )
        assert response.status_code == 404

    @pytest.mark.parametrize("num_similar", [0, -1, 101])
    def test_invalid_num_similar_returns_422(self, client, num_similar):
        response = client.get(
            "/find_similar_products",
            params={"product_id": "id0000", "num_similar": num_similar},
        )
        assert response.status_code == 422

    def test_missing_product_id_returns_422(self, client):
        assert client.get("/find_similar_products").status_code == 422

    def test_ids_endpoint_returns_plain_list(self, client):
        body = client.get(
            "/find_similar_product_ids", params={"product_id": "id0000", "num_similar": 4}
        ).json()
        assert isinstance(body["product_ids"], list)
        assert all(isinstance(pid, str) for pid in body["product_ids"])

    def test_hnsw_backend_is_selectable(self, client):
        response = client.get(
            "/find_similar_products",
            params={"product_id": "id0000", "num_similar": 3, "backend": "hnsw"},
        )
        assert response.status_code == 200
        assert response.json()["backend"] == "hnsw"

    def test_unknown_backend_rejected(self, client):
        response = client.get(
            "/find_similar_products",
            params={"product_id": "id0000", "backend": "faiss"},
        )
        assert response.status_code == 422


# --------------------------------------------------------------------------
# Integration - real dataset, skipped when absent
# --------------------------------------------------------------------------

requires_dataset = pytest.mark.skipif(
    not DATASET_PATH.exists(), reason="dataset not present; unzip data/archive.zip"
)


@requires_dataset
def test_real_dataset_loads_with_expected_shape():
    from app.data import load_and_normalise

    df = load_and_normalise(DATASET_PATH)
    assert len(df) == 30_000
    assert df["product_id"].is_unique
    # Documented coverage from profiling - a regression here means the
    # parsers silently stopped working.
    assert df["child_category"].notna().mean() > 0.80
    assert df["weight_grams"].notna().mean() < 0.25


@requires_dataset
def test_real_dataset_end_to_end():
    index = SimilarityIndex.from_dataset(build_ann=False)
    pid = index.df.iloc[0]["product_id"]
    results = index.search(pid, 5)
    assert len(results) == 5
    assert all(0.0 <= r.score <= 1.0 for r in results)
