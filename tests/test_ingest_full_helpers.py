"""Smoke tests for full Milvus ingestion helpers."""

from __future__ import annotations

import numpy as np

import src.config as config
from src.vectorstore.ingest import (
    embed_profile_texts,
    generate_profile_text,
    make_demo_master_table,
    normalize_vectors,
)


def main() -> None:
    demo = make_demo_master_table()
    profile = generate_profile_text(demo.iloc[0])

    required_terms = ["Stephen Curry", "Davidson", "NCAA profile"]
    missing_terms = [term for term in required_terms if term not in profile]
    if missing_terms:
        raise AssertionError(f"Missing profile text terms: {missing_terms}")

    vectors = normalize_vectors(demo)
    if len(vectors) != len(demo):
        raise AssertionError("Numeric vector count does not match row count")
    if len(vectors[0]) != config.NUMERIC_VECTOR_DIM:
        raise AssertionError("Unexpected numeric vector dimension")
    if np.isnan(np.asarray(vectors, dtype=float)).any():
        raise AssertionError("Numeric vectors contain NaNs")

    text_vectors = embed_profile_texts([profile])
    if len(text_vectors[0]) != config.EMBEDDING_DIM:
        raise AssertionError("Unexpected text vector dimension")

    print("Full Milvus ingestion helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
