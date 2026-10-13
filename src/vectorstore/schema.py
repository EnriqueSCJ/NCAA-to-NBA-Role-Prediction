"""Milvus schema for NCAA to NBA player profiles."""

from __future__ import annotations

from pymilvus import CollectionSchema, DataType, FieldSchema

import src.config as config


def get_collection_schema() -> CollectionSchema:
    """Return the Milvus schema used for player comparable search."""
    fields = [
        FieldSchema(
            name="player_id",
            dtype=DataType.VARCHAR,
            max_length=256,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(name="player_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="season", dtype=DataType.INT32),
        FieldSchema(name="college", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="height_in", dtype=DataType.FLOAT),
        FieldSchema(name="weight_lb", dtype=DataType.FLOAT),
        FieldSchema(name="draft_year", dtype=DataType.INT32),
        FieldSchema(name="draft_pick", dtype=DataType.INT32),
        FieldSchema(name="target_role", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(
            name="numeric_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.NUMERIC_VECTOR_DIM,
        ),
        FieldSchema(
            name="text_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.EMBEDDING_DIM,
        ),
        FieldSchema(name="profile_text", dtype=DataType.VARCHAR, max_length=2000),
    ]

    return CollectionSchema(
        fields=fields,
        description="NCAA to NBA player profiles for comparable search",
        enable_dynamic_field=False,
    )


def get_index_params() -> dict[str, dict[str, object]]:
    """Return vector index params for numeric and text embeddings."""
    return {
        "numeric_vector": {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        },
        "text_vector": {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        },
    }


def get_search_params() -> dict[str, object]:
    """Return default search params compatible with the HNSW index."""
    return {"metric_type": "COSINE", "params": {"ef": 32}}
