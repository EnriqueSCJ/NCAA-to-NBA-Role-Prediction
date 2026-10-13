"""Search comparable players in Milvus."""

from __future__ import annotations

import random
from typing import Any

from pymilvus import MilvusClient

import src.config as config
from src.vectorstore.schema import get_search_params


def get_milvus_uri() -> str:
    """Build the Milvus URI from project configuration."""
    host = config.MILVUS_HOST
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}:{config.MILVUS_PORT}"


def connect_to_milvus() -> MilvusClient:
    """Connect to Milvus using project configuration."""
    return MilvusClient(uri=get_milvus_uri())


def deterministic_query_vector(seed: int = 101) -> list[float]:
    """Generate a deterministic query vector for checkpoint searches."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(config.NUMERIC_VECTOR_DIM)]


def search_comparable_players(
    query_vector: list[float],
    top_k: int = 5,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Search comparable players by numeric profile vector."""
    client = connect_to_milvus()
    target_collection = collection_name or config.MILVUS_COLLECTION_NAME

    if not client.has_collection(target_collection):
        raise ValueError(f"Milvus collection does not exist: {target_collection}")

    client.load_collection(target_collection)

    raw_results = client.search(
        collection_name=target_collection,
        data=[query_vector],
        anns_field="numeric_vector",
        search_params=get_search_params(),
        limit=top_k,
        output_fields=[
            "player_id",
            "player_name",
            "position",
            "target_role",
            "draft_year",
            "college",
        ],
    )

    comparable_players: list[dict[str, Any]] = []
    for hits in raw_results:
        for hit in hits:
            entity = hit.get("entity", {})
            comparable_players.append(
                {
                    "player_id": hit.get("id") or entity.get("player_id"),
                    "player_name": entity.get("player_name"),
                    "position": entity.get("position"),
                    "target_role": entity.get("target_role"),
                    "draft_year": entity.get("draft_year"),
                    "college": entity.get("college"),
                    "score": float(hit.get("distance", hit.get("score", 0.0))),
                }
            )

    return comparable_players


def print_results(results: list[dict[str, Any]]) -> None:
    """Print comparable player results in a compact report format."""
    print(f"Top {len(results)} comparable players:")
    for player in results:
        print(
            "  - "
            f"{player['player_name']} ({player['position']}) | "
            f"{player['college']} {player['draft_year']} | "
            f"Role: {player['target_role']} | "
            f"Score: {player['score']:.4f}"
        )


def main() -> None:
    """Run a search checkpoint against the sample collection."""
    query_vector = deterministic_query_vector()
    results = search_comparable_players(query_vector=query_vector, top_k=5)
    print_results(results)
    print("\nMilvus search checkpoint completed.")


if __name__ == "__main__":
    main()
