#!/usr/bin/env python
"""Small CLI to inspect and manage Milvus collections.

Run from Docker Compose for the cleanest networking:

    docker compose run --rm milvus-viewer python milvus_viewer.py --list
    docker compose run --rm milvus-viewer python milvus_viewer.py --create-demo
    docker compose run --rm milvus-viewer

When running directly from the host, install pymilvus first and make sure
Milvus is reachable on localhost:19530.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

import src.config as config


def get_milvus_uri() -> str:
    """Build a Milvus URI that works both inside and outside Docker Compose."""
    explicit_uri = os.getenv("MILVUS_URI")
    if explicit_uri:
        return explicit_uri

    host = os.getenv("MILVUS_HOST", config.MILVUS_HOST)
    port = os.getenv("MILVUS_PORT", str(config.MILVUS_PORT))

    if host.startswith(("http://", "https://")):
        return host

    return f"http://{host}:{port}"


CLIENT = MilvusClient(uri=get_milvus_uri())
DEMO_COLLECTION_NAME = "demo_players"
DEMO_VECTOR_FIELD = "embeddings"


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def list_collections() -> list[str]:
    """List all Milvus collections."""
    print_header("COLECCIONES EN MILVUS")
    collections = CLIENT.list_collections()

    if not collections:
        print("No hay colecciones. La base vectorial esta vacia.")
        return collections

    print(f"Total de colecciones: {len(collections)}\n")
    for collection in collections:
        print(f"  - {collection}")

    return collections


def collection_stats(collection_name: str) -> None:
    """Print schema and row count for one collection."""
    print_header(f"ESTADISTICAS: {collection_name}")

    try:
        description = CLIENT.describe_collection(collection_name)
        stats = CLIENT.get_collection_stats(collection_name)
    except Exception as exc:
        print(f"Error leyendo la coleccion: {exc}")
        return

    row_count = stats.get("row_count", "desconocido")
    print(f"Documentos: {row_count}")

    fields: list[dict[str, Any]] = description.get("fields", [])
    if fields:
        print("\nCampos:")
        for field in fields:
            print(f"  - {field}")


def get_row_count(collection_name: str) -> int:
    """Return the row count reported by Milvus, falling back to zero."""
    try:
        stats = CLIENT.get_collection_stats(collection_name)
    except Exception:
        return 0

    value = stats.get("row_count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def create_demo_schema() -> CollectionSchema:
    """Build the schema used by the demo collection."""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="player_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="height", dtype=DataType.FLOAT),
        FieldSchema(name=DEMO_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    return CollectionSchema(fields=fields, description="Demo de jugadores")


def demo_rows() -> list[dict[str, Any]]:
    """Return deterministic demo rows."""
    return [
        {
            "id": 1,
            "player_name": "Stephen Curry",
            "position": "G",
            "height": 75.0,
            DEMO_VECTOR_FIELD: [0.1] * 384,
        },
        {
            "id": 2,
            "player_name": "LeBron James",
            "position": "F",
            "height": 81.0,
            DEMO_VECTOR_FIELD: [0.2] * 384,
        },
        {
            "id": 3,
            "player_name": "Kevin Durant",
            "position": "F",
            "height": 82.0,
            DEMO_VECTOR_FIELD: [0.3] * 384,
        },
    ]


def insert_demo_rows(collection_name: str) -> None:
    """Insert demo rows into an empty collection."""
    demo_data = demo_rows()
    CLIENT.insert(collection_name=collection_name, data=demo_data)
    CLIENT.flush(collection_name)
    print(f"{len(demo_data)} registros insertados.")


def ensure_demo_index(collection_name: str) -> None:
    """Create the vector index required to load/search the demo collection."""
    index_params = CLIENT.prepare_index_params()
    index_params.add_index(
        field_name=DEMO_VECTOR_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 8, "efConstruction": 64},
    )

    try:
        CLIENT.create_index(collection_name=collection_name, index_params=index_params)
        print(f"Indice creado en '{DEMO_VECTOR_FIELD}'.")
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "exist" in message:
            print(f"Indice de '{DEMO_VECTOR_FIELD}' ya existe.")
        else:
            raise


def create_demo_collection() -> None:
    """Create, populate, or repair a demo collection for viewer testing."""
    print_header("CREANDO O REPARANDO COLECCION DE DEMO")

    collection_name = DEMO_COLLECTION_NAME
    if CLIENT.has_collection(collection_name):
        print(f"La coleccion '{collection_name}' ya existe.")
    else:
        CLIENT.create_collection(
            collection_name=collection_name,
            schema=create_demo_schema(),
        )
        print(f"Coleccion '{collection_name}' creada.")

    row_count = get_row_count(collection_name)
    if row_count == 0:
        insert_demo_rows(collection_name)
    else:
        print(f"La coleccion ya contiene {row_count} registros.")

    ensure_demo_index(collection_name)

    CLIENT.load_collection(collection_name)
    print("Coleccion cargada en memoria.")


def query_collection(collection_name: str, limit: int = 10) -> None:
    """Print the first rows from one collection."""
    print_header(f"DATOS: {collection_name} (primeros {limit})")

    try:
        CLIENT.load_collection(collection_name)
        results = CLIENT.query(
            collection_name=collection_name,
            filter="",
            output_fields=["*"],
            limit=limit,
        )
    except Exception as exc:
        print(f"Error al consultar: {exc}")
        return

    if not results:
        print("Sin datos en esta coleccion.")
        return

    print(f"{len(results)} registros encontrados:\n")
    for index, document in enumerate(results, 1):
        print(f"{index}. {compact_document(document)}")


def compact_value(value: Any) -> Any:
    """Compact long vectors so collection previews stay readable."""
    if isinstance(value, list) and len(value) > 12:
        numeric = all(isinstance(item, (int, float)) for item in value[:12])
        if numeric:
            preview = ", ".join(f"{item:.4g}" for item in value[:5])
            return f"[{preview}, ...] len={len(value)}"
    return value


def compact_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a printable document with large vector fields summarized."""
    return {key: compact_value(value) for key, value in document.items()}


def health_check() -> bool:
    """Check whether Milvus is reachable."""
    print_header("ESTADO DE MILVUS")
    print(f"URI: {get_milvus_uri()}")

    try:
        version = CLIENT.get_server_version()
    except Exception as exc:
        print(f"Milvus no responde: {exc}")
        return False

    print("Milvus esta corriendo.")
    print(f"Version: {version}")
    return True


def show_menu() -> None:
    """Interactive menu for manual inspection."""
    while True:
        print("\n" + "=" * 60)
        print("  MILVUS VIEWER - GESTOR DE COLECCIONES")
        print("=" * 60)
        print("1. Ver estado de Milvus")
        print("2. Listar colecciones")
        print("3. Ver estadisticas de coleccion")
        print("4. Consultar datos")
        print("5. Crear coleccion de demostracion")
        print("6. Salir")

        choice = input("Selecciona una opcion (1-6): ").strip()

        if choice == "1":
            health_check()
        elif choice == "2":
            list_collections()
        elif choice == "3":
            collection_name = input("Nombre de la coleccion: ").strip()
            collection_stats(collection_name)
        elif choice == "4":
            collection_name = input("Nombre de la coleccion: ").strip()
            limit = int(input("Registros a mostrar (default 10): ") or "10")
            query_collection(collection_name, limit)
        elif choice == "5":
            create_demo_collection()
        elif choice == "6":
            print("\nHasta luego.")
            break
        else:
            print("Opcion no valida.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Milvus collections.")
    parser.add_argument("--health", action="store_true", help="Check Milvus status.")
    parser.add_argument("--list", action="store_true", help="List collections.")
    parser.add_argument("--create-demo", action="store_true", help="Create demo data.")
    parser.add_argument("--stats", metavar="COLLECTION", help="Show collection stats.")
    parser.add_argument("--query", metavar="COLLECTION", help="Show collection rows.")
    parser.add_argument("--limit", type=int, default=10, help="Rows to show in queries.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the interactive menu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ran_command = False
    if args.health:
        health_check()
        ran_command = True
    if args.list:
        list_collections()
        ran_command = True
    if args.create_demo:
        create_demo_collection()
        ran_command = True
    if args.stats:
        collection_stats(args.stats)
        ran_command = True
    if args.query:
        query_collection(args.query, args.limit)
        ran_command = True
    if args.interactive or not ran_command:
        health_check()
        list_collections()
        show_menu()


if __name__ == "__main__":
    main()
