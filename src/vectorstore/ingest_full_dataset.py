"""Entrypoint for full Milvus player ingestion."""

from __future__ import annotations

from src.vectorstore.ingest import main_full


if __name__ == "__main__":
    main_full()
