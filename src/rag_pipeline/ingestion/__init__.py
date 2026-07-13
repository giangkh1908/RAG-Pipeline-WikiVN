"""Ingestion pipeline: Raw JSON → Source → Document → Chunk → Storage."""

from rag_pipeline.ingestion.loader import VietnamTourismLoader
from rag_pipeline.ingestion.pipeline import IngestionPipeline

__all__ = ["VietnamTourismLoader", "IngestionPipeline"]
