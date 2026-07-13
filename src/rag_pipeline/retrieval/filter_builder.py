"""Build Qdrant payload filters from processed queries."""

from __future__ import annotations

from typing import Any

from rag_pipeline.retrieval.llm_query_processor import ProcessedQuery


class FilterBuilder:
    """Rule-based filter builder for Qdrant payload filters."""

    # Intents where reference sections are unlikely to be relevant.
    _EXCLUDE_REFERENCE_INTENTS = {
        "factual",
        "recommendation",
        "comparison",
        "list",
    }

    def build(self, processed: ProcessedQuery) -> dict[str, Any] | None:
        """Build a filter dict for QdrantVectorStore.

        Returns ``None`` if no filters should be applied.
        """
        filters: dict[str, Any] = {}

        # By default, exclude reference sections for most intents.
        if processed.intent in self._EXCLUDE_REFERENCE_INTENTS:
            filters["is_reference_section"] = False

        # Title filter: if the rewritten query contains a known topic keyword,
        # we could add it here. For now we keep filters minimal to avoid
        # over-filtering and rely on the hybrid retriever for ranking.

        return filters if filters else None
