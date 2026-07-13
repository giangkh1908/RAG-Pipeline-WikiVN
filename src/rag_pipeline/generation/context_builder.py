"""Build prompt context from retrieval results."""

from __future__ import annotations

from rag_pipeline.config import ContextBuilderConfig
from rag_pipeline.generation.models import BuiltContext
from rag_pipeline.retrieval.models import RetrievalResult


class NoRelevantContextError(Exception):
    """Raised when no retrieval results are available to build context."""


class CitationContextBuilder:
    """Build a citation-indexed context string from ranked retrieval results."""

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self.config = config or ContextBuilderConfig()

    def build(self, results: list[RetrievalResult], query: str | None = None) -> BuiltContext:
        """Assemble context with citation markers.

        Parameters
        ----------
        results:
            Ranked retrieval results.
        query:
            Optional query string (unused by the builder but kept for interface
            symmetry and future query-aware truncation).

        Returns
        -------
        ``BuiltContext`` containing the formatted context and citation mapping.

        Raises
        ------
        NoRelevantContextError
            If ``results`` is empty.
        """
        if not results:
            raise NoRelevantContextError("No retrieval results to build context from")

        selected = results[: self.config.max_chunks]
        chunks: list[str] = []
        citations: dict[str, RetrievalResult] = {}

        for idx, result in enumerate(selected, start=1):
            citation = self.config.citation_format.format(id=idx)
            citations[citation] = result

            title = result.metadata.get("title", "")
            if self.config.include_title and title:
                header = f"{citation} Tiêu đề: {title}"
            else:
                header = citation

            chunks.append(f"{header}\n{result.content.strip()}")

        context = "\n\n".join(chunks)
        return BuiltContext(context=context, citations=citations)
