"""High-level retrieval pipeline: preprocess + hybrid search."""

from __future__ import annotations

from rag_pipeline.retrieval.filter_builder import FilterBuilder
from rag_pipeline.retrieval.hybrid_retriever import HybridRetriever
from rag_pipeline.retrieval.llm_query_processor import LLMQueryProcessor, ProcessedQuery
from rag_pipeline.retrieval.models import RetrievalResult


class RetrievalPipeline:
    """End-to-end retrieval pipeline.

    Orchestrates query preprocessing (rewrite + intent), filter building, and
    hybrid dense + sparse retrieval with RRF fusion.
    """

    def __init__(
        self,
        llm_processor: LLMQueryProcessor,
        filter_builder: FilterBuilder,
        retriever: HybridRetriever,
    ) -> None:
        self.llm_processor = llm_processor
        self.filter_builder = filter_builder
        self.retriever = retriever

    def preprocess(
        self, query: str, conversation_context: str | None = None
    ) -> ProcessedQuery:
        """Preprocess a raw query.

        When ``conversation_context`` is provided, the LLM rewriter uses
        it to resolve references like "5 cái trên", "chỗ đó", etc.
        """
        return self.llm_processor.process(query, conversation_context)

    def search_processed(
        self,
        processed: ProcessedQuery,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search using an already preprocessed query."""
        filters = self.filter_builder.build(processed)
        search_query = processed.rewritten_query or processed.normalized_query
        return self.retriever.retrieve(search_query, top_k=top_k, filters=filters)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Search for relevant chunks given a raw user query."""
        processed = self.preprocess(query)
        return self.search_processed(processed, top_k=top_k)

    def close(self) -> None:
        self.llm_processor.close()
        self.retriever.dense_embedder.close()
