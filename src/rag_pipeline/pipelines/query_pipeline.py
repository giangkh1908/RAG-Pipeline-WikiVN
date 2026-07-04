"""Query processing pipeline — orchestrates normalize → rewrite → guardrails."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.config import QueryConfig
from rag_pipeline.models import ProcessedQuery
from rag_pipeline.query.guardrails import QueryGuardrails
from rag_pipeline.query.normalizer import QueryNormalizer
from rag_pipeline.query.rewriter import QueryRewriter


@dataclass(slots=True)
class QueryPipeline:
    """Orchestrates query processing: guardrails → normalize → rewrite.

    Flow:
    1. Guardrails check (prompt injection, unsafe content)
    2. Normalize (Vietnamese text normalization, intent classification)
    3. LLM rewrite (if enabled) — produces normalized, rewrite, bm25 variants
    4. Return ProcessedQuery ready for Phase 3 retrieval
    """

    config: QueryConfig
    normalizer: QueryNormalizer
    guardrails: QueryGuardrails
    rewriter: QueryRewriter | None = None

    def run(
        self,
        query: str,
        qid: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> ProcessedQuery:
        """Process a query through the full pipeline.

        Args:
            query: Current user question
            qid: Query ID
            history: Optional conversation history for context-aware rewriting
        """
        # Step 1: Guardrails
        risk_flags: list[str] = []
        if self.config.enable_guardrails:
            guardrail_result = self.guardrails.check(query)
            risk_flags = guardrail_result.risk_flags

        # Step 2: Normalize
        normalized = self.normalizer.normalize(query)

        # Step 3: LLM Rewrite (if enabled and rewriter available)
        if self.config.enable_rewrite and self.rewriter is not None:
            rewrite_result = self.rewriter.rewrite(query, history=history)
            return ProcessedQuery(
                qid=qid,
                original_query=query,
                normalized_query=rewrite_result.normalized_query,
                rewrite_query=rewrite_result.rewrite_query,
                bm25_query=rewrite_result.bm25_query,
                intent=rewrite_result.intent,
                filters=normalized.filters,
                risk_flags=risk_flags,
            )

        # Fallback: use normalization only
        return ProcessedQuery(
            qid=qid,
            original_query=query,
            normalized_query=normalized.normalized_text,
            rewrite_query=query,
            bm25_query=normalized.normalized_text,
            intent=normalized.intent,
            filters=normalized.filters,
            risk_flags=risk_flags,
        )
