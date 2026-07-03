"""Answer generator — calls LLM and parses structured answer."""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from rag_pipeline.config import GenerationConfig
from rag_pipeline.generation.prompt_builder import PromptBuilder
from rag_pipeline.indexing.llm_client import LLMClient, OpenRouterLLMClient
from rag_pipeline.models import AnswerResult, Citation, RetrievalResult


@dataclass(slots=True)
class AnswerGenerator:
    """Generates answers using LLM with structured output parsing.

    Flow:
    1. Build prompt via PromptBuilder
    2. Call LLM with chat_json
    3. Parse JSON response into AnswerResult with Citations
    4. Fallback: wrap raw text as answer if JSON parse fails
    """

    llm_client: LLMClient
    prompt_builder: PromptBuilder
    config: GenerationConfig

    def generate(self, retrieval_result: RetrievalResult) -> AnswerResult:
        """Generate an answer from retrieval result.

        Args:
            retrieval_result: Output from Phase 3 retrieval

        Returns:
            AnswerResult with answer, citations, and confidence
        """
        messages = self.prompt_builder.build(retrieval_result)

        try:
            response = self.llm_client.chat_json(
                messages,
                max_tokens=self.config.max_answer_tokens,
                temperature=self.config.temperature,
            )
            return self._parse_response(response, retrieval_result)
        except Exception:
            # Fallback: try plain text response
            raw_text = self.llm_client.chat(
                messages,
                max_tokens=self.config.max_answer_tokens,
                temperature=self.config.temperature,
            )
            if not raw_text:
                return AnswerResult(
                    question=retrieval_result.query.original_query,
                    answer="Không thể tạo câu trả lời (LLM không phản hồi).",
                    citations=[],
                    confidence=0.0,
                    passages_used=len(retrieval_result.passages),
                    metadata={"parse_mode": "llm_no_response"},
                )
            return AnswerResult(
                question=retrieval_result.query.original_query,
                answer=raw_text.strip(),
                citations=[],
                confidence=0.3,
                passages_used=len(retrieval_result.passages),
                metadata={"parse_mode": "fallback_text"},
            )

    def generate_stream(self, retrieval_result: RetrievalResult) -> tuple[Generator[str, None, None], Any]:
        """Get streaming generator for answer.

        Streaming uses plain text prompt (no JSON). Citations are built
        from retrieval passages directly.

        Returns:
            Tuple of (chunk_generator, build_result_func).
            - chunk_generator: yields text chunks
            - build_result_func: call with accumulated text to get AnswerResult
        """
        # Use streaming prompt (plain text, no JSON)
        messages = self.prompt_builder.build_streaming(retrieval_result)

        if not isinstance(self.llm_client, OpenRouterLLMClient):
            result = self.generate(retrieval_result)

            def _fake_stream():
                yield result.answer

            return _fake_stream(), lambda _: result

        chunk_gen = self.llm_client.stream(
            messages,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
        )

        def build_result(full_text: str) -> AnswerResult:
            """Build AnswerResult from stream text + retrieval passages."""
            # Build citations from passages used in generation
            citations = []
            for p in retrieval_result.passages[:5]:  # top 5 passages
                if p.title:
                    # Normalize score to [0, 1] (rerank_score can be > 1)
                    raw_score = p.rerank_score or p.rrf_score or p.dense_score or 0.5
                    score = min(max(raw_score, 0.0), 1.0)
                    citations.append(
                        Citation(
                            claim="",
                            chunk_id=p.chunk_id,
                            doc_id=p.doc_id,
                            title=p.title,
                            source_url=p.source_url or "",
                            confidence=score,
                        )
                    )

            return AnswerResult(
                question=retrieval_result.query.original_query,
                answer=full_text.strip(),
                citations=citations,
                confidence=0.7,
                passages_used=len(retrieval_result.passages),
                metadata={"parse_mode": "streaming"},
            )

        return chunk_gen, build_result

    def _parse_response(
        self, response: dict[str, Any], retrieval_result: RetrievalResult
    ) -> AnswerResult:
        """Parse LLM JSON response into AnswerResult."""
        answer_text = response.get("answer", "")
        raw_citations = response.get("citations", [])
        confidence = response.get("confidence", 0.5)

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))

        # Map source_index back to passages
        passages = retrieval_result.passages
        citations: list[Citation] = []
        for raw_cite in raw_citations:
            source_idx = raw_cite.get("source_index", 0)
            # source_index is 1-based (matches passage rank)
            passage = self._find_passage(passages, source_idx)
            if passage:
                citations.append(
                    Citation(
                        claim=raw_cite.get("claim", ""),
                        chunk_id=passage.chunk_id,
                        doc_id=passage.doc_id,
                        title=passage.title,
                        source_url=passage.source_url,
                        confidence=confidence,
                    )
                )

        return AnswerResult(
            question=retrieval_result.query.original_query,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            passages_used=len(passages),
            metadata={"parse_mode": "structured_json"},
        )

    @staticmethod
    def _find_passage(passages: list, rank: int):
        """Find passage by rank (1-based)."""
        for p in passages:
            if p.rank == rank:
                return p
        return None
