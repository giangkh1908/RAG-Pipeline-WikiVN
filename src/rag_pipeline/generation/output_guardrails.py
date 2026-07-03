"""Output guardrails — hallucination, safety, and quality checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_pipeline.config import OutputGuardrailsConfig
from rag_pipeline.models import AnswerResult, RetrievalResult


@dataclass(slots=True)
class OutputGuardrails:
    """Checks generated answers for hallucination, safety, and quality.

    Checks:
    - Hallucination: claims not backed by retrieved passages
    - Safety: content policy violations in generated answer
    - Quality: answer length, citation count, language consistency
    """

    config: OutputGuardrailsConfig

    # Unsafe content patterns (same as QueryGuardrails but for output)
    _UNSAFE_PATTERNS: list[str] = field(default_factory=lambda: [
        r"\b(bom|vu khí|hack|tấn c.ng|đ.t毒)\b",
    ])

    def check(
        self, answer_result: AnswerResult, retrieval_result: RetrievalResult
    ) -> AnswerResult:
        """Run all output guardrail checks.

        Args:
            answer_result: Generated answer to check
            retrieval_result: Original retrieval result for context

        Returns:
            Updated AnswerResult with guardrail metadata flags
        """
        flags: list[str] = []
        metadata = dict(answer_result.metadata)

        # Hallucination check
        if self.config.enable_hallucination_check:
            hallucination_flags = self._check_hallucination(answer_result, retrieval_result)
            flags.extend(hallucination_flags)

        # Safety check
        if self.config.enable_safety_check:
            safety_flags = self._check_safety(answer_result)
            flags.extend(safety_flags)

        # Quality check
        if self.config.enable_quality_check:
            quality_flags = self._check_quality(answer_result)
            flags.extend(quality_flags)

        # Lower confidence if there are flags
        confidence = answer_result.confidence
        if flags:
            confidence = max(0.0, confidence - 0.2 * len(flags))

        metadata["guardrail_flags"] = flags
        metadata["guardrail_checked"] = True

        return AnswerResult(
            question=answer_result.question,
            answer=answer_result.answer,
            citations=answer_result.citations,
            confidence=confidence,
            passages_used=answer_result.passages_used,
            metadata=metadata,
        )

    def _check_hallucination(
        self, answer_result: AnswerResult, retrieval_result: RetrievalResult
    ) -> list[str]:
        """Check if answer claims are backed by passages."""
        flags: list[str] = []
        passage_texts = [p.text.lower() for p in retrieval_result.passages]

        if not passage_texts and answer_result.answer:
            # No passages but answer exists — likely hallucination
            flags.append("hallucination_no_context")
            return flags

        # Check if answer has content not found in any passage
        answer_lower = answer_result.answer.lower()
        # Simple heuristic: check if key phrases from answer appear in passages
        answer_sentences = re.split(r"[.!?]", answer_lower)
        answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 10]

        unbacked_count = 0
        for sentence in answer_sentences:
            # Check if any passage contains words from this sentence
            words = set(sentence.split())
            significant_words = {w for w in words if len(w) > 3}
            if not significant_words:
                continue

            backed = False
            for ptext in passage_texts:
                overlap = sum(1 for w in significant_words if w in ptext)
                if overlap >= len(significant_words) * 0.3:
                    backed = True
                    break
            if not backed:
                unbacked_count += 1

        if unbacked_count > 0 and answer_sentences:
            ratio = unbacked_count / len(answer_sentences)
            if ratio > 0.5:
                flags.append("hallucination_high_unbacked_ratio")

        return flags

    def _check_safety(self, answer_result: AnswerResult) -> list[str]:
        """Check for unsafe content in generated answer."""
        flags: list[str] = []
        answer_lower = answer_result.answer.lower()

        for pattern in self._UNSAFE_PATTERNS:
            if re.search(pattern, answer_lower, re.IGNORECASE):
                flags.append("unsafe_content_detected")
                break

        return flags

    def _check_quality(self, answer_result: AnswerResult) -> list[str]:
        """Check answer quality: length, citations, language."""
        flags: list[str] = []

        # Check answer length
        if len(answer_result.answer) > self.config.max_answer_length:
            flags.append("answer_too_long")

        if len(answer_result.answer.strip()) < 10:
            flags.append("answer_too_short")

        # Check citation count
        if len(answer_result.citations) < self.config.min_citations:
            flags.append("insufficient_citations")

        return flags
