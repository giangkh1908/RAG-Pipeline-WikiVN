"""End-to-end RAG pipeline with streaming support."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation.answer_generator import LLMAnswerGenerator
from rag_pipeline.generation.context_builder import (
    CitationContextBuilder,
    NoRelevantContextError,
)
from rag_pipeline.generation.memory import ConversationMemory, est_tokens
from rag_pipeline.generation.models import (
    AnswerResult,
    BuiltContext,
    GenerationEvent,
)
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage.base import Storage
from rag_pipeline.storage.conversation import ChatTurn, ConversationStore


class RAGPipeline:
    """Full RAG pipeline: retrieval → context → answer generation."""

    _NO_CONTEXT_MESSAGE = "Không đủ thông tin để trả lời câu hỏi này."

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: CitationContextBuilder,
        answer_generator: LLMAnswerGenerator,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.memory = memory

    @classmethod
    def from_config(
        cls,
        config: RAGConfig,
        storage: Storage,
        vector_store: Any,
        dense_embedder: Any,
        sparse_embedder: Any,
        conversation_store: ConversationStore | None = None,
        compactor: Any = None,
    ) -> "RAGPipeline":
        """Build a RAGPipeline from configuration and dependencies."""
        cache = QueryCache(storage)
        llm_processor = LLMQueryProcessor(config.retrieval.llm_query, cache=cache)
        filter_builder = FilterBuilder()
        retriever = HybridRetriever(
            config=config.retrieval,
            storage=storage,
            vector_store=vector_store,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        retrieval_pipeline = RetrievalPipeline(llm_processor, filter_builder, retriever)
        context_builder = CitationContextBuilder(config.context_builder)
        answer_generator = LLMAnswerGenerator(config.generation)
        memory: ConversationMemory | None = None
        if config.memory.enabled and conversation_store is not None:
            memory = ConversationMemory(
                config.memory, conversation_store, compactor=compactor
            )
        return cls(retrieval_pipeline, context_builder, answer_generator, memory=memory)

    def answer(self, query: str, session_id: str | None = None) -> AnswerResult:
        """Generate a complete answer synchronously."""
        for event in self.answer_stream(query, session_id=session_id):
            if event.type == "done":
                return event.data
            if event.type == "error":
                return AnswerResult(
                    query=query,
                    answer=event.message or self._NO_CONTEXT_MESSAGE,
                    context="",
                    sources=[],
                )
        return AnswerResult(
            query=query,
            answer=self._NO_CONTEXT_MESSAGE,
            context="",
            sources=[],
        )

    def answer_stream(
        self, query: str, session_id: str | None = None
    ) -> Iterator[GenerationEvent]:
        """Stream RAG progress events and answer tokens.

        Events are yielded as ``GenerationEvent`` objects. Consumers can
        serialize them to JSON for frontend streaming.

        When ``session_id`` is provided and memory is enabled, the pipeline
        persists every turn to ``ConversationStore`` and folds the prior
        history into the LLM message list.
        """
        memory_used = False
        turn_no: int | None = None
        store: ConversationStore | None = self.memory.store if self.memory else None

        if store is not None and session_id is not None:
            store.upsert_session(session_id)
            turn_no = store.next_turn_no(session_id)
            store.insert_turn(
                ChatTurn(session_id=session_id, turn_no=turn_no, question=query)
            )

        # Load recent conversation context BEFORE rewrite so the LLM
        # rewriter can resolve references like "5 cái trên", "chỗ đó".
        conversation_context: str | None = None
        if store is not None and session_id is not None:
            recent = store.load_completed_turns(session_id)
            if recent:
                conversation_context = self._build_rewrite_context(recent[-3:])

        yield GenerationEvent(
            type="progress",
            step="rewrite",
            message="Đang viết lại câu hỏi...",
        )
        processed = self.retrieval_pipeline.preprocess(query, conversation_context)

        yield GenerationEvent(
            type="progress",
            step="retrieval",
            message="Đang tìm kiếm thông tin...",
        )
        results = self.retrieval_pipeline.search_processed(processed)

        if not results:
            if store is not None and session_id is not None and turn_no is not None:
                store.update_turn_answer(
                    session_id, turn_no, self._NO_CONTEXT_MESSAGE, processed.intent, 0
                )
            yield GenerationEvent(
                type="error",
                message=self._NO_CONTEXT_MESSAGE,
            )
            return

        yield GenerationEvent(
            type="progress",
            step="context",
            message=f"Tìm thấy {len(results)} đoạn văn bản",
        )
        try:
            built = self.context_builder.build(results, query=query)
        except NoRelevantContextError:
            if store is not None and session_id is not None and turn_no is not None:
                store.update_turn_answer(
                    session_id, turn_no, self._NO_CONTEXT_MESSAGE, processed.intent, 0
                )
            yield GenerationEvent(
                type="error",
                message=self._NO_CONTEXT_MESSAGE,
            )
            return

        yield GenerationEvent(
            type="progress",
            step="generation",
            message="Đang tạo câu trả lời...",
        )

        # Build LLM messages. With memory: history + summary + question.
        # Without memory: legacy (query, context) path.
        if self.memory is not None and session_id is not None:
            history = self.memory.build_history(
                session_id=session_id,
                current_question=query,
                system_guideline=self.answer_generator._SYSTEM_PROMPT,
                rag_context=built.context,
            )
            messages = history.messages
            memory_used = history.used
            token_iter = self.answer_generator.generate_stream_messages(messages)
        else:
            token_iter = self.answer_generator.generate_stream(query, built.context)

        answer_parts: list[str] = []
        for token in token_iter:
            answer_parts.append(token)
            yield GenerationEvent(type="token", data=token)

        answer = "".join(answer_parts)
        answer = self._strip_question_echo(query, answer)
        answer_tokens_hint = est_tokens(answer, self.answer_generator.config.max_tokens // 4 or 3)
        sources = self._extract_sources(answer, built)
        result = AnswerResult(
            query=query,
            answer=answer,
            context=built.context,
            sources=sources,
            intent=processed.intent,
            session_id=session_id,
            turn_no=turn_no,
            memory_used=memory_used,
        )

        if store is not None and session_id is not None and turn_no is not None:
            try:
                store.update_turn_answer(
                    session_id, turn_no, answer, processed.intent, answer_tokens_hint
                )
                store.add_to_token_total(session_id, answer_tokens_hint)
            except Exception:
                pass

        yield GenerationEvent(type="done", data=result)

    @staticmethod
    def _build_rewrite_context(turns: list[ChatTurn]) -> str:
        """Format recent turns as compact text for the query rewriter.

        Each answer is truncated to ~300 chars so the context stays small
        and the rewrite LLM call remains cheap.
        """
        parts: list[str] = []
        for t in turns:
            parts.append(f"User: {t.question}")
            answer = t.answer or ""
            if len(answer) > 300:
                answer = answer[:300] + "..."
            parts.append(f"Assistant: {answer}")
        return "\n".join(parts)

    @staticmethod
    def _strip_question_echo(query: str, answer: str) -> str:
        """Remove the question text if the LLM echoed it at the start of its answer.

        When chat memory is enabled, the LLM sees the conversation history
        and sometimes begins its response by repeating the user's latest
        question before answering it. This strips that echo so the user
        doesn't see a duplicate of their own question inside the answer
        bubble.
        """
        if not query or not answer:
            return answer

        # Normalise query: lowercase, collapse whitespace, strip trailing punctuation.
        q_norm = " ".join(query.strip().lower().split())
        q_norm = re.sub(r"[?!.,;:]+$", "", q_norm).strip()
        if not q_norm:
            return answer

        # Quick check: does the normalised answer start with the normalised query?
        a_stripped = answer.lstrip()
        a_norm = " ".join(a_stripped.lower().split())
        if not a_norm.startswith(q_norm):
            return answer

        # Walk through the original answer to find where the echo ends,
        # tolerating whitespace differences between the normalised query
        # and the original text.
        lower = a_stripped.lower()
        qi = 0
        cut = 0
        for i in range(len(lower)):
            if qi >= len(q_norm):
                cut = i
                break
            ch = lower[i]
            qch = q_norm[qi]

            if ch.isspace() and qch.isspace():
                qi += 1
                continue
            if ch.isspace():
                continue  # extra whitespace in answer — skip
            if qch.isspace():
                # consume remaining whitespace in query
                while qi < len(q_norm) and q_norm[qi].isspace():
                    qi += 1
                if qi >= len(q_norm):
                    cut = i
                    break
                qch = q_norm[qi]
            if ch == qch:
                qi += 1
            else:
                return answer  # mismatch — not an echo
        else:
            cut = len(lower)

        if qi < len(q_norm):
            return answer

        remainder = a_stripped[cut:]
        # Only strip if there's a clear separator (newline or punctuation)
        # between the echo and the real answer. If the text flows directly
        # into a letter (e.g. "Hà Nội là thủ đô"), it's not an echo — it's
        # just the topic name opening the sentence.
        after_echo = remainder.lstrip()
        if not after_echo:
            return answer
        first_char = after_echo[0]
        if first_char.isalpha() or first_char.isdigit():
            # Flows into text — not an echo, just a shared topic word.
            return answer

        # Strip the separator (punctuation, newlines, whitespace).
        remainder = re.sub(r"^[.!?,;:\n\r\t ]+", "", remainder).lstrip()

        if not remainder or len(remainder.strip()) < 5:
            return answer
        return remainder

    @staticmethod
    def _extract_sources(answer: str, built: BuiltContext) -> list[dict[str, Any]]:
        """Extract citations used in the answer and map them to source metadata."""
        used_citations = sorted(set(re.findall(r"\[(\d+)\]", answer)))
        sources: list[dict[str, Any]] = []
        for num in used_citations:
            citation = f"[{num}]"
            if citation not in built.citations:
                continue
            result = built.citations[citation]
            sources.append(
                {
                    "citation": citation,
                    "title": result.metadata.get("title", ""),
                    "content": result.content,
                    "chunk_id": str(result.chunk_id),
                }
            )
        return sources

    def close(self) -> None:
        self.retrieval_pipeline.close()
        self.answer_generator.close()
