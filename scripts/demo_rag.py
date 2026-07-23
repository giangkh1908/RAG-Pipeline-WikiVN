"""Interactive RAG demo: ask questions and stream answers.

Usage:
    python scripts/demo_rag.py

The script builds a full RAG pipeline from configuration, asks a sample
question, and prints streaming events including rewrite, retrieval, context
building, token generation, and final sources.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation import RAGPipeline
from rag_pipeline.indexing import (
    DenseEmbedder,
    QdrantVectorStore,
    SparseEmbedder,
)
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage import SQLiteStorage

load_dotenv()


def build_pipeline(config: RAGConfig) -> RAGPipeline:
    """Build RAG pipeline from configuration."""
    storage = SQLiteStorage(config.retrieval.storage.db_path)
    vector_store = QdrantVectorStore(config.retrieval.qdrant)
    dense_embedder = DenseEmbedder(config.retrieval.dense)
    sparse_embedder = SparseEmbedder(config.retrieval.sparse)

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

    from rag_pipeline.generation import CitationContextBuilder, LLMAnswerGenerator

    context_builder = CitationContextBuilder(config.context_builder)
    answer_generator = LLMAnswerGenerator(config.generation)

    return RAGPipeline(retrieval_pipeline, context_builder, answer_generator)


def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable.")
        raise SystemExit(1)

    config = RAGConfig()
    pipeline = build_pipeline(config)

    sample_questions = [
        "Ha Long Bay nằm ở đâu?",
        "Du lịch Hội An nên đi mùa nào?",
        "Có món ăn đặc sản nào ở Đà Nẵng?",
    ]

    print("Sample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print("(Press Enter to use question 1, or type your own)")

    choice = input("Question: ").strip()
    if not choice:
        query = sample_questions[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(sample_questions):
        query = sample_questions[int(choice) - 1]
    else:
        query = choice

    print(f"\n> {query}\n")
    answer_text = ""

    try:
        for event in pipeline.answer_stream(query):
            if event.type == "progress":
                print(f"[{event.step}] {event.message}")
            elif event.type == "token":
                print(event.data, end="", flush=True)
                answer_text += event.data
                time.sleep(0.01)
            elif event.type == "done":
                print("\n\n--- Sources ---")
                for source in event.data.sources:
                    print(f"{source['citation']} {source['title']}")
            elif event.type == "error":
                print(f"\nError: {event.message}")
                break
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
