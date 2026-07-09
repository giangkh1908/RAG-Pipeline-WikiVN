from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Auto-load .env từ thư mục gốc của project
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from rag_pipeline.config import (
    EmbeddingConfig,
    EvalConfig,
    GenerationConfig,
    IngestConfig,
    LLMConfig,
    OutputGuardrailsConfig,
    QdrantConfig,
    QueryConfig,
    RetrievalConfig,
)
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import DeterministicTestEmbedder, LocalEmbedder, OpenRouterEmbeddingClient
from rag_pipeline.indexing.llm_client import DeterministicTestLLM, OpenRouterLLMClient
from rag_pipeline.indexing.reranker import BGEReranker, CohereReranker, DeterministicTestReranker
from rag_pipeline.indexing.vector_store import InMemoryVectorStore, QdrantVectorStore
from rag_pipeline.ingest.dataset import (
    ChunkedJsonlReader,
    HuggingFaceDatasetReader,
    LocalCorpusCsvReader,
    LocalJsonlReader,
    LocalQueryCsvReader,
)
from rag_pipeline.ingest.normalize import UVWWikipediaDocumentNormalizer
from rag_pipeline.models import AnswerResult, ProcessedQuery, QueryRecord, RetrievalResult
from rag_pipeline.eval.report import EvalReport
from rag_pipeline.eval.runner import EvalRunner
from rag_pipeline.generation.answer_generator import AnswerGenerator
from rag_pipeline.generation.output_guardrails import OutputGuardrails
from rag_pipeline.generation.prompt_builder import PromptBuilder
from rag_pipeline.pipelines.answer_pipeline import AnswerPipeline
from rag_pipeline.pipelines.ingest_pipeline import IngestPipeline
from rag_pipeline.pipelines.query_pipeline import QueryPipeline
from rag_pipeline.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.query.guardrails import QueryGuardrails
from rag_pipeline.query.normalizer import QueryNormalizer
from rag_pipeline.query.rewriter import QueryRewriter
from rag_pipeline.transform.cleaner import WikipediaArticleCleaner
from rag_pipeline.transform.structure_chunker import StructuredChunker


def build_ingest_pipeline(config: IngestConfig, use_qdrant: bool = False, skip_qdrant_check: bool = False) -> IngestPipeline:
    """Build pipeline: InMemory + DeterministicEmbedder (dev) or Qdrant + OpenRouter/Local (prod)."""
    embedding_mode = config.embedding.embedding_mode

    if use_qdrant:
        # Set vector size based on embedding mode
        if embedding_mode == "local":
            config.qdrant.vector_size = 1024  # Qwen3-Embedding-0.6B
        else:
            config.qdrant.vector_size = 2048  # OpenRouter

        vector_store = QdrantVectorStore(config.qdrant)

        if embedding_mode == "local":
            # Local GPU embedding
            embedder = LocalEmbedder(
                model_name=config.embedding.local_model_name,
                batch_size=config.embedding.local_batch_size,
                device=config.embedding.local_device,
            )
        else:
            # API embedding (OpenRouter)
            if not os.getenv(config.embedding.api_key_env):
                raise RuntimeError(
                    f"Production mode requires {config.embedding.api_key_env} environment variable."
                )
            embedder = OpenRouterEmbeddingClient(
                config.embedding,
                parallel_workers=config.embedding.parallel_workers,
            )
    else:
        vector_store = InMemoryVectorStore()
        embedder = DeterministicTestEmbedder()

    bm25_index = BM25Index(
        index_path=config.bm25_index_path,
        tokenizer_name=config.bm25_tokenizer,
    )

    return IngestPipeline(
        normalizer=UVWWikipediaDocumentNormalizer(
            jurisdiction=config.jurisdiction,
            language=config.language,
        ),
        cleaner=WikipediaArticleCleaner(),
        chunker=StructuredChunker(config.chunking),
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        skip_qdrant_check=skip_qdrant_check,
    )


def ingest_from_huggingface(config: IngestConfig, use_qdrant: bool = False, skip_qdrant_check: bool = False) -> list[str]:
    pipeline = build_ingest_pipeline(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    reader = HuggingFaceDatasetReader(
        dataset_name=config.hf_dataset_name,
        split=config.hf_dataset_split,
        sample_percent=config.hf_sample_percent,
        min_quality_score=config.hf_min_quality_score,
    )
    results = pipeline.run(reader.read())
    return [result.document.doc_id for result in results if result.updated]


def ingest_from_local_jsonl(config: IngestConfig, use_qdrant: bool = False, skip_qdrant_check: bool = False) -> list[str]:
    pipeline = build_ingest_pipeline(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    reader = LocalJsonlReader(config.jsonl_path, sample_percent=config.jsonl_sample_percent)
    results = pipeline.run(reader.read())
    return [result.document.doc_id for result in results if result.updated]


def ingest_from_local_corpus(config: IngestConfig, use_qdrant: bool = False, skip_qdrant_check: bool = False) -> list[str]:
    pipeline = build_ingest_pipeline(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    reader = LocalCorpusCsvReader(config.corpus_path)
    results = pipeline.run(reader.read())
    return [result.document.doc_id for result in results if result.updated]


def ingest(config: IngestConfig, use_qdrant: bool = False, skip_qdrant_check: bool = False) -> list[str]:
    if config.source_type == "local_jsonl":
        return ingest_from_local_jsonl(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    if config.source_type == "huggingface":
        return ingest_from_huggingface(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    if config.source_type == "local_corpus":
        return ingest_from_local_corpus(config, use_qdrant=use_qdrant, skip_qdrant_check=skip_qdrant_check)
    raise ValueError(f"Unsupported source_type: {config.source_type}")


def load_eval_queries(config: IngestConfig, split: str = "val") -> list[QueryRecord]:
    query_paths = {
        "train": config.train_queries_path,
        "train_split": config.train_split_queries_path,
        "val": config.validation_queries_path,
        "public_test": config.public_test_queries_path,
    }
    reader = LocalQueryCsvReader(query_paths[split])
    return list(reader.read())


def build_query_pipeline(config: QueryConfig, use_llm: bool = False) -> QueryPipeline:
    """Build query processing pipeline.

    Args:
        config: Query configuration
        use_llm: If True, use OpenRouter LLM for rewrite. If False, use normalization only.

    Returns:
        QueryPipeline instance
    """
    normalizer = QueryNormalizer()
    guardrails = QueryGuardrails()

    rewriter = None
    if use_llm and config.enable_rewrite:
        llm_client = OpenRouterLLMClient(config.llm)
        rewriter = QueryRewriter(llm=llm_client)

    return QueryPipeline(
        config=config,
        normalizer=normalizer,
        guardrails=guardrails,
        rewriter=rewriter,
    )


def build_retrieval_pipeline(
    retrieval_config: RetrievalConfig | None = None,
    qdrant_config: QdrantConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    use_qdrant: bool = True,
    use_reranker: bool = False,
) -> RetrievalPipeline:
    """Build retrieval pipeline.

    Args:
        retrieval_config: Retrieval configuration
        qdrant_config: Qdrant configuration
        embedding_config: Embedding configuration
        use_qdrant: If True, use Qdrant. If False, use InMemory.
        use_reranker: If True, use BGE re-ranker. If False, use test reranker.

    Returns:
        RetrievalPipeline instance
    """
    if retrieval_config is None:
        retrieval_config = RetrievalConfig()
    if qdrant_config is None:
        qdrant_config = QdrantConfig()
    if embedding_config is None:
        embedding_config = EmbeddingConfig()

    # Vector store
    if use_qdrant:
        vector_store = QdrantVectorStore(qdrant_config)
        embedder = OpenRouterEmbeddingClient(embedding_config)
    else:
        vector_store = InMemoryVectorStore()
        embedder = DeterministicTestEmbedder()

    # BM25 index
    bm25_index = BM25Index(
        index_path=retrieval_config.bm25_index_path,
        tokenizer_name=retrieval_config.bm25_tokenizer,
    )
    bm25_index.load()  # Try to load existing index

    # Re-ranker
    reranker = None
    if retrieval_config.enable_rerank:
        if use_reranker:
            if retrieval_config.rerank_provider == "cohere":
                reranker = CohereReranker(
                    model_name=retrieval_config.rerank_model,
                    api_key_env=retrieval_config.api_key_env,
                )
            else:
                reranker = BGEReranker(model_name=retrieval_config.rerank_model)
        else:
            reranker = DeterministicTestReranker()

    return RetrievalPipeline(
        config=retrieval_config,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )


def search(question: str, use_qdrant: bool = True, use_reranker: bool = False) -> RetrievalResult:
    """Run full search pipeline: query processing → retrieval.

    Args:
        question: User question
        use_qdrant: If True, use Qdrant
        use_reranker: If True, use BGE re-ranker

    Returns:
        RetrievalResult with passages and context
    """
    # Step 1: Query processing
    query_config = QueryConfig()
    processed = process_query(question, config=query_config, use_llm=True)

    # Step 2: Retrieval
    retrieval_config = RetrievalConfig()
    pipeline = build_retrieval_pipeline(
        retrieval_config=retrieval_config,
        use_qdrant=use_qdrant,
        use_reranker=use_reranker,
    )
    return pipeline.run(processed)


def process_query(query: str, config: QueryConfig | None = None, use_llm: bool = False) -> ProcessedQuery:
    """Process a single query through the pipeline.

    Args:
        query: User question
        config: Query configuration (uses default if None)
        use_llm: If True, use LLM for rewrite

    Returns:
        ProcessedQuery ready for retrieval
    """
    if config is None:
        config = QueryConfig()

    pipeline = build_query_pipeline(config, use_llm=use_llm)
    return pipeline.run(query, qid="cli")


def build_generation_pipeline(
    gen_config: GenerationConfig | None = None,
    output_config: OutputGuardrailsConfig | None = None,
    llm_config: LLMConfig | None = None,
    retrieval_pipeline: RetrievalPipeline | None = None,
    query_pipeline: QueryPipeline | None = None,
    use_test_llm: bool = False,
) -> AnswerPipeline:
    """Build the full RAG answer pipeline.

    Args:
        gen_config: Generation configuration
        output_config: Output guardrails configuration
        llm_config: LLM configuration
        retrieval_pipeline: Pre-built retrieval pipeline (builds default if None)
        query_pipeline: Pre-built query pipeline (builds default if None)
        use_test_llm: If True, use DeterministicTestLLM instead of OpenRouter

    Returns:
        AnswerPipeline instance
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    if output_config is None:
        output_config = OutputGuardrailsConfig()
    if llm_config is None:
        llm_config = LLMConfig()

    # Build sub-pipelines if not provided
    if query_pipeline is None:
        query_pipeline = build_query_pipeline(QueryConfig(llm=llm_config), use_llm=True)
    if retrieval_pipeline is None:
        retrieval_pipeline = build_retrieval_pipeline()

    # LLM client for generation
    if use_test_llm:
        llm_client = DeterministicTestLLM(response_mode="generation")
    else:
        llm_client = OpenRouterLLMClient(llm_config)

    prompt_builder = PromptBuilder(gen_config, llm_client=llm_client)
    answer_generator = AnswerGenerator(
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        config=gen_config,
    )
    output_guardrails = OutputGuardrails(output_config)

    return AnswerPipeline(
        query_pipeline=query_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        answer_generator=answer_generator,
        output_guardrails=output_guardrails,
    )


def ask(
    question: str,
    use_qdrant: bool = True,
    use_reranker: bool = False,
    use_llm: bool = True,
) -> AnswerResult:
    """Run full RAG pipeline: question → answer with citations.

    Args:
        question: User question
        use_qdrant: If True, use Qdrant
        use_reranker: If True, use re-ranker
        use_llm: If True, use LLM for query rewrite

    Returns:
        AnswerResult with answer, citations, and confidence
    """
    query_config = QueryConfig()
    query_pipeline = build_query_pipeline(query_config, use_llm=use_llm)

    retrieval_config = RetrievalConfig()
    retrieval_pipeline = build_retrieval_pipeline(
        retrieval_config=retrieval_config,
        use_qdrant=use_qdrant,
        use_reranker=use_reranker,
    )

    gen_config = GenerationConfig()
    llm_config = LLMConfig()
    pipeline = build_generation_pipeline(
        gen_config=gen_config,
        llm_config=llm_config,
        retrieval_pipeline=retrieval_pipeline,
        query_pipeline=query_pipeline,
    )
    return pipeline.ask(question)


def build_ask_pipeline(
    use_qdrant: bool = True,
    use_reranker: bool = False,
    use_llm: bool = True,
) -> AnswerPipeline:
    """Build the full answer pipeline for streaming use.

    Args:
        use_qdrant: If True, use Qdrant
        use_reranker: If True, use re-ranker
        use_llm: If True, use LLM for query rewrite

    Returns:
        AnswerPipeline instance
    """
    query_config = QueryConfig()
    query_pipeline = build_query_pipeline(query_config, use_llm=use_llm)

    retrieval_config = RetrievalConfig()
    retrieval_pipeline = build_retrieval_pipeline(
        retrieval_config=retrieval_config,
        use_qdrant=use_qdrant,
        use_reranker=use_reranker,
    )

    gen_config = GenerationConfig()
    llm_config = LLMConfig()
    return build_generation_pipeline(
        gen_config=gen_config,
        llm_config=llm_config,
        retrieval_pipeline=retrieval_pipeline,
        query_pipeline=query_pipeline,
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys

    # Fix Windows encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Qdrant")
    ingest_parser.add_argument("--sample", type=float, default=100.0, help="Sample percent (default: 100 = full)")
    ingest_parser.add_argument("--qdrant", action="store_true", default=True, help="Use Qdrant (default: True)")
    ingest_parser.add_argument("--clear", action="store_true", default=False, help="Clear Qdrant collection before ingest")

    # Chunk command (Phase 1: offline chunking)
    chunk_parser = subparsers.add_parser("chunk", help="Phase 1: Chunk documents → JSONL (no API needed)")
    chunk_parser.add_argument("--input", type=str, default="documents/train.jsonl", help="Input JSONL path")
    chunk_parser.add_argument("--output", type=str, default="chunks/chunks.jsonl", help="Output chunks JSONL path")
    chunk_parser.add_argument("--sample", type=float, default=100.0, help="Sample percent (default: 100 = full)")
    chunk_parser.add_argument("--limit", type=int, default=None, help="Only process first N documents")

    # Embed command (Phase 2: embed + index)
    embed_parser = subparsers.add_parser("embed", help="Phase 2: Embed chunks → Qdrant + BM25 (needs API)")
    embed_parser.add_argument("--input", type=str, default="chunks/chunks.jsonl", help="Input chunks JSONL path")
    embed_parser.add_argument("--qdrant", action="store_true", default=True, help="Use Qdrant (default: True)")
    embed_parser.add_argument("--no-qdrant", action="store_true", default=False, help="Use InMemory instead of Qdrant")
    embed_parser.add_argument("--clear", action="store_true", default=False, help="Clear Qdrant collection before embed")

    # Query command
    query_parser = subparsers.add_parser("query", help="Process a query")
    query_parser.add_argument("--question", type=str, required=True, help="Question to process")
    query_parser.add_argument("--no-llm", action="store_true", default=False, help="Disable LLM for query rewrite")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--question", type=str, required=True, help="Question to search")
    search_parser.add_argument("--no-qdrant", action="store_true", default=False, help="Use InMemory instead of Qdrant")
    search_parser.add_argument("--rerank", action="store_true", default=False, help="Use BGE re-ranker")

    # Ask command (Phase 4: full RAG pipeline)
    ask_parser = subparsers.add_parser("ask", help="Ask a question (full RAG pipeline)")
    ask_parser.add_argument("--question", type=str, required=True, help="Question to ask")
    ask_parser.add_argument("--no-qdrant", action="store_true", default=False, help="Use InMemory instead of Qdrant")
    ask_parser.add_argument("--rerank", action="store_true", default=False, help="Use re-ranker")
    ask_parser.add_argument("--no-llm", action="store_true", default=False, help="Disable LLM query rewrite")
    ask_parser.add_argument("--text", action="store_true", default=False, help="Output answer text only (user-friendly)")
    ask_parser.add_argument("--stream", action="store_true", default=False, help="Stream response tokens in real-time")

    # Eval command (Phase 5: RAGAS evaluation)
    eval_parser = subparsers.add_parser("eval", help="Run RAGAS evaluation")
    eval_parser.add_argument("--dataset", type=str, default="documents/eval.csv", help="Eval dataset CSV path")
    eval_parser.add_argument("--output", type=str, default="eval_report.json", help="Output report path (JSON)")
    eval_parser.add_argument("--limit", type=int, default=50, help="Max samples to evaluate")
    eval_parser.add_argument("--no-qdrant", action="store_true", default=False, help="Use InMemory instead of Qdrant")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.clear and args.qdrant:
            from qdrant_client import QdrantClient
            from rag_pipeline.config import QdrantConfig
            cfg = QdrantConfig()
            client = QdrantClient(url=cfg.url)
            try:
                client.delete_collection(cfg.collection_name)
                print(f"🗑️  Deleted collection '{cfg.collection_name}'")
            except Exception:
                print(f"ℹ️  Collection '{cfg.collection_name}' not found — starting fresh")

        skip_qdrant_check = args.clear

        config = IngestConfig(
            source_type="local_jsonl",
            jsonl_path="documents/train.jsonl",
            jsonl_sample_percent=args.sample,
        )
        doc_ids = ingest(config, use_qdrant=args.qdrant, skip_qdrant_check=skip_qdrant_check)
        print(f"✅ Indexed {len(doc_ids)} documents")

    elif args.command == "chunk":
        config = IngestConfig(
            source_type="local_jsonl",
            jsonl_path=args.input,
            jsonl_sample_percent=args.sample,
        )
        pipeline = build_ingest_pipeline(config, use_qdrant=False)
        reader = LocalJsonlReader(config.jsonl_path, sample_percent=config.jsonl_sample_percent)

        # Apply limit if specified
        if args.limit:
            import itertools
            records = itertools.islice(reader.read(), args.limit)
        else:
            records = reader.read()

        total_chunks = pipeline.run_chunking(records, Path(args.output))
        print(f"✅ Chunked into {total_chunks:,} chunks")

    elif args.command == "embed":
        use_qdrant = args.qdrant and not args.no_qdrant

        if args.clear and use_qdrant:
            from qdrant_client import QdrantClient
            from rag_pipeline.config import QdrantConfig
            cfg = QdrantConfig()
            client = QdrantClient(url=cfg.url)
            try:
                client.delete_collection(cfg.collection_name)
                print(f"🗑️  Deleted collection '{cfg.collection_name}'")
            except Exception:
                print(f"ℹ️  Collection '{cfg.collection_name}' not found — starting fresh")

        config = IngestConfig()
        pipeline = build_ingest_pipeline(config, use_qdrant=use_qdrant, skip_qdrant_check=args.clear)
        results = pipeline.run_embedding(Path(args.input), use_qdrant=use_qdrant)
        indexed = sum(1 for r in results if r.updated)
        print(f"✅ Embedded {indexed:,} documents")

    elif args.command == "query":
        result = process_query(args.question, use_llm=not args.no_llm)
        print(json.dumps({
            "qid": result.qid,
            "original_query": result.original_query,
            "normalized_query": result.normalized_query,
            "rewrite_query": result.rewrite_query,
            "bm25_query": result.bm25_query,
            "intent": result.intent,
            "filters": result.filters,
            "risk_flags": result.risk_flags,
        }, ensure_ascii=False, indent=2))

    elif args.command == "search":
        result = search(
            question=args.question,
            use_qdrant=not args.no_qdrant,
            use_reranker=args.rerank,
        )
        print(json.dumps({
            "query": result.query.original_query,
            "passages": [
                {
                    "rank": p.rank,
                    "title": p.title,
                    "text": p.text[:200] + "..." if len(p.text) > 200 else p.text,
                    "source_url": p.source_url,
                    "scores": {
                        "dense": round(p.dense_score, 4),
                        "bm25": round(p.bm25_score, 4),
                        "rrf": round(p.rrf_score, 4),
                        "rerank": round(p.rerank_score, 4),
                    },
                }
                for p in result.passages
            ],
            "context": result.context[:500] + "..." if len(result.context) > 500 else result.context,
            "metadata": result.metadata,
        }, ensure_ascii=False, indent=2))

    elif args.command == "ask":
        # LangSmith tracing auto-enabled when LANGSMITH_TRACING_V2=true in .env
        if args.stream:
            # Streaming mode: print tokens as they arrive
            import sys

            pipeline = build_ask_pipeline(
                use_qdrant=not args.no_qdrant,
                use_reranker=args.rerank,
                use_llm=not args.no_llm,
            )

            # Run query + retrieval first
            processed_query = pipeline._run_query_processing(args.question)
            retrieval_result = pipeline._run_retrieval(processed_query)

            # Stream generation
            chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval_result)

            full_text = ""
            for chunk in chunk_gen:
                full_text += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()

            # Build final result with guardrails
            answer_result = build_result(full_text)
            result = pipeline._run_output_guardrails(answer_result, retrieval_result)

            # Print metadata
            if args.text:
                source_url = result.citations[0].source_url if result.citations else ""
                if source_url:
                    print(f"\n\nNguồn: {source_url}")
            else:
                print("\n")
                print(json.dumps({
                    "question": result.question,
                    "confidence": round(result.confidence, 4),
                    "passages_used": result.passages_used,
                }, ensure_ascii=False, indent=2))
        else:
            # Non-streaming mode
            result = ask(
                question=args.question,
                use_qdrant=not args.no_qdrant,
                use_reranker=args.rerank,
                use_llm=not args.no_llm,
            )

            if args.text:
                # User-friendly output: answer + 1 source
                source_url = result.citations[0].source_url if result.citations else ""
                print(result.answer)
                if source_url:
                    print(f"\nNguồn: {source_url}")
            else:
                # Full JSON output (for developers/API)
                print(json.dumps({
                    "question": result.question,
                    "answer": result.answer,
                    "citations": [
                        {
                            "claim": c.claim,
                            "title": c.title,
                            "source_url": c.source_url,
                            "confidence": round(c.confidence, 4),
                        }
                        for c in result.citations
                    ],
                    "confidence": round(result.confidence, 4),
                    "passages_used": result.passages_used,
                    "metadata": result.metadata,
                }, ensure_ascii=False, indent=2))

    elif args.command == "eval":
        from rag_pipeline.eval.runner import EvalRunner

        # Build pipeline
        eval_config = EvalConfig(eval_dataset_path=Path(args.dataset))
        pipeline = build_generation_pipeline(
            retrieval_pipeline=build_retrieval_pipeline(use_qdrant=not args.no_qdrant),
            query_pipeline=build_query_pipeline(QueryConfig(), use_llm=True),
        )

        runner = EvalRunner(pipeline=pipeline, config=eval_config)
        report = runner.run(limit=args.limit)

        # Export report
        report.print_summary()
        report.to_json(Path(args.output))
        report.to_markdown(Path(args.output).with_suffix(".md"))
        print(f"\n📄 Report saved to: {args.output}")

    else:
        parser.print_help()
