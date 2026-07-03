"""Evaluation endpoint."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from rag_pipeline.api.schemas import EvalRequest, EvalResponse

router = APIRouter(tags=["eval"])


@router.post("/eval", response_model=EvalResponse)
async def run_eval(request: EvalRequest) -> EvalResponse:
    """Run RAGAS evaluation on the pipeline.

    Args:
        request: EvalRequest with dataset path, limit, and options

    Returns:
        EvalResponse with quality scores and latency metrics
    """
    from rag_pipeline.config import EvalConfig, QueryConfig
    from rag_pipeline.eval.runner import EvalRunner
    from rag_pipeline.main import (
        build_generation_pipeline,
        build_query_pipeline,
        build_retrieval_pipeline,
    )

    # Build pipeline
    query_pipeline = build_query_pipeline(QueryConfig(), use_llm=True)
    retrieval_pipeline = build_retrieval_pipeline(use_qdrant=request.use_qdrant)
    pipeline = build_generation_pipeline(
        retrieval_pipeline=retrieval_pipeline,
        query_pipeline=query_pipeline,
    )

    # Run evaluation
    eval_config = EvalConfig(eval_dataset_path=Path(request.dataset))
    runner = EvalRunner(pipeline=pipeline, config=eval_config)
    report = runner.run(limit=request.limit)

    return EvalResponse(
        scores=report.scores,
        latency=report.latency_summary,
        sample_count=len(report.samples),
        passed=report._check_thresholds(),
    )
