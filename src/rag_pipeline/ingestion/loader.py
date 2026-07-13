"""Loaders for external datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RawParagraph:
    """A raw paragraph from a dataset topic."""

    title: str
    context: str

    def to_raw_document(self, document_id, source_id, title):
        from rag_pipeline.chunking.models import RawDocument

        return RawDocument(
            document_id=document_id,
            source_id=source_id,
            title=title,
            raw_content=self.context,
            metadata={"topic_title": self.title},
        )


@dataclass
class RawTopic:
    """A raw topic from a dataset."""

    title: str
    paragraphs: list[RawParagraph]


class VietnamTourismLoader:
    """Load topics and paragraphs from the Vietnam Tourism v2 JSON format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Iterable[RawTopic]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for topic in data.get("data", []):
            title = topic.get("title", "")
            paragraphs = [
                RawParagraph(title=title, context=p.get("context", ""))
                for p in topic.get("paragraphs", [])
                if p.get("context", "").strip()
            ]
            if paragraphs:
                yield RawTopic(title=title, paragraphs=paragraphs)
