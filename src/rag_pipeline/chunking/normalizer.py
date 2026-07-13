"""Document normalization stage."""

from __future__ import annotations

import re
import unicodedata

from rag_pipeline.chunking.base import Normalizer
from rag_pipeline.chunking.models import NormalizedDocument, RawDocument


class DocumentNormalizer(Normalizer):
    """Normalize unicode, line endings, and whitespace."""

    def normalize(self, document: RawDocument) -> NormalizedDocument:
        content = document.raw_content or ""
        content = unicodedata.normalize("NFKC", content)
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)

        return NormalizedDocument(
            document_id=document.document_id,
            source_id=document.source_id,
            title=document.title.strip(),
            content=content.strip(),
            metadata=document.metadata,
        )
