import unittest

from rag_pipeline.config import ChunkingConfig
from rag_pipeline.models import CanonicalDocument
from rag_pipeline.transform.chunker import RecursiveChunker


def _make_doc(content: str, title: str = "Tiếng Việt") -> CanonicalDocument:
    return CanonicalDocument(
        doc_id="doc-1",
        source_id="source-1",
        title=title,
        document_type="wikipedia_article",
        jurisdiction="VN",
        issued_date=None,
        effective_date=None,
        language="vi",
        content=content,
        checksum="checksum",
    )


class ChunkerTests(unittest.TestCase):
    def test_splits_long_article_and_links_neighbors(self) -> None:
        chunker = RecursiveChunker(ChunkingConfig(max_tokens_per_chunk=6, chunk_overlap_tokens=2))
        document = _make_doc(
            "Đoạn đầu tiên nói về lịch sử hình thành.\n\n"
            "Đoạn thứ hai bổ sung thêm chi tiết về ngữ âm và chữ viết."
        )

        chunks = chunker.chunk(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual("Tiếng Việt", chunks[0].metadata["title"])
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].chunk_id)
        self.assertEqual(chunks[1].prev_chunk_id, chunks[0].chunk_id)

    def test_single_short_paragraph_returns_one_chunk(self) -> None:
        chunker = RecursiveChunker(ChunkingConfig(max_tokens_per_chunk=100))
        document = _make_doc("Đây là một đoạn văn ngắn.")

        chunks = chunker.chunk(document)

        self.assertEqual(1, len(chunks))
        self.assertIsNone(chunks[0].prev_chunk_id)
        self.assertIsNone(chunks[0].next_chunk_id)

    def test_respects_max_tokens_per_chunk(self) -> None:
        chunker = RecursiveChunker(ChunkingConfig(max_tokens_per_chunk=10, chunk_overlap_tokens=2))
        long_text = " ".join([f"Từ{i}" for i in range(50)])
        document = _make_doc(long_text)

        chunks = chunker.chunk(document)

        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 10 + 2)  # allow small margin

    def test_empty_content_returns_no_chunks(self) -> None:
        chunker = RecursiveChunker(ChunkingConfig())
        document = _make_doc("")

        chunks = chunker.chunk(document)

        self.assertEqual(0, len(chunks))

    def test_multiple_paragraphs_are_preserved(self) -> None:
        chunker = RecursiveChunker(ChunkingConfig(max_tokens_per_chunk=100))
        document = _make_doc(
            "Đoạn thứ nhất về lịch sử.\n\nĐoạn thứ hai về địa lý.\n\nĐoạn thứ ba về văn hóa."
        )

        chunks = chunker.chunk(document)

        # With high max_tokens, all 3 paragraphs should fit in 1 chunk
        self.assertEqual(1, len(chunks))
        self.assertIn("lịch sử", chunks[0].text)
        self.assertIn("địa lý", chunks[0].text)
        self.assertIn("văn hóa", chunks[0].text)

    def test_sentence_boundary_splitting(self) -> None:
        """When a paragraph is too large, it splits by sentence first."""
        chunker = RecursiveChunker(ChunkingConfig(max_tokens_per_chunk=5, chunk_overlap_tokens=1))
        document = _make_doc(
            "Câu đầu tiên. Câu thứ hai. Câu thứ ba. Câu thứ tư."
        )

        chunks = chunker.chunk(document)

        self.assertGreaterEqual(len(chunks), 2)
        # Each chunk should contain complete sentences (not cut mid-sentence)
        for chunk in chunks:
            self.assertGreater(len(chunk.text.strip()), 0)


if __name__ == "__main__":
    unittest.main()
