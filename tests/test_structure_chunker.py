import unittest

from rag_pipeline.config import ChunkingConfig
from rag_pipeline.models import CanonicalDocument
from rag_pipeline.transform.structure_chunker import StructuredChunker


def _make_doc(content: str, title: str = "Việt Nam") -> CanonicalDocument:
    return CanonicalDocument(
        doc_id="doc-1",
        source_id="src-1",
        title=title,
        document_type="wikipedia_article",
        jurisdiction="VN",
        issued_date=None,
        effective_date=None,
        language="vi",
        content=content,
        checksum="checksum1",
        source_url="https://vi.wikipedia.org/wiki/Việt_Nam",
    )


class HeadingDetectionTests(unittest.TestCase):
    """Test _looks_like_heading heuristic."""

    def test_short_standalone_text_is_heading(self) -> None:
        self.assertTrue(StructuredChunker._looks_like_heading("Lịch sử"))
        self.assertTrue(StructuredChunker._looks_like_heading("Địa lý"))
        self.assertTrue(StructuredChunker._looks_like_heading("Thân thế"))
        self.assertTrue(StructuredChunker._looks_like_heading("Vị trí địa lý"))

    def test_long_text_is_not_heading(self) -> None:
        self.assertFalse(StructuredChunker._looks_like_heading(
            "Đây là một câu văn rất dài không thể là heading được"
        ))

    def test_list_item_is_not_heading(self) -> None:
        self.assertFalse(StructuredChunker._looks_like_heading("* Điểm cực bắc"))
        self.assertFalse(StructuredChunker._looks_like_heading("- Hà Nội"))
        self.assertFalse(StructuredChunker._looks_like_heading("1. Mục đầu tiên"))

    def test_sentence_ending_is_not_heading(self) -> None:
        self.assertFalse(StructuredChunker._looks_like_heading("Kết thúc."))
        self.assertFalse(StructuredChunker._looks_like_heading("Tại sao?"))
        self.assertFalse(StructuredChunker._looks_like_heading("Như sau:"))

    def test_url_is_not_heading(self) -> None:
        self.assertFalse(StructuredChunker._looks_like_heading("https://example.com"))

    def test_lowercase_start_is_not_heading(self) -> None:
        self.assertFalse(StructuredChunker._looks_like_heading("một vài thứ"))
        self.assertFalse(StructuredChunker._looks_like_heading("và sau đó"))


class ReferenceDetectionTests(unittest.TestCase):
    """Test _is_reference_heading."""

    def test_known_reference_headings(self) -> None:
        self.assertTrue(StructuredChunker._is_reference_heading("Tham khảo"))
        self.assertTrue(StructuredChunker._is_reference_heading("Liên kết ngoài"))
        self.assertTrue(StructuredChunker._is_reference_heading("Xem thêm"))
        self.assertTrue(StructuredChunker._is_reference_heading("Chú thích"))

    def test_normal_heading_is_not_reference(self) -> None:
        self.assertFalse(StructuredChunker._is_reference_heading("Lịch sử"))
        self.assertFalse(StructuredChunker._is_reference_heading("Địa lý"))


class ContextTests(unittest.TestCase):
    """Test natural-language contextual prefix."""

    def test_doc_only(self) -> None:
        result = StructuredChunker._build_context("Việt Nam", "", ["Việt Nam"])
        self.assertIn("This chunk is from the 'Việt Nam' document", result)
        self.assertTrue(result.endswith("."))

    def test_doc_with_summary(self) -> None:
        result = StructuredChunker._build_context(
            "Việt Nam", "Việt Nam là một quốc gia nằm ở Đông Nam Á.", ["Việt Nam"])
        self.assertIn("which describes", result)
        self.assertIn("Đông Nam Á", result)

    def test_doc_with_section(self) -> None:
        result = StructuredChunker._build_context(
            "Việt Nam", "Việt Nam là một quốc gia.", ["Việt Nam", "Lịch sử"])
        self.assertIn("specifically the 'Lịch sử' section", result)

    def test_doc_with_nested_section(self) -> None:
        result = StructuredChunker._build_context(
            "Việt Nam", "Việt Nam là một quốc gia.",
            ["Việt Nam", "Lịch sử", "Thời Lý"])
        self.assertIn("specifically the 'Lịch sử > Thời Lý' section", result)

    def test_summary_truncation(self) -> None:
        long_summary = "X " * 80
        result = StructuredChunker._build_context("Test", long_summary, ["Test"])
        self.assertLess(len(result), 300)


class ChunkerTests(unittest.TestCase):
    """Integration tests for full chunking pipeline."""

    def test_empty_content_returns_no_chunks(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc("")
        chunks = chunker.chunk(doc)
        self.assertEqual(0, len(chunks))

    def test_single_paragraph_returns_one_chunk_with_context(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc("Hà Nội là thủ đô của Việt Nam, nằm ở phía bắc.")
        chunks = chunker.chunk(doc)

        self.assertEqual(1, len(chunks))
        self.assertIn("This chunk is from the 'Việt Nam' document", chunks[0].text)
        self.assertIn("Hà Nội là thủ đô", chunks[0].text)

    def test_heading_splits_into_separate_chunks(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc(
            "Hà Nội là thủ đô của Việt Nam, nằm ở phía bắc.\n\n"
            "Lịch sử\n\n"
            "Hà Nội có lịch sử hơn 1000 năm văn hiến."
        )
        chunks = chunker.chunk(doc)

        self.assertGreaterEqual(len(chunks), 2)
        # First chunk should have doc-level context
        self.assertIn("This chunk is from the 'Việt Nam' document", chunks[0].text)
        # Second chunk should mention the Lịch sử section
        self.assertIn("Lịch sử", chunks[1].text)

    def test_nested_headings_build_section_path(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc(
            "Mở đầu bài viết.\n\n"
            "Thân thế\n\n"
            "Ông sinh ra trong một gia đình.\n\n"
            "Dòng dõi\n\n"
            "Gia đình ông có truyền thống."
        )
        chunks = chunker.chunk(doc)

        paths = [c.section_path for c in chunks]
        self.assertIn(["Việt Nam", "Thân thế"], paths)
        self.assertIn(["Việt Nam", "Thân thế", "Dòng dõi"], paths)

    def test_reference_section_is_marked(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc(
            "Nội dung chính của bài viết.\n\n"
            "Tham khảo\n\n"
            "Nguyễn Văn A. Sách Lịch sử. 2020."
        )
        chunks = chunker.chunk(doc)

        # Find the reference chunk
        ref_chunks = [c for c in chunks if c.metadata.get("is_reference_section")]
        self.assertGreaterEqual(len(ref_chunks), 1)

    def test_list_items_stay_together(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc(
            "Địa lý\n\n"
            "Tỉnh có các điểm cực:\n"
            "* Điểm cực bắc: xã A\n"
            "* Điểm cực nam: xã B\n"
            "* Điểm cực đông: xã C\n"
            "* Điểm cực tây: xã D"
        )
        chunks = chunker.chunk(doc)

        # All list items should be in one chunk (small list)
        geog_chunks = [c for c in chunks if "Địa lý" in c.section_path]
        self.assertGreaterEqual(len(geog_chunks), 1)
        # Check list items are together
        list_text = geog_chunks[0].text
        self.assertIn("Điểm cực bắc", list_text)
        self.assertIn("Điểm cực nam", list_text)

    def test_neighbors_are_linked(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=50))
        doc = _make_doc(
            "Lịch sử\n\n"
            + " ".join(["từ"] * 60) + "\n\n"
            "Địa lý\n\n"
            + " ".join(["từ"] * 60)
        )
        chunks = chunker.chunk(doc)

        self.assertGreaterEqual(len(chunks), 2)
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i].next_chunk_id, chunks[i + 1].chunk_id)
            self.assertEqual(chunks[i + 1].prev_chunk_id, chunks[i].chunk_id)

    def test_chunk_id_is_deterministic(self) -> None:
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=300))
        doc = _make_doc("Nội dung bài viết.")

        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)

        self.assertEqual(chunks1[0].chunk_id, chunks2[0].chunk_id)

    def test_large_paragraph_splits_without_crossing_heading(self) -> None:
        """Large text gets split but still within section boundaries."""
        chunker = StructuredChunker(ChunkingConfig(max_tokens_per_chunk=10))
        doc = _make_doc(
            "đoạn một có nội dung dài về lịch sử hình thành phát triển\n\n"
            "Văn hóa\n\n"
            "đoạn hai có nội dung về văn hóa nghệ thuật truyền thống"
        )
        chunks = chunker.chunk(doc)

        # All chunks under "Văn hóa" must have that in path
        for c in chunks:
            if "văn hóa" in c.text.lower():
                self.assertIn("Văn hóa", c.section_path)


if __name__ == "__main__":
    unittest.main()
