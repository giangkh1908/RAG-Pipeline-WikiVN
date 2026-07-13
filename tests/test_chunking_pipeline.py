"""Tests for the chunking pipeline."""

from uuid import uuid4

from rag_pipeline.chunking import ChunkingPipeline, RawDocument, RecursiveChunker
from rag_pipeline.chunking.cleaner import DocumentCleaner
from rag_pipeline.chunking.section_detector import HeadingSectionDetector


class TestChunkingPipeline:
    def test_pipeline_produces_valid_chunks(self) -> None:
        long_content = (
            "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á, "
            "phía bắc giáp Trung Quốc, phía tây giáp Lào và Campuchia. "
            "Đây là quốc gia có dân số đông thứ ba trong khu vực.\n\n"
            "#Lịch sử\n\n"
            "Lịch sử Việt Nam bắt đầu từ thờikỳ các vua Hùng dựng nước, "
            "trải qua nhiều thờikỳ phong kiến và các cuộc chiến tranh bảo vệ tổ quốc."
        )
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Việt Nam",
            raw_content=long_content,
        )
        pipeline = ChunkingPipeline(chunker=RecursiveChunker(max_tokens=50))

        chunks = pipeline.process(document)

        assert len(chunks) >= 1
        assert all(c.document_id == document.document_id for c in chunks)
        assert all(c.token_count >= 20 for c in chunks)

    def test_cleaner_removes_wikilinks(self) -> None:
        cleaner = DocumentCleaner()
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Test",
            raw_content=(
                "Xem [[Việt Nam|Việt Nam]] và [[Lịch sử]] để hiểu thêm về "
                "quá trình hình thành và phát triển của đất nước qua nhiều thế kỷ "
                "với nhiều biến cố lịch sử quan trọng."
            ),
        )
        pipeline = ChunkingPipeline(
            chunker=RecursiveChunker(max_tokens=50),
            cleaner=cleaner,
        )

        chunks = pipeline.process(document)

        assert len(chunks) == 1
        assert "[[" not in chunks[0].content
        assert "Việt Nam" in chunks[0].content

    def test_section_detector_splits_markdown_headings(self) -> None:
        detector = HeadingSectionDetector()
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Test",
            raw_content=(
                "Đây là phần mở đầu giới thiệu tổng quan về chủ đề.\n"
                "# Section A\n"
                "Nội dung của section A rất dài và chứa nhiều thông tin quan trọng "
                "cần được trích xuất để phục vụ cho việc tìm kiếm sau này.\n"
                "# Section B\n"
                "Nội dung của section B cũng rất dài và có nhiều chi tiết thú vị "
                "giúp ngườidùng hiểu rõ hơn về vấn đề đang được thảo luận."
            ),
        )
        pipeline = ChunkingPipeline(
            chunker=RecursiveChunker(max_tokens=50),
            section_detector=detector,
        )

        chunks = pipeline.process(document)

        assert len(chunks) >= 2
        titles = {c.metadata.get("section_title", "") for c in chunks}
        assert "Section A" in titles or any("Section A" in c.content for c in chunks)

    def test_validator_drops_empty_chunks(self) -> None:
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Test",
            raw_content="A " * 5,  # Too short, should be filtered
        )
        pipeline = ChunkingPipeline(chunker=RecursiveChunker(max_tokens=50))

        chunks = pipeline.process(document)

        assert len(chunks) == 0
