"""Tests for the structure-aware chunker."""

from uuid import uuid4

from rag_pipeline.chunking import ChunkingPipeline, RawDocument, StructureChunker


class TestStructureChunker:
    def test_keeps_list_items_together(self) -> None:
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Việt Nam",
            raw_content=(
                "Địa lý Việt Nam rất đa dạng với nhiều vùng miền khác nhau "
                "từ bắc vào nam, từ đồng bằng đến miền núi.\n\n"
                "* Điểm cực bắc: xã Lũng Cú\n"
                "* Điểm cực nam: đảo Phú Quý\n"
                "* Điểm cực đông: mũi Đôi\n"
                "* Điểm cực tây: xã Sín Thâu"
            ),
        )
        pipeline = ChunkingPipeline(chunker=StructureChunker(max_tokens=100))

        chunks = pipeline.process(document)

        assert len(chunks) >= 1
        list_chunk = [c for c in chunks if "Điểm cực bắc" in c.content]
        assert len(list_chunk) >= 1
        assert "Điểm cực nam" in list_chunk[0].content

    def test_adds_context_prefix(self) -> None:
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Việt Nam",
            raw_content=(
                "#Lịch sử\n\n"
                "Việt Nam có lịch sử hơn bốn nghìn năm với nhiều triều đại "
                "và các cuộc chiến tranh bảo vệ độclập."
            ),
        )
        pipeline = ChunkingPipeline(chunker=StructureChunker(max_tokens=100))

        chunks = pipeline.process(document)

        assert len(chunks) >= 1
        assert "This chunk is from the 'Việt Nam' document" in chunks[0].content
        assert "Lịch sử" in chunks[0].content

    def test_mark_reference_section(self) -> None:
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Việt Nam",
            raw_content=(
                "Nội dung chính của bài viết rất dài và có nhiều thông tin "
                "quan trọng cần được ghi nhớ.\n\n"
                "Tham khảo\n\n"
                "Nguyễn Văn A. Sách Lịch sử Việt Nam. Nhà xuất bản Văn học."
            ),
        )
        pipeline = ChunkingPipeline(chunker=StructureChunker(max_tokens=100))

        chunks = pipeline.process(document)

        ref_chunks = [c for c in chunks if c.metadata.get("is_reference_section")]
        assert len(ref_chunks) >= 1

    def test_respects_section_boundaries(self) -> None:
        document = RawDocument(
            document_id=uuid4(),
            source_id=uuid4(),
            title="Việt Nam",
            raw_content=(
                "#Địa lý\n\n"
                "Việt Nam nằm ở Đông Nam Á với nhiều địa hình đa dạng "
                "từ đồng bằng sông Hồng đến cao nguyên Tây Nguyên.\n\n"
                "#Lịch sử\n\n"
                "Lịch sử Việt Nam trải dài hàng nghìn năm với nhiều triều đại."
            ),
        )
        pipeline = ChunkingPipeline(chunker=StructureChunker(max_tokens=100))

        chunks = pipeline.process(document)

        assert len(chunks) >= 2
        sections = {tuple(c.metadata.get("section_path", [])) for c in chunks}
        assert any("Địa lý" in path for path in sections)
        assert any("Lịch sử" in path for path in sections)
