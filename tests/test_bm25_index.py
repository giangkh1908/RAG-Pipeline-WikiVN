"""Tests for BM25 index (SQLite FTS5 backend)."""

import tempfile
from pathlib import Path

from rag_pipeline.indexing.bm25_index import BM25Index


class TestBM25Index:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_path = Path(self.tmpdir) / "bm25.db"

    def test_insert_and_search(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.insert("c1", "d1", "thủ đô của việt nam là hà nội", "full text 1", [], "cs1")
        index.insert("c2", "d1", "dân số trung quốc rất lớn", "full text 2", [], "cs1")
        index.insert("c3", "d2", "sơn tùng mtp là ca sĩ nổi tiếng", "full text 3", [], "cs2")

        results = index.search("thủ đô", top_k=2)
        assert len(results) >= 1
        # chunk_id, doc_id, score, full_text
        assert results[0][0] == "c1"  # c1 should be top result

    def test_insert_batch(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        items = [
            {"chunk_id": "c1", "doc_id": "d1", "raw_content": "hello world", "full_text": "full 1", "section_path": [], "checksum": ""},
            {"chunk_id": "c2", "doc_id": "d1", "raw_content": "foo bar", "full_text": "full 2", "section_path": [], "checksum": ""},
        ]
        count = index.insert_batch(items)
        assert count == 2
        assert index.doc_count == 2

    def test_save_and_reopen(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.insert("c1", "d1", "hello world", "full 1", [], "")
        index.insert("c2", "d1", "foo bar", "full 2", [], "")
        index.close()

        # Reopen in new instance
        index2 = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        assert index2.is_loaded is True
        assert index2.doc_count == 2

        results = index2.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "c1"
        index2.close()

    def test_open_nonexistent(self):
        index = BM25Index(index_path=Path("/nonexistent/bm25.db"))
        assert index.is_loaded is False
        assert index.doc_count == 0

    def test_search_empty_index(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        results = index.search("test", top_k=5)
        assert results == []

    def test_empty_corpus(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.build([])
        results = index.search("test")
        assert results == []

    def test_search_returns_full_text(self):
        """Search results should include full_text for RRF fusion."""
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.insert("c1", "d1", "test content", "This is the full text", [], "")

        results = index.search("test", top_k=1)
        assert len(results) == 1
        chunk_id, doc_id, score, full_text = results[0]
        assert full_text == "This is the full text"

    def test_incremental_insert(self):
        """Multiple inserts should accumulate, not replace."""
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.insert("c1", "d1", "first document", "full 1", [], "")
        assert index.doc_count == 1

        index.insert("c2", "d2", "second document", "full 2", [], "")
        assert index.doc_count == 2

        # Search should find both
        results = index.search("document", top_k=10)
        assert len(results) == 2


class TestBM25Tokenizer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_path = Path(self.tmpdir) / "bm25.db"

    def test_simple_tokenizer(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        tokens = index._tokenize("Thủ đô Việt Nam")
        assert tokens == "thủ đô việt nam"

    def test_fallback_tokenizer(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="unknown")
        tokens = index._tokenize("Hello World")
        assert tokens == "hello world"
