"""Tests for BM25 index."""

import tempfile
from pathlib import Path

from rag_pipeline.indexing.bm25_index import BM25Index


class TestBM25Index:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_path = Path(self.tmpdir) / "bm25.pkl"

    def test_build_and_search(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        documents = [
            ("doc1", "thủ đô của việt nam là hà nội"),
            ("doc2", "dân số trung quốc rất lớn"),
            ("doc3", "sơn tùng mtp là ca sĩ nổi tiếng"),
        ]
        index.build(documents)

        results = index.search("thủ đô việt nam", top_k=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"  # doc1 should be top result

    def test_load_save(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        documents = [
            ("doc1", "hello world"),
            ("doc2", "foo bar"),
        ]
        index.build(documents)

        # Load in new instance
        index2 = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        assert index2.load() is True
        assert index2.doc_count == 2

    def test_load_nonexistent(self):
        index = BM25Index(index_path=Path("/nonexistent/bm25.pkl"))
        assert index.load() is False

    def test_search_before_load_raises(self):
        index = BM25Index(index_path=self.index_path)
        import pytest
        # Empty index returns empty results
        assert index.search("test") == []
        # Non-empty index without load raises
        index._doc_ids = ["doc1"]
        with pytest.raises(RuntimeError, match="not loaded"):
            index.search("test")

    def test_empty_corpus(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        index.build([])
        results = index.search("test")
        assert results == []


class TestBM25Tokenizer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.index_path = Path(self.tmpdir) / "bm25.pkl"

    def test_simple_tokenizer(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="simple")
        tokens = index._tokenize("Thủ đô Việt Nam")
        assert tokens == ["thủ", "đô", "việt", "nam"]

    def test_fallback_tokenizer(self):
        index = BM25Index(index_path=self.index_path, tokenizer_name="unknown")
        tokens = index._tokenize("Hello World")
        assert tokens == ["hello", "world"]
