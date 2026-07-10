import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_pipeline.config import ChunkingConfig, IngestConfig
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import DeterministicTestEmbedder
from rag_pipeline.indexing.vector_store import InMemoryVectorStore
from rag_pipeline.ingest.dataset import (
    HuggingFaceDatasetReader,
    LocalCorpusCsvReader,
    LocalJsonlReader,
    LocalQueryCsvReader,
)
from rag_pipeline.ingest.normalize import UVWWikipediaDocumentNormalizer
from rag_pipeline.main import build_ingest_pipeline, ingest
from rag_pipeline.models import SourceRecord
from rag_pipeline.pipelines.ingest_pipeline import IngestPipeline
from rag_pipeline.transform.chunker import RecursiveChunker
from rag_pipeline.transform.cleaner import WikipediaArticleCleaner


class PipelineTests(unittest.TestCase):
    def test_pipeline_is_idempotent_for_identical_records(self) -> None:
        vector_store = InMemoryVectorStore()
        bm25_index = BM25Index(index_path=Path(tempfile.mktemp(suffix='.db')))
        pipeline = IngestPipeline(
            normalizer=UVWWikipediaDocumentNormalizer(),
            cleaner=WikipediaArticleCleaner(),
            chunker=RecursiveChunker(ChunkingConfig()),
            embedder=DeterministicTestEmbedder(),
            vector_store=vector_store,
            bm25_index=bm25_index,
            bm25_index=bm25_index,
        )
        record = SourceRecord(
            source_id="record-1",
            payload={
                "id": "Viet_Nam",
                "title": "Việt Nam",
                "content": "Việt Nam là một quốc gia ở Đông Nam Á.",
                "quality": 9,
                "wikidata_id": "Q881",
                "main_category": "quốc gia có chủ quyền",
            },
        )

        first = pipeline.run([record])
        second = pipeline.run([record])

        self.assertTrue(first[0].updated)
        self.assertFalse(second[0].updated)
        self.assertEqual(1, len(vector_store.documents))

    def test_local_corpus_reader_emits_source_records(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "corpus.csv"
            csv_path.write_text('text,cid\n"Điều 1",123\n', encoding="utf-8")

            records = list(LocalCorpusCsvReader(csv_path).read())

            self.assertEqual(1, len(records))
            self.assertEqual("123", records[0].source_id)
            self.assertEqual("Điều 1", records[0].payload["content"])

    def test_local_jsonl_reader_emits_source_records(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "train.jsonl"
            jsonl_path.write_text(
                '{"id":"Tieng_Viet","title":"Tiếng Việt","content":"Bài viết mẫu.","num_chars":12,"num_sentences":1,"quality":9}\n',
                encoding="utf-8",
            )

            records = list(LocalJsonlReader(jsonl_path).read())

            self.assertEqual(1, len(records))
            self.assertEqual("Tieng_Viet", records[0].source_id)
            self.assertEqual("Tiếng Việt", records[0].payload["title"])

    def test_local_query_reader_keeps_queries_separate_from_corpus(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "val.csv"
            csv_path.write_text(
                "question,context,cid,qid\n"
                "\"Hỏi gì?\",\"['Đoạn 1']\",\"[42]\",q-1\n",
                encoding="utf-8",
            )

            queries = list(LocalQueryCsvReader(csv_path).read())

            self.assertEqual(1, len(queries))
            self.assertEqual("q-1", queries[0].qid)
            self.assertEqual(["Đoạn 1"], queries[0].context)
            self.assertEqual(["42"], queries[0].cids)

    def test_huggingface_reader_resolves_percentage_split(self) -> None:
        reader = HuggingFaceDatasetReader(
            dataset_name="undertheseanlp/UVW-2026",
            split="train",
            sample_percent=1.0,
        )
        self.assertEqual("train[:1.0%]", reader._resolved_split())

    def test_ingest_dispatches_by_source_type(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "train.jsonl"
            jsonl_path.write_text(
                '{"id":"Tieng_Viet","title":"Tiếng Việt","content":"Bài viết mẫu.","num_chars":12,"num_sentences":1,"quality":9}\n',
                encoding="utf-8",
            )

            config = IngestConfig(source_type="local_jsonl", jsonl_path=jsonl_path)
            document_ids = ingest(config)

            self.assertEqual(1, len(document_ids))


if __name__ == "__main__":
    unittest.main()
