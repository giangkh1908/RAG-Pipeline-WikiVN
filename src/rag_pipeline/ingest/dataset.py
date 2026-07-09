from __future__ import annotations

import ast
import csv
import gzip
import json
import struct
import sys
from collections.abc import Iterable
from pathlib import Path

from rag_pipeline.models import QueryRecord, SourceRecord

# Offset index header: magic (4B) + version (4B) + count (8B) = 16 bytes
_IDX_MAGIC = b"RGOF"
_IDX_VERSION = 1
_IDX_HEADER_FMT = "<4sIQ"  # magic(4) + version(I=4) + count(Q=8)
_IDX_ENTRY_FMT = "<Q"      # each offset is uint64
_IDX_HEADER_SIZE = struct.calcsize(_IDX_HEADER_FMT)
_IDX_ENTRY_SIZE = struct.calcsize(_IDX_ENTRY_FMT)


class HuggingFaceDatasetReader:
    """Thin adapter around `datasets.load_dataset` to keep the pipeline testable."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        sample_percent: float = 100.0,
        min_quality_score: int | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.sample_percent = sample_percent
        self.min_quality_score = min_quality_score

    def read(self) -> Iterable[SourceRecord]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The `datasets` package is required to load Hugging Face sources. "
                "Install with `pip install .[ingest]`."
            ) from exc

        split = self._resolved_split()
        dataset = load_dataset(self.dataset_name, split=split)
        if self.min_quality_score is not None and "quality_score" in dataset.column_names:
            dataset = dataset.filter(lambda row: row["quality_score"] >= self.min_quality_score)
        for index, row in enumerate(dataset):
            source_id = str(row.get("id") or row.get("document_id") or index)
            yield SourceRecord(source_id=source_id, payload=dict(row))

    def _resolved_split(self) -> str:
        if self.sample_percent <= 0 or self.sample_percent >= 100:
            return self.split
        return f"{self.split}[:{self.sample_percent}%]"


class LocalCorpusCsvReader:
    """Reads the passage corpus used for RAG indexing."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def read(self) -> Iterable[SourceRecord]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cid = str(row["cid"]).strip()
                text = str(row["text"]).strip()
                yield SourceRecord(
                    source_id=cid,
                    payload={
                        "cid": cid,
                        "title": f"corpus-{cid}",
                        "content": text,
                        "document_type": "passage",
                    },
                )


class LocalJsonlReader:
    """Reads line-delimited JSON article corpora with optional offset index.

    On first read, a `.idx` file is created alongside the JSONL.
    Subsequent reads use the index for fast seeking — especially
    beneficial when sample_percent < 100.
    """

    def __init__(self, jsonl_path: str | Path, sample_percent: float = 100.0) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.sample_percent = sample_percent

    def read(self) -> Iterable[SourceRecord]:
        index_path = self._index_path()

        if index_path.exists():
            yield from self._read_with_index(index_path)
        else:
            yield from self._read_and_build_index(index_path)

    # ── Index-based reading (fast) ─────────────────────────────────

    def _read_with_index(self, index_path: Path) -> Iterable[SourceRecord]:
        offsets = self._load_index(index_path)
        total = len(offsets)
        selected = self._select_offsets(offsets)

        with self.jsonl_path.open("rb") as handle:
            for idx, byte_offset in selected:
                handle.seek(byte_offset)
                line = handle.readline().decode("utf-8-sig").strip()
                if not line:
                    continue
                payload = json.loads(line)
                source_id = str(payload.get("id") or payload.get("document_id") or idx)
                yield SourceRecord(source_id=source_id, payload=payload)

    def _select_offsets(self, offsets: list[int]) -> list[tuple[int, int]]:
        """Return [(line_index, byte_offset), ...] for lines to include."""
        if self.sample_percent >= 100:
            return list(enumerate(offsets))
        if self.sample_percent <= 0:
            return []

        stride = max(1, round(100 / self.sample_percent))
        return [(i, offsets[i]) for i in range(0, len(offsets), stride)]

    # ── Scan + build index (first run) ─────────────────────────────

    def _read_and_build_index(self, index_path: Path) -> Iterable[SourceRecord]:
        offsets: list[int] = []
        stride = max(1, round(100 / self.sample_percent)) if 0 < self.sample_percent < 100 else 1

        with self.jsonl_path.open("rb") as handle:
            idx = 0
            while True:
                byte_offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break

                offsets.append(byte_offset)

                if not self._should_include(idx, stride):
                    idx += 1
                    continue

                line = raw.decode("utf-8-sig").strip()
                if not line:
                    idx += 1
                    continue

                payload = json.loads(line)
                source_id = str(payload.get("id") or payload.get("document_id") or idx)
                yield SourceRecord(source_id=source_id, payload=payload)
                idx += 1

        self._save_index(index_path, offsets)

    # ── Index file I/O ─────────────────────────────────────────────

    def _index_path(self) -> Path:
        return self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".idx")

    def _save_index(self, path: Path, offsets: list[int]) -> None:
        with path.open("wb") as f:
            header = struct.pack(_IDX_HEADER_FMT, _IDX_MAGIC, _IDX_VERSION, len(offsets))
            f.write(header)
            for offset in offsets:
                f.write(struct.pack(_IDX_ENTRY_FMT, offset))

    def _load_index(self, path: Path) -> list[int]:
        with path.open("rb") as f:
            header = f.read(_IDX_HEADER_SIZE)
            magic, version, count = struct.unpack(_IDX_HEADER_FMT, header)
            if magic != _IDX_MAGIC:
                raise ValueError(f"Invalid index file: {path}")
            if version != _IDX_VERSION:
                raise ValueError(f"Unsupported index version: {version}")
            offsets = []
            for _ in range(count):
                data = f.read(_IDX_ENTRY_SIZE)
                offsets.append(struct.unpack(_IDX_ENTRY_FMT, data)[0])
        return offsets

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _should_include(index: int, stride: int) -> bool:
        return index % stride == 0


class LocalQueryCsvReader:
    """Reads train/validation/test query files without confusing them for corpus data."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def read(self) -> Iterable[QueryRecord]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield QueryRecord(
                    qid=str(row["qid"]).strip(),
                    question=str(row["question"]).strip(),
                    context=self._parse_list_field(row.get("context")),
                    cids=self._parse_list_field(row.get("cid")),
                )

    def _parse_list_field(self, value: str | None) -> list[str]:
        if value is None:
            return []
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
        return [str(parsed).strip()]


class ChunkedJsonlReader:
    """Reads pre-chunked JSONL files produced by Phase 1 (chunking).

    Each line is a JSON object with fields:
        chunk_id, doc_id, title, source_url, section_path,
        context, text, chunk_index, token_count, etc.
    """

    def __init__(self, chunk_path: str | Path, sample_percent: float = 100.0) -> None:
        self.chunk_path = Path(chunk_path)
        self.sample_percent = sample_percent

    def read(self) -> Iterable[SourceRecord]:
        if self.chunk_path.suffix == ".gz":
            fin_ctx = gzip.open(self.chunk_path, "rt", encoding="utf-8")
        else:
            fin_ctx = open(self.chunk_path, "r", encoding="utf-8")

        with fin_ctx as fin:
            for line_no, line in enumerate(fin, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skipping malformed chunk line {line_no}: {exc}", file=sys.stderr)
                    continue

                chunk_id = payload.get("chunk_id")
                if not chunk_id:
                    continue

                # Apply sampling
                if self.sample_percent < 100.0:
                    import random
                    if random.random() * 100 >= self.sample_percent:
                        continue

                yield SourceRecord(source_id=chunk_id, payload=payload)
