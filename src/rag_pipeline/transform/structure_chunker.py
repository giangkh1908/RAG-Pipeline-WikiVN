"""Structure-aware chunker with Anthropic-style contextual retrieval.

Parses cleaned Wikipedia text into structural blocks (heading, paragraph, list),
then chunks respecting heading boundaries and prepends natural-language context
so embedding models understand what each chunk is about.

Chunk text format:  "{context}\n\n{raw_content}"
- context: "This chunk is from the 'Title' document, which describes: ..."
- raw_content: the actual passage text
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from rag_pipeline.config import ChunkingConfig
from rag_pipeline.models import CanonicalDocument, DocumentChunk
from rag_pipeline.utils.hashing import stable_hash

_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;…])\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?;…])\s+")

# ── Reference section patterns ──────────────────────────────────────


def _strip_vietnamese_diacritics(text: str) -> str:
    """Remove Vietnamese tone marks so "Tham khảo" → "Tham khao"."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


_REFERENCE_HEADINGS: set[str] = {
    "tham khảo", "liên kết ngoài", "xem thêm", "chú thích",
    "tài liệu tham khảo", "nguồn tham khảo", "đọc thêm",
    "tham khảo thêm", "ghi chú", "chú giải", "trích dẫn",
    "references", "see also", "external links", "notes",
    "footnotes", "further reading", "bibliography",
}

# Pre-computed diacritic-stripped form of _REFERENCE_HEADINGS, used so that
# content which has lost its Vietnamese tone marks still matches.
_REFERENCE_HEADINGS_NO_ACCENTS: set[str] = {
    _strip_vietnamese_diacritics(h) for h in _REFERENCE_HEADINGS
}

# ── Major section headings (level-1) ────────────────────────────────

_MAJOR_SECTIONS: set[str] = {
    "lịch sử", "địa lý", "vị trí địa lý", "địa hình", "khí hậu",
    "thân thế", "tiểu sử", "sự nghiệp", "cuộc đời",
    "kinh tế", "văn hóa", "giáo dục", "chính trị", "hành chính",
    "dân số", "dân cư", "tôn giáo", "ngôn ngữ",
    "đặc điểm", "phân loại", "mô tả", "tổng quan",
    "kiến trúc", "cấu trúc", "chức năng", "nhiệm vụ",
    "giải thưởng", "vinh danh", "di sản",
    "tên gọi", "nguồn gốc", "ý nghĩa",
}

# ── Sub-section heading patterns (level-2) ──────────────────────────

_SUB_SECTION_RE = re.compile(
    r"(dòng\s+dõi|gia\s+đình|thời\s+thơ\s+ấu|thời\s+kỳ|"
    r"giai\s+đoạn|thế\s+kỷ|năm\s+\d{4}|từ\s+năm|"
    r"các\s+(loài|loại|dạng|vùng|khu|huyện|quận|tỉnh|nước|quốc gia)|"
    r"một\s+số|danh\s+sách|các\s+đời|đời\s+sống|"
    r"trong\s+văn\s+hóa|trong\s+nghệ\s+thuật|trong\s+điện\s+ảnh)"
)


@dataclass
class Block:
    """A structural block within an article."""

    kind: Literal["heading", "paragraph", "list"]
    text: str
    level: int = 0
    section_path: list[str] = field(default_factory=list)


class StructuredChunker:
    """Chunk documents by section structure with contextual prefixes.

    Differences from v1 RecursiveChunker:
    - Detects headings (without requiring == markers) and uses them as hard boundaries
    - Prepends Anthropic-style natural-language context for embedding
    - Keeps list items together in a single chunk when possible
    - Marks reference sections (Tham khảo, Liên kết ngoài) as low-priority
    - Stores prev/next chunk links for context expansion at retrieval time
    - Respects ``max_tokens_per_chunk`` for the *full* chunk text (context +
      body) and merges tail chunks smaller than ``min_chunk_tokens``
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    # ── Public API ──────────────────────────────────────────────────

    def chunk(self, document: CanonicalDocument) -> list[DocumentChunk]:
        """Parse, group, and wrap blocks into DocumentChunk list.

        Each chunk's text is: "{context}\\n\\n{raw_content}"
        where context is Anthropic-style natural language describing the
        document and section the chunk belongs to.
        """
        blocks = self._parse_blocks(document)
        if not blocks:
            return []

        doc_summary = self._extract_doc_summary(blocks)
        groups = self._group_into_chunks(blocks, document.title, doc_summary)

        chunks: list[DocumentChunk] = []
        for idx, (raw_text, section_path, is_ref) in enumerate(groups):
            context = self._build_context(document.title, doc_summary, section_path)
            full_text = f"{context}\n\n{raw_text}"

            chunks.append(DocumentChunk(
                chunk_id=stable_hash({"doc_id": document.doc_id, "chunk_index": idx, "text": full_text})[:24],
                doc_id=document.doc_id,
                text=full_text,
                section_path=section_path,
                article_number=document.article_number,
                clause_number=document.clause_number,
                chunk_index=idx,
                token_count=self._count_tokens(full_text),
                parent_chunk_id=None,
                prev_chunk_id=None,
                next_chunk_id=None,
                checksum=stable_hash({"doc_id": document.doc_id, "text": full_text}),
                metadata={
                    "title": document.title,
                    "source_url": document.source_url,
                    "is_reference_section": is_ref,
                    **document.metadata,
                },
            ))

        self._link_neighbors(chunks)
        return chunks

    @staticmethod
    def split_context_and_text(chunk_text: str) -> tuple[str, str]:
        """Split a chunk's full text into (context, raw_text)."""
        if "\n\n" in chunk_text:
            context, text = chunk_text.split("\n\n", 1)
            return context, text
        return chunk_text, ""

    # ── Chunk grouping ──────────────────────────────────────────────

    def _group_into_chunks(
        self,
        blocks: list[Block],
        title: str,
        doc_summary: str,
    ) -> list[tuple[str, list[str], bool]]:
        """Group blocks while enforcing the full context + content budget."""
        groups: list[tuple[str, list[str], bool]] = []
        current_texts: list[str] = []
        current_kinds: list[str] = []
        current_path: list[str] = [blocks[0].section_path[0]] if blocks else []
        current_tokens = 0
        in_reference = False

        def _flush_current() -> None:
            nonlocal current_tokens
            if current_texts:
                # Use newline as separator whenever any list block is in the
                # group so list items stay on their own lines.
                sep = "\n" if "list" in current_kinds else " "
                groups.append((sep.join(current_texts), list(current_path), in_reference))
                current_texts.clear()
                current_kinds.clear()
                current_tokens = 0

        for block in blocks:
            if block.kind == "heading":
                _flush_current()
                current_path = list(block.section_path)
                in_reference = self._is_reference_heading(block.text)
                # Keep headings searchable and avoid losing content when a
                # short standalone line is misclassified as a heading.
                current_texts = [block.text]
                current_kinds = [block.kind]
                current_tokens = self._count_tokens(block.text)
                continue

            if self._is_reference_heading(block.text):
                in_reference = True

            budget = self._content_budget(title, doc_summary, current_path)
            overlap = self._effective_overlap(budget)
            piece_budget = max(1, budget - overlap)

            # Split oversized paragraphs/lists before trying to fit them.
            pieces = self._split_text(block.text, piece_budget)
            for piece in pieces:
                piece_tokens = self._count_tokens(piece)
                # Defensive hard split: a single piece should never exceed the
                # budget after _split_text, but guard against pathological cases.
                if piece_tokens > budget:
                    words = _WORD_RE.findall(piece)
                    for start in range(0, len(words), budget):
                        subpiece = " ".join(words[start:start + budget])
                        sub_tokens = self._count_tokens(subpiece)
                        if current_tokens + sub_tokens <= budget:
                            current_texts.append(subpiece)
                            current_kinds.append(block.kind)
                            current_tokens += sub_tokens
                        else:
                            _flush_current()
                            current_texts.append(subpiece)
                            current_kinds.append(block.kind)
                            current_tokens = sub_tokens
                    continue

                if current_tokens + piece_tokens <= budget:
                    current_texts.append(piece)
                    current_kinds.append(block.kind)
                    current_tokens += piece_tokens
                    continue

                previous_text = " ".join(current_texts)
                _flush_current()
                overlap_text = self._overlap_tail(
                    previous_text,
                    min(overlap, budget - piece_tokens),
                )
                if overlap_text:
                    current_texts.append(overlap_text)
                    current_kinds.append("paragraph")
                    current_tokens += self._count_tokens(overlap_text)
                current_texts.append(piece)
                current_kinds.append(block.kind)
                current_tokens += piece_tokens

        _flush_current()
        return self._merge_small_groups(groups, title, doc_summary)

    def _content_budget(self, title: str, doc_summary: str, path: list[str]) -> int:
        """Return the raw-content budget that keeps the full chunk under limit.

        The full chunk is ``context + \\n\\n + raw_content``.  We subtract the
        actual context length plus a configurable safety margin so that real
        embedding tokenizers (which usually count more than whitespace words)
        do not push the input over ``max_tokens_per_chunk``.
        """
        context_tokens = self._count_tokens(self._build_context(title, doc_summary, path))
        margin = getattr(self.config, "estimated_context_tokens", 0)
        budget = self.config.max_tokens_per_chunk - context_tokens - margin
        # Never drop below 1 token of raw content; the real problem (context
        # larger than the limit) is caught by the caller through tiny chunks.
        return max(1, budget)

    def _effective_overlap(self, budget: int) -> int:
        return max(0, min(self.config.chunk_overlap_tokens, budget // 3))

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        """Split oversized text on sentences, then fall back to word windows."""
        if self._count_tokens(text) <= max_tokens:
            return [text]

        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]

        for sentence in sentences:
            words = _WORD_RE.findall(sentence)
            if len(words) > max_tokens:
                if current:
                    pieces.append(" ".join(current))
                    current = []
                    current_tokens = 0
                pieces.extend(
                    " ".join(words[start:start + max_tokens])
                    for start in range(0, len(words), max_tokens)
                )
            elif current_tokens + len(words) <= max_tokens:
                current.append(sentence)
                current_tokens += len(words)
            else:
                pieces.append(" ".join(current))
                current = [sentence]
                current_tokens = len(words)

        if current:
            pieces.append(" ".join(current))
        return pieces

    @staticmethod
    def _overlap_tail(text: str, overlap_tokens: int) -> str:
        if overlap_tokens <= 0 or not text:
            return ""
        return " ".join(_WORD_RE.findall(text)[-overlap_tokens:])

    def _merge_small_groups(
        self,
        groups: list[tuple[str, list[str], bool]],
        title: str,
        doc_summary: str,
    ) -> list[tuple[str, list[str], bool]]:
        """Merge tiny tails when doing so does not cross a section or budget."""
        merged: list[tuple[str, list[str], bool]] = []
        for text, path, is_ref in groups:
            if (
                merged
                and self._count_tokens(text) < self.config.min_chunk_tokens
                and merged[-1][1] == path
                and merged[-1][2] == is_ref
            ):
                previous_text, _, _ = merged[-1]
                combined = f"{previous_text} {text}"
                if self._count_tokens(combined) <= self._content_budget(title, doc_summary, path):
                    merged[-1] = (combined, path, is_ref)
                    continue
            merged.append((text, path, is_ref))
        return merged

    # ── Block parsing ───────────────────────────────────────────────

    def _parse_blocks(self, document: CanonicalDocument) -> list[Block]:
        """Parse cleaned text into heading / paragraph / list blocks."""
        text = document.content.strip()
        if not text:
            return []

        paragraphs = re.split(r"\n\n+", text)
        blocks: list[Block] = []
        section_path: list[str] = [document.title]

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            lines = para.split("\n")

            # Single short line → could be a heading
            if len(lines) == 1 and self._looks_like_heading(lines[0]):
                heading_text = lines[0]
                level = self._estimate_heading_level(heading_text)
                section_path = section_path[:level] + [heading_text]
                blocks.append(Block(
                    kind="heading", text=heading_text,
                    level=level, section_path=list(section_path),
                ))
                continue

            # Multiple lines all starting with list markers → list block
            if all(self._is_list_line(ln) for ln in lines if ln.strip()):
                blocks.append(Block(
                    kind="list", text=para,
                    section_path=list(section_path),
                ))
                continue

            # Default: paragraph block
            blocks.append(Block(
                kind="paragraph", text=para,
                section_path=list(section_path),
            ))

        return blocks

    # ── Heading detection ───────────────────────────────────────────

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        """Heuristic: is this a Wikipedia section heading?

        True for lines like: Lịch sử, Địa lý, Thân thế, Vị trí địa lý
        False for: sentences, list items, URLs, infobox remnants.
        """
        stripped = line.strip()
        if not stripped:
            return False

        words = _WORD_RE.findall(stripped)
        if len(words) > 10 or len(stripped) > 100:
            return False
        if re.match(r"^[*#\-]|\d+[.)]\s", stripped):
            return False  # list item
        if re.search(r"[.!?;:]$", stripped):
            return False  # sentence ending
        if re.match(r"https?://", stripped):
            return False  # URL
        if not re.match(r"^[A-ZÀ-ỸĐ0-9\"'“‘\(\[]", stripped):
            return False  # starts lowercase
        if re.match(r"^[\(\[\{].*[\)\]\}]", stripped) and "~" in stripped:
            return False  # infobox remnant: "(HN) (Huế) ~ (HCM)"
        if stripped.count("(") >= 2 and len(words) <= 4:
            return False  # multiple parenthetical groups

        return True

    @staticmethod
    def _estimate_heading_level(heading_text: str) -> int:
        """Estimate heading nesting level.

        1 = major section (Lịch sử, Địa lý, Thân thế)
        2 = sub-section (Dòng dõi, Thời kỳ, Các loài)
        """
        text = heading_text.lower().strip()
        if text in _MAJOR_SECTIONS:
            return 1
        if _SUB_SECTION_RE.search(text):
            return 2
        return 1

    # ── List & reference detection ──────────────────────────────────

    @staticmethod
    def _is_list_line(line: str) -> bool:
        return bool(re.match(r"^[*#\-]|\d+[.)]\s", line.strip()))

    @staticmethod
    def _is_reference_heading(text: str) -> bool:
        """Return True if ``text`` is a known reference-section heading.

        Matching is diacritic-insensitive so that content which has lost its
        Vietnamese tone marks (e.g. ``"Tham khao"``) is still recognised as
        ``"Tham khảo"``.
        """
        return _strip_vietnamese_diacritics(text.strip().lower()) in _REFERENCE_HEADINGS_NO_ACCENTS

    # ── Context building (Anthropic-style) ──────────────────────────

    @staticmethod
    def _extract_doc_summary(blocks: list[Block]) -> str:
        """Extract the first meaningful sentence as document-level context.

        Skips blocks that look like infobox remnants.
        """
        for block in blocks:
            if block.kind != "paragraph" or not block.text:
                continue
            # Skip infobox remnant lines like "(Hà Nội) (Huế) ~ (TP. HCM)"
            if re.match(r"^[\(\[\{]", block.text) and "~" in block.text:
                continue

            m = re.match(r"^(.+?[.!?])(?:\s|$)", block.text)
            if m and len(m.group(1)) > 10:
                return m.group(1)
            return block.text[:100].rsplit(" ", 1)[0]

        return ""

    @staticmethod
    def _build_context(title: str, doc_summary: str, section_path: list[str]) -> str:
        """Build Anthropic-style natural-language context string.

        No bracket markup — embedding models understand natural language better.

        Examples:
          "This chunk is from the 'Việt Nam' document."
          "This chunk is from the 'Việt Nam' document, which describes: ..."
          "This chunk is from the 'Việt Nam' document, ..., specifically the 'Lịch sử' section."
        """
        ctx = f"This chunk is from the '{title}' document"
        if doc_summary:
            ctx += f", which describes: {doc_summary[:120]}"
        if len(section_path) > 1:
            sec = " > ".join(section_path[1:])
            ctx += f", specifically the '{sec}' section"
        return ctx + "."

    # ── Token counting & neighbor linking ───────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(_WORD_RE.findall(text))

    @staticmethod
    def _link_neighbors(chunks: list[DocumentChunk]) -> None:
        for i, c in enumerate(chunks):
            c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
