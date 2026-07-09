from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"\S+")


class WikipediaArticleCleaner:
    """Clean Wikipedia markup from article text.

    v2 improvements over v1:
    - Handles nested templates via iterative removal (loop until stable).
    - Removes multi-line infobox value continuations.
    - No new dependencies; still pure stdlib.
    """

    # Patterns applied after template removal
    REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
    HTML_RE = re.compile(r"<[^>]+>", re.DOTALL)
    MAGIC_WORD_RE = re.compile(r"__NOTOC__|__NOEDITSECTION__|__TOC__")

    WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

    TEMPLATE_LINE_RE = re.compile(r"^[|!].*", re.MULTILINE)
    CLOSE_BRACE_RE = re.compile(r"^}+", re.MULTILINE)

    def clean(self, text: str) -> str:
        """Full cleanup pipeline: normalize → templates → references → wikilinks → artifacts → repair."""
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        normalized = self._remove_templates(normalized)
        normalized = self._remove_references_and_html(normalized)
        normalized = self._unwrap_wikilinks(normalized)
        normalized = self._strip_formatting(normalized)
        normalized = self._remove_template_artifacts(normalized)
        normalized = self._repair_broken_lines(normalized)

        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    # ── Template removal ───────────────────────────────────────────

    def _remove_templates(self, text: str) -> str:
        """Remove all {{templates}} including nested ones.

        Repeatedly strips innermost {{...}} pairs until the text stabilises.
        Handles cases like {{Infobox | image = {{Image | url=x}} }}.
        """
        # Matches a {{ }} pair that contains NO inner braces — the innermost level.
        INNERMOST_TEMPLATE = re.compile(r"\{\{[^{}]*?\}\}", re.DOTALL)

        max_iterations = 30
        for _ in range(max_iterations):
            new_text = INNERMOST_TEMPLATE.sub(" ", text)
            if new_text == text:
                return new_text
            text = new_text
        return text

    def _remove_references_and_html(self, text: str) -> str:
        cleaned = self.REF_RE.sub(" ", text)
        cleaned = self.HTML_RE.sub(" ", cleaned)
        cleaned = self.MAGIC_WORD_RE.sub(" ", cleaned)
        return cleaned

    # ── Wikilink unwrapping ─────────────────────────────────────────

    def _unwrap_wikilinks(self, text: str) -> str:
        """Convert [[target|display]] → display, [[target]] → target."""

        def _unwrap(match: re.Match) -> str:
            inner = match.group(1)
            parts = inner.split("|", 1)
            display = parts[-1].strip()
            return display if display else parts[0]

        return self.WIKILINK_RE.sub(_unwrap, text)

    # ── Formatting ──────────────────────────────────────────────────

    def _strip_formatting(self, text: str) -> str:
        return text.replace("'''", "").replace("''", "")

    # ── Post-template artifacts ─────────────────────────────────────

    def _remove_template_artifacts(self, text: str) -> str:
        """Remove leftover | lines, }} lines, and orphaned infobox value continuations."""
        lines = text.split("\n")
        result: list[str] = []
        in_infobox_valley = False

        for line in lines:
            stripped = line.strip()

            # Infobox field definition or table cell
            if stripped.startswith("|") or stripped.startswith("!"):
                in_infobox_valley = True
                continue

            # Closing braces
            if re.match(r"^}+", stripped):
                in_infobox_valley = False
                continue

            # Blank line resets context
            if not stripped:
                in_infobox_valley = False
                result.append("")
                continue

            # Inside infobox value continuation zone — heuristically skip noise
            if in_infobox_valley:
                if self._looks_like_real_content(stripped):
                    in_infobox_valley = False
                    result.append(stripped)
                # else: skip (noise from infobox value continuation)
            else:
                result.append(stripped)

        cleaned = "\n".join(result)
        return re.sub(r"\n[ \t]*\n", "\n\n", cleaned)

    @staticmethod
    def _looks_like_real_content(line: str) -> bool:
        """Heuristic: does this line look like article content vs infobox noise?"""
        # Section heading markers
        if re.match(r"^={2,}", line):
            return True
        # List or enumerated items
        if re.match(r"^[*#\-]|\d+\.", line):
            return True
        # Infobox remnants: parenthetical noise like "(Hà Nội) (Huế)"
        if re.match(r"^[\(\[\{]", line) and "~" in line:
            return False
        if line.count("(") >= 2 and len(_WORD_RE.findall(line)) <= 4:
            return False
        # Starts uppercase (Vietnamese/English) with reasonable length
        if re.match(r"^[A-ZÀ-ỸĐ]", line) and len(line) > 25:
            return True
        # Has sentence-ending punctuation and decent length
        if re.search(r"[.!?]$", line) and len(line) > 15:
            return True
        return False

    # ── Line repair ─────────────────────────────────────────────────

    def _repair_broken_lines(self, text: str) -> str:
        """Join continuation lines that were split mid-sentence."""
        repaired: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not repaired:
                repaired.append(stripped)
                continue
            if not stripped:
                repaired.append("")
                continue
            if self._starts_new_block(stripped):
                repaired.append(stripped)
                continue
            prev = repaired[-1]
            if prev and not prev.endswith((".", ":", ";", "?", "!")):
                repaired[-1] = f"{prev} {stripped}"
            else:
                repaired.append(stripped)
        return "\n".join(repaired)

    @staticmethod
    def _starts_new_block(line: str) -> bool:
        return bool(re.match(r"^([*#\-]|\d+\.|==+)", line))
