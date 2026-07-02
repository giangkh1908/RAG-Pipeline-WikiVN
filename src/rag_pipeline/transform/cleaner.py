from __future__ import annotations

import re
import unicodedata


class WikipediaArticleCleaner:
    NOISE_PATTERNS = (
        re.compile(r"\{\{[^{}]*\}\}", re.DOTALL),
        re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<[^>]+>", re.DOTALL),
        re.compile(r"__NOTOC__|__NOEDITSECTION__|__TOC__"),
    )

    WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

    TEMPLATE_LINE_RE = re.compile(r"^[|!].*", re.MULTILINE)
    CLOSE_BRACE_RE = re.compile(r"^}+", re.MULTILINE)

    def clean(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._remove_markup_residue(normalized)
        normalized = self._unwrap_wikilinks(normalized)
        normalized = self._strip_formatting(normalized)
        normalized = self._remove_template_artifacts(normalized)
        normalized = self._repair_broken_lines(normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _remove_markup_residue(self, text: str) -> str:
        cleaned = text
        for pattern in self.NOISE_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        return cleaned

    def _unwrap_wikilinks(self, text: str) -> str:
        """Convert [[target|display]] → display,  [[target]] → target."""

        def _unwrap(match: re.Match) -> str:
            inner = match.group(1)
            parts = inner.split("|", 1)
            display = parts[-1].strip()
            return display if display else parts[0]

        return self.WIKILINK_RE.sub(_unwrap, text)

    def _strip_formatting(self, text: str) -> str:
        return text.replace("'''", "").replace("''", "")

    def _remove_template_artifacts(self, text: str) -> str:
        cleaned = self.TEMPLATE_LINE_RE.sub("", text)
        cleaned = self.CLOSE_BRACE_RE.sub("", cleaned)
        cleaned = re.sub(r"\n[ \t]*\n", "\n\n", cleaned)
        return cleaned

    def _repair_broken_lines(self, text: str) -> str:
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

    def _starts_new_block(self, line: str) -> bool:
        return bool(re.match(r"^([*#-]|\d+\.|==+)", line))
