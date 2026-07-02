"""Vietnamese text normalization for Wikipedia queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# Common Vietnamese abbreviations → full form
VIETNAMESE_ABBREVIATIONS: dict[str, str] = {
    "tp": "thành phố",
    "tp.": "thành phố",
    "tphcm": "thành phố hồ chí minh",
    "hn": "hà nội",
    "đn": "đà nẵng",
    "vt": "vũng tàu",
    "qn": "quảng ninh",
    "tq": "trung quốc",
    "hq": "hàn quốc",
    "nb": "nhật bản",
    "my": "mỹ",
    "us": "mỹ",
    "uk": "anh",
    "vh": "văn hóa",
    "kt": "kinh tế",
    "ls": "lịch sử",
    "đl": "địa lý",
    "khhđ": "khoa học tự nhiên",
    "xh": "xã hội",
    "tt": "thông tin",
    "gd": "giáo dục",
    "yt": "y tế",
    "qs": "quân sự",
    "tn": "thiên nhiên",
    "mh": "môi trường",
    "cntt": "công nghệ thông tin",
    "khcn": "khoa học công nghệ",
}

# Topic keywords for intent classification
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "definition": ["là gì", "định nghĩa", "có nghĩa", "nghĩa là", "what is", "define"],
    "person": ["ai", "là ai", "sinh", "mất", "năm sinh", "quê", "who"],
    "location": ["ở đâu", "nằm ở", "thuộc", "vị trí", "where", "thủ đô"],
    "time": ["khi nào", "năm nào", "thời gian", "when", "bao giờ"],
    "number": ["bao nhiêu", "dân số", "diện tích", "how many", "how much"],
    "history": ["lịch sử", "thành lập", "ra đời", "history", "founded"],
    "comparison": ["so sánh", "khác nhau", "giống", "khác", "compare", "vs"],
}


@dataclass(slots=True)
class NormalizedResult:
    """Result of query normalization."""

    normalized_text: str
    intent: str = "general"
    filters: dict[str, str] = field(default_factory=dict)
    expansions: list[str] = field(default_factory=list)


class QueryNormalizer:
    """Normalize Vietnamese Wikipedia queries.

    Steps:
    1. Unicode NFC normalization
    2. Lowercase
    3. Expand Vietnamese abbreviations
    4. Normalize whitespace and punctuation
    5. Classify intent from question patterns
    6. Generate synonym expansions
    """

    def normalize(self, query: str) -> NormalizedResult:
        """Normalize a query and extract metadata."""
        text = query.strip()

        # Step 1: Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # Step 2: Lowercase
        text_lower = text.lower()

        # Step 3: Expand abbreviations
        text_expanded = self._expand_abbreviations(text_lower)

        # Step 4: Normalize whitespace
        text_normalized = re.sub(r"\s+", " ", text_expanded).strip()

        # Step 5: Classify intent
        intent = self._classify_intent(text_normalized)

        # Step 6: Generate expansions
        expansions = self._generate_expansions(text_normalized)

        return NormalizedResult(
            normalized_text=text_normalized,
            intent=intent,
            expansions=expansions,
        )

    def _expand_abbreviations(self, text: str) -> str:
        """Expand Vietnamese abbreviations to full form."""
        result = text
        for abbr, full in VIETNAMESE_ABBREVIATIONS.items():
            # Match abbreviation with word boundaries
            pattern = r"\b" + re.escape(abbr) + r"\b"
            result = re.sub(pattern, full, result)
        return result

    def _classify_intent(self, text: str) -> str:
        """Classify query intent from question patterns."""
        for intent, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        return "general"

    def _generate_expansions(self, text: str) -> list[str]:
        """Generate query expansions for better retrieval."""
        expansions: list[str] = []

        # Add "Việt Nam" if query mentions Vietnamese topics without explicit country
        vietnamese_topics = ["thủ đô", "tiểu bang", "tỉnh", "thành phố", "quốc gia"]
        for topic in vietnamese_topics:
            if topic in text and "việt nam" not in text:
                expansions.append(f"{text} việt nam")
                break

        # Add Wikipedia context for definition queries
        if any(kw in text for kw in ["là gì", "định nghĩa", "có nghĩa"]):
            expansions.append(f"{text} wikipedia")

        return expansions[:2]  # Max 2 expansions
