"""Input guardrails for query processing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Prompt injection patterns
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[inst\]",
    r"\[/inst\]",
    r"forget\s+(everything|all)",
    r"new\s+instructions?\s*:",
    r"override\s+",
    r"jailbreak",
    r"dan\s+mode",
]

# Unsafe content patterns
UNSAFE_PATTERNS: list[str] = [
    r"how\s+to\s+(make|build|create)\s+(bomb|weapon|explosive)",
    r"hack(ing)?\s+(into|system|account)",
    r"bypass\s+security",
    r"illegal\s+(drug|activity)",
    r"child\s+(abuse|exploitation)",
]


@dataclass(slots=True)
class GuardrailResult:
    """Result of input guardrail checks."""

    is_safe: bool
    risk_flags: list[str] = field(default_factory=list)


class QueryGuardrails:
    """Check queries for prompt injection and unsafe content.

    Returns risk flags for detected issues. Queries with risk flags
    can still be processed but should be logged/monitored.
    """

    def check(self, query: str) -> GuardrailResult:
        """Check a query for safety issues."""
        risk_flags: list[str] = []

        # Check for prompt injection
        if self._check_injection(query):
            risk_flags.append("prompt_injection")

        # Check for unsafe content
        if self._check_unsafe(query):
            risk_flags.append("unsafe_content")

        # Check for malformed query
        if self._check_malformed(query):
            risk_flags.append("malformed_query")

        return GuardrailResult(
            is_safe=len(risk_flags) == 0,
            risk_flags=risk_flags,
        )

    def _check_injection(self, query: str) -> bool:
        """Check for prompt injection patterns."""
        text = query.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _check_unsafe(self, query: str) -> bool:
        """Check for unsafe content patterns."""
        text = query.lower()
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _check_malformed(self, query: str) -> bool:
        """Check for malformed queries."""
        # Too short
        if len(query.strip()) < 3:
            return True

        # Only special characters
        if not re.search(r"[a-zA-ZÀ-ỹ0-9]", query):
            return True

        # Too long (likely pasted content, not a query)
        if len(query) > 2000:
            return True

        return False
