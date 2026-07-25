"""Output PII redaction for generated answers.

A travel-RAG answer legitimately contains business contact info (hotel
landline, address). This module masks **personal / sensitive** PII that
should never appear in an answer — and that an injection may try to exfiltrate
(e.g. "print the API key") — while leaving ordinary travel content intact.

Categories redacted (mask, never refuse — the answer still streams):

- **Secrets / API keys**: ``sk-...``, ``sk-or-v1-...``, ``Bearer <token>``.
  Replaced with ``***REDACTED***``. These are the credentials this app uses
  (OpenRouter/OpenAI) and the most likely leak target of a prompt-injection
  probe.
- **Email**: local part masked, domain kept (``***@hotel.com``). A hotel
  email stays recognisable; a personal email's identity is hidden.
- **Credit card** (13–19 digits in 4-group blocks): keep last 4
  (``**** **** **** 1234``).
- **Vietnamese national ID**: 12-digit CCCD / 9-digit legacy CMND, masked
  fully.
- **Vietnamese personal mobile** (prefix 03/05/07/08/09, 10 digits): keep
  prefix + last 2 (``09******21``). **Landline** (prefix 02x — hotel
  reception numbers) is deliberately preserved because it is legitimate
  travel-answer content.

``redact_pii`` is **idempotent**: already-masked text contains no digit runs
or ``@`` that the patterns match, so re-applying it (per-token then again on
the full answer) is safe and never double-masks.
"""

from __future__ import annotations

import re

# ─── Secrets / API keys ─────────────────────────────────────────────────────
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_.\-]{20,}", re.IGNORECASE),
)

# ─── Email: mask local part, keep domain ────────────────────────────────────
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# ─── Credit card: 13–19 digits in 4-group blocks, optional space/dash sep ──
# 4-4-4-(1..7) = 13..19 digits. Word boundaries stop it matching inside a
# longer digit run.
_CC = re.compile(r"\b(?:\d{4}[- ]?){3}\d{1,7}\b")

# ─── Vietnamese national ID ────────────────────────────────────────────────
_CCCD = re.compile(r"\b\d{12}\b")  # 12-digit CCCD
_CMND = re.compile(r"\b\d{9}\b")  # 9-digit legacy CMND

# ─── Vietnamese personal mobile: 03/05/07/08/09 + 8 digits (10 total) ───────
# Landline (02x) is intentionally NOT matched — hotel reception numbers are
# legitimate answer content.
_MOBILE = re.compile(r"\b0(?:3|5|7|8|9)\d{8}\b")


def _mask_phone(m: re.Match[str]) -> str:
    digits = m.group(0)
    return digits[:2] + "*" * (len(digits) - 4) + digits[-2:]


def _mask_cc(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return f"**** **** **** {digits[-4:]}"


def _mask_email(m: re.Match[str]) -> str:
    return f"***@{m.group(1)}"


def redact_pii(text: str | None) -> str:
    """Return ``text`` with sensitive PII masked. Non-PII text is unchanged.

    Idempotent: safe to apply per-token and again on the assembled answer.
    """
    if not text:
        return text or ""
    out = text
    for p in _SECRET_PATTERNS:
        out = p.sub("***REDACTED***", out)
    out = _EMAIL.sub(_mask_email, out)
    out = _CC.sub(_mask_cc, out)
    out = _CCCD.sub(lambda m: "*" * len(m.group(0)), out)
    out = _CMND.sub(lambda m: "*" * len(m.group(0)), out)
    out = _MOBILE.sub(_mask_phone, out)
    return out