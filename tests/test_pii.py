"""Tests for output PII redaction (pii.py)."""

from __future__ import annotations

from rag_pipeline.generation.pii import redact_pii


class TestRedactSecrets:
    def test_openrouter_key_redacted(self) -> None:
        out = redact_pii("leak sk-or-v1-abcdef1234567890abcdef1234567890 here")
        assert "sk-or-v1-" not in out
        assert "***REDACTED***" in out
        assert "leak" in out and "here" in out

    def test_openai_key_redacted(self) -> None:
        out = redact_pii("token sk-1234567890abcdefghijklmnopqrstuv end")
        assert "sk-1234" not in out
        assert "***REDACTED***" in out

    def test_bearer_token_redacted(self) -> None:
        out = redact_pii("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "eyJhbGci" not in out
        assert "***REDACTED***" in out

    def test_short_sk_not_matched(self) -> None:
        # "sk-abcd" is too short to be a real key — left alone (avoid false masks).
        out = redact_pii("the stock ticker sk-abcd rose")
        assert out == "the stock ticker sk-abcd rose"


class TestRedactEmail:
    def test_email_local_part_masked_domain_kept(self) -> None:
        out = redact_pii("mail nguyenvana@gmail.com now")
        assert "nguyenvana" not in out
        assert "@gmail.com" in out
        assert out.startswith("mail ***@gmail.com now")

    def test_hotel_email_domain_kept(self) -> None:
        out = redact_pii("lien he info@vinpearl.com nhe")
        assert "@vinpearl.com" in out
        assert "info" not in out


class TestRedactCard:
    def test_16_digit_card_masked_keep_last4(self) -> None:
        out = redact_pii("Card 4111 1111 1111 1111 ok")
        assert "4111 1111 1111 " not in out
        assert "1111 ok" in out  # last 4 kept
        assert "****" in out

    def test_dashed_card_masked(self) -> None:
        out = redact_pii("Card 4111-1111-1111-1111 ok")
        assert "1111 ok" in out
        assert "4111-1111" not in out


class TestRedactNationalId:
    def test_cccd_12_digit_masked(self) -> None:
        out = redact_pii("CCCD 079123456789 day")
        assert "079123456789" not in out
        assert "************" in out

    def test_cmnd_9_digit_masked(self) -> None:
        out = redact_pii("CMND 123456789 cu")
        assert "123456789" not in out
        assert "*********" in out


class TestRedactPhone:
    def test_personal_mobile_masked(self) -> None:
        out = redact_pii("Call 0987654321 now")
        assert "0987654321" not in out
        assert "09" not in out  # prefix masked too, not just the middle
        assert out.endswith("21 now")  # last 2 kept
        assert "********" in out

    def test_landline_preserved(self) -> None:
        # 02x = Vietnamese landline (hotel reception) — legitimate answer content.
        for landline in ["02363888888", "0241234567", "0287654321"]:
            out = redact_pii(f"Lien he {landline} nhe")
            assert landline in out, f"landline wrongly masked: {landline}"
            assert "*" not in out


class TestPreservesLegitContent:
    def test_citations_preserved(self) -> None:
        out = redact_pii("Vinh Ha Long [1] o Quang Ninh [2]")
        assert "[1]" in out and "[2]" in out
        assert "*" not in out

    def test_plain_travel_answer_unchanged(self) -> None:
        text = "Vinh Ha Long nam o phia dong bac Viet Nam, tinh Quang Ninh."
        assert redact_pii(text) == text

    def test_numbers_in_prose_unchanged(self) -> None:
        # Small numbers, years, counts — not PII.
        text = "Hoi An cach Da Nang 30 km, nen o 2-3 ngay. Nam 2024 co 500 van khach."
        assert redact_pii(text) == text


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert redact_pii("") == ""

    def test_none_returns_empty(self) -> None:
        assert redact_pii(None) == ""

    def test_idempotent(self) -> None:
        text = "Call 0987654321, email a@b.com, key sk-or-v1-abcdef1234567890abcdef1234567890, card 4111 1111 1111 1111"
        once = redact_pii(text)
        twice = redact_pii(once)
        assert once == twice  # no double-masking

    def test_mixed_in_one_answer(self) -> None:
        out = redact_pii("Lien he 0912345678 hoac info@vinpearl.com, key sk-abcdef1234567890abcdefghijkl")
        assert "0912345678" not in out
        assert "info" not in out
        assert "sk-abcdef" not in out
        assert "***REDACTED***" in out
        assert "@vinpearl.com" in out