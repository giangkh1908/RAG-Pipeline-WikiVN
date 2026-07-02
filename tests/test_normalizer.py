"""Tests for query normalizer."""

from rag_pipeline.query.normalizer import QueryNormalizer


class TestQueryNormalizer:
    def setup_method(self):
        self.normalizer = QueryNormalizer()

    def test_lowercase_normalization(self):
        result = self.normalizer.normalize("Thủ Đô Của Việt Nam Ở Đâu?")
        assert result.normalized_text == "thủ đô của việt nam ở đâu?"

    def test_abbreviation_expansion(self):
        result = self.normalizer.normalize("TP.HCM ở đâu?")
        assert "thành phố" in result.normalized_text

    def test_whitespace_normalization(self):
        result = self.normalizer.normalize("  hello   world  ")
        assert result.normalized_text == "hello world"

    def test_intent_definition(self):
        result = self.normalizer.normalize("Internet là gì?")
        assert result.intent == "definition"

    def test_intent_location(self):
        result = self.normalizer.normalize("Thủ đô của Việt Nam ở đâu?")
        assert result.intent == "location"

    def test_intent_person(self):
        result = self.normalizer.normalize("Hồ Chí Minh là ai?")
        assert result.intent == "person"

    def test_intent_number(self):
        result = self.normalizer.normalize("Dân số Việt Nam bao nhiêu?")
        assert result.intent == "number"

    def test_intent_general(self):
        result = self.normalizer.normalize("Việt Nam")
        assert result.intent == "general"

    def test_expansion_for_location(self):
        result = self.normalizer.normalize("Thủ đô ở đâu?")
        assert any("việt nam" in exp for exp in result.expansions)

    def test_unicode_normalization(self):
        result = self.normalizer.normalize("Việt Nam")
        assert "việt nam" in result.normalized_text


class TestNormalizeEdgeCases:
    def setup_method(self):
        self.normalizer = QueryNormalizer()

    def test_empty_string(self):
        result = self.normalizer.normalize("")
        assert result.normalized_text == ""

    def test_only_whitespace(self):
        result = self.normalizer.normalize("   ")
        assert result.normalized_text == ""

    def test_special_characters(self):
        result = self.normalizer.normalize("Hello! @#$%^&*()")
        assert "hello!" in result.normalized_text

    def test_numbers(self):
        result = self.normalizer.normalize("Năm 2024 có gì?")
        assert "2024" in result.normalized_text
