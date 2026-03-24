"""Tests for keyword extraction utility."""

from app.utils.keyword_extraction import extract_keywords


class TestExtractKeywords:
    def test_extracts_keywords(self):
        text = (
            "The GRIFFIN trial demonstrated that daratumumab plus VRd improved "
            "progression-free survival in transplant-eligible NDMM patients. "
            "The overall response rate was 99%. Daratumumab combined with "
            "bortezomib, lenalidomide, and dexamethasone showed superior efficacy."
        )
        keywords = extract_keywords(text, top_n=5)
        assert len(keywords) > 0
        assert len(keywords) <= 5

    def test_empty_text(self):
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_short_text_fallback(self):
        keywords = extract_keywords("Daratumumab treatment protocol", top_n=5)
        assert len(keywords) > 0

    def test_respects_top_n(self):
        text = (
            "Multiple myeloma is a cancer of plasma cells. Treatment involves "
            "chemotherapy, immunotherapy, and stem cell transplant. Common drugs "
            "include bortezomib, lenalidomide, and daratumumab. Clinical trials "
            "have shown improved survival outcomes with combination therapies."
        )
        keywords = extract_keywords(text, top_n=3)
        assert len(keywords) <= 3

    def test_returns_strings(self):
        text = "Bortezomib and lenalidomide are commonly used. Dexamethasone is added."
        keywords = extract_keywords(text)
        for kw in keywords:
            assert isinstance(kw, str)
