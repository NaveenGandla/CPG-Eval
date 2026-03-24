"""Tests for section builder: heading detection, section splitting, type inference."""

import pytest

from app.services.section_builder import (
    _is_heading,
    build_sections,
    infer_section_type,
)


class TestIsHeading:
    def test_di_role_title(self):
        assert _is_heading("Introduction", "title") is True

    def test_di_role_section_heading(self):
        assert _is_heading("Methods", "sectionHeading") is True

    def test_numbered_heading(self):
        assert _is_heading("1. Introduction", None) is True
        assert _is_heading("1.1 Background", None) is True
        assert _is_heading("2.3.1 Study Design", None) is True

    def test_all_caps_heading(self):
        assert _is_heading("INTRODUCTION", None) is True
        assert _is_heading("TREATMENT GUIDELINES", None) is True

    def test_short_title_line(self):
        assert _is_heading("Background", None) is True

    def test_long_body_text_not_heading(self):
        text = "This is a long paragraph that describes the treatment protocol in detail and should not be detected as a heading."
        assert _is_heading(text, None) is False

    def test_paragraph_role_not_heading(self):
        assert _is_heading("Some paragraph text here.", None) is False


class TestInferSectionType:
    def test_definitions(self):
        assert infer_section_type("Definitions") == "definitions"
        assert infer_section_type("Key Definitions") == "definitions"

    def test_abbreviations(self):
        assert infer_section_type("Abbreviations") == "abbreviations"
        assert infer_section_type("Acronyms and Glossary") == "abbreviations"

    def test_guideline(self):
        assert infer_section_type("Treatment Guidelines") == "guideline"
        assert infer_section_type("Dosing Recommendations") == "guideline"
        assert infer_section_type("Safety Profile") == "guideline"
        assert infer_section_type("Efficacy Outcomes") == "guideline"

    def test_general(self):
        assert infer_section_type("Introduction") == "general"
        assert infer_section_type("References") == "general"


class TestBuildSections:
    def test_basic_section_building(self):
        layout = {
            "paragraphs": [
                {"content": "Introduction", "role": "sectionHeading"},
                {"content": "This guideline covers treatment of NDMM.", "role": None},
                {"content": "Background information on the disease.", "role": None},
                {"content": "Treatment Guidelines", "role": "sectionHeading"},
                {"content": "D-VRd is the recommended first-line therapy.", "role": None},
                {"content": "The GRIFFIN trial showed 99% ORR.", "role": None},
            ],
            "tables": [],
        }

        sections = build_sections(layout)

        assert len(sections) == 2
        assert sections[0].title == "Introduction"
        assert sections[0].section_type == "general"
        assert "NDMM" in sections[0].content
        assert sections[1].title == "Treatment Guidelines"
        assert sections[1].section_type == "guideline"

    def test_numbered_headings(self):
        layout = {
            "paragraphs": [
                {"content": "1. Overview", "role": None},
                {"content": "This section provides an overview.", "role": None},
                {"content": "2. Methods", "role": None},
                {"content": "We used a systematic review approach.", "role": None},
            ],
            "tables": [],
        }

        sections = build_sections(layout)

        assert len(sections) == 2
        assert sections[0].title == "Overview"
        assert sections[1].title == "Methods"

    def test_sections_have_keywords(self):
        layout = {
            "paragraphs": [
                {"content": "Treatment Protocol", "role": "sectionHeading"},
                {
                    "content": (
                        "The treatment protocol involves daratumumab combined with "
                        "bortezomib, lenalidomide, and dexamethasone (D-VRd). "
                        "Patients receive induction therapy followed by autologous "
                        "stem cell transplant. The GRIFFIN trial demonstrated significant "
                        "improvement in progression-free survival."
                    ),
                    "role": None,
                },
            ],
            "tables": [],
        }

        sections = build_sections(layout)
        assert len(sections) == 1
        assert len(sections[0].keywords) > 0

    def test_sections_ordered(self):
        layout = {
            "paragraphs": [
                {"content": "First", "role": "sectionHeading"},
                {"content": "Content one.", "role": None},
                {"content": "Second", "role": "sectionHeading"},
                {"content": "Content two.", "role": None},
                {"content": "Third", "role": "sectionHeading"},
                {"content": "Content three.", "role": None},
            ],
            "tables": [],
        }

        sections = build_sections(layout)
        orders = [s.order for s in sections]
        assert orders == sorted(orders)

    def test_empty_input(self):
        sections = build_sections({"paragraphs": [], "tables": []})
        assert sections == []

    def test_tables_included(self):
        layout = {
            "paragraphs": [
                {"content": "Results", "role": "sectionHeading"},
                {"content": "The key findings are shown below.", "role": None},
            ],
            "tables": ["Drug | Response Rate\nD-VRd | 99%\nVRd | 87%"],
        }

        sections = build_sections(layout)
        assert len(sections) == 1
        assert "D-VRd" in sections[0].content

    def test_fallback_pseudo_sections(self):
        """When no headings are detected, create pseudo-sections."""
        layout = {
            "paragraphs": [
                {"content": f"Paragraph {i} with enough text to be meaningful." * 3, "role": None}
                for i in range(12)
            ],
            "tables": [],
        }

        sections = build_sections(layout)
        # Should create pseudo-sections rather than one giant section
        assert len(sections) > 1

    def test_section_ids_are_unique(self):
        layout = {
            "paragraphs": [
                {"content": "A", "role": "sectionHeading"},
                {"content": "Content A.", "role": None},
                {"content": "B", "role": "sectionHeading"},
                {"content": "Content B.", "role": None},
            ],
            "tables": [],
        }

        sections = build_sections(layout)
        ids = [s.id for s in sections]
        assert len(ids) == len(set(ids))
