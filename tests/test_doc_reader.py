"""Tests for khonliang_researcher.doc_reader."""

from pathlib import Path

import pytest

from khonliang_researcher.doc_reader import (
    DEFAULT_REFERENCE_PATTERN,
    DocContent,
    LocalDocReader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# read() — basic
# ---------------------------------------------------------------------------

def test_read_returns_doc_content(tmp_path):
    path = write(tmp_path, "doc.md", "# Heading\n\nBody text.\n")
    reader = LocalDocReader()
    doc = reader.read(path)
    assert isinstance(doc, DocContent)
    assert doc.path == path
    assert "Body text" in doc.text


def test_read_empty_file(tmp_path):
    path = write(tmp_path, "empty.md", "")
    doc = LocalDocReader().read(path)
    assert doc.text == ""
    assert doc.frontmatter == {}
    assert doc.sections == {}
    assert doc.references == []


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter_yaml(tmp_path):
    content = (
        "---\n"
        "title: Test Doc\n"
        "fr_id: fr_researcher_abc\n"
        "tags:\n"
        "  - spec\n"
        "  - draft\n"
        "---\n"
        "# Body\n"
    )
    path = write(tmp_path, "fm.md", content)
    doc = LocalDocReader().read(path)
    assert doc.frontmatter["title"] == "Test Doc"
    assert doc.frontmatter["fr_id"] == "fr_researcher_abc"
    assert doc.frontmatter["tags"] == ["spec", "draft"]


def test_parse_frontmatter_helper(tmp_path):
    path = write(tmp_path, "fm.md", "---\nname: x\n---\nBody")
    fm = LocalDocReader().parse_frontmatter(path)
    assert fm == {"name": "x"}


def test_no_frontmatter_returns_empty_dict(tmp_path):
    path = write(tmp_path, "plain.md", "# Just a heading")
    doc = LocalDocReader().read(path)
    assert doc.frontmatter == {}


def test_invalid_frontmatter_returns_empty_dict(tmp_path):
    path = write(tmp_path, "bad.md", "---\nthis is: : invalid: yaml: [\n---\nbody")
    doc = LocalDocReader().read(path)
    assert doc.frontmatter == {}


def test_frontmatter_stripped_from_body(tmp_path):
    path = write(
        tmp_path,
        "fm.md",
        "---\ntitle: x\n---\n# Real Heading\n\nbody",
    )
    doc = LocalDocReader().read(path)
    # Frontmatter should not appear as a section
    assert "title: x" not in str(doc.sections)
    assert "Real Heading" in doc.sections


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_index_sections_by_heading(tmp_path):
    content = (
        "# First\nfirst body\n\n"
        "# Second\nsecond body\n\n"
        "## Subsection\nsub body\n"
    )
    path = write(tmp_path, "sec.md", content)
    doc = LocalDocReader().read(path)
    assert "First" in doc.sections
    assert "Second" in doc.sections
    assert "Subsection" in doc.sections
    assert "first body" in doc.sections["First"]


def test_extract_section_by_heading(tmp_path):
    path = write(
        tmp_path,
        "doc.md",
        "# Acceptance Criteria\nDo X\nDo Y\n\n# Other\nignore me\n",
    )
    section = LocalDocReader().extract_section(path, "Acceptance Criteria")
    assert "Do X" in section
    assert "Do Y" in section
    assert "ignore me" not in section


def test_extract_section_case_insensitive(tmp_path):
    path = write(tmp_path, "doc.md", "# Why\nbecause\n")
    section = LocalDocReader().extract_section(path, "why")
    assert "because" in section


def test_extract_section_strips_hash_markers(tmp_path):
    path = write(tmp_path, "doc.md", "## Design\nspec details\n")
    section = LocalDocReader().extract_section(path, "## Design")
    assert "spec details" in section


def test_extract_missing_section_returns_empty(tmp_path):
    path = write(tmp_path, "doc.md", "# Only\nbody\n")
    section = LocalDocReader().extract_section(path, "Missing")
    assert section == ""


def test_duplicate_heading_keys_disambiguated(tmp_path):
    content = "# Notes\nfirst\n\n# Notes\nsecond\n"
    path = write(tmp_path, "dup.md", content)
    doc = LocalDocReader().read(path)
    assert "Notes" in doc.sections
    assert "Notes#2" in doc.sections
    assert "first" in doc.sections["Notes"]
    assert "second" in doc.sections["Notes#2"]


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def test_default_reference_pattern_matches_fr_ids():
    pattern = DEFAULT_REFERENCE_PATTERN
    import re

    matches = re.findall(pattern, "See fr_researcher_c6b7dca8 and fr_khonliang_03f461fa.")
    assert "fr_researcher_c6b7dca8" in matches
    assert "fr_khonliang_03f461fa" in matches


def test_find_references_in_doc(tmp_path):
    path = write(
        tmp_path,
        "doc.md",
        "Implements fr_researcher_abc.\nDepends on fr_khonliang_xyz.\n",
    )
    refs = LocalDocReader().find_references(path)
    assert "fr_researcher_abc" in refs
    assert "fr_khonliang_xyz" in refs


def test_find_references_unique_in_order(tmp_path):
    path = write(
        tmp_path,
        "doc.md",
        "fr_a_1 fr_b_2 fr_a_1 fr_c_3",
    )
    refs = LocalDocReader().find_references(path)
    assert refs == ["fr_a_1", "fr_b_2", "fr_c_3"]


def test_custom_reference_pattern(tmp_path):
    path = write(tmp_path, "doc.md", "MILESTONE-12 and MILESTONE-99")
    refs = LocalDocReader().find_references(path, pattern=r"MILESTONE-\d+")
    assert refs == ["MILESTONE-12", "MILESTONE-99"]


def test_references_extracted_during_read(tmp_path):
    path = write(tmp_path, "doc.md", "Tracked under fr_developer_28a11ce2.")
    doc = LocalDocReader().read(path)
    assert "fr_developer_28a11ce2" in doc.references


# ---------------------------------------------------------------------------
# glob_docs
# ---------------------------------------------------------------------------

def test_glob_docs_finds_markdown(tmp_path):
    write(tmp_path, "a.md", "a")
    write(tmp_path, "b.md", "b")
    write(tmp_path, "ignored.txt", "x")
    docs = LocalDocReader().glob_docs(str(tmp_path))
    assert len(docs) == 2
    assert all(d.endswith(".md") for d in docs)


def test_glob_docs_recursive(tmp_path):
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "deep.md").write_text("x")
    write(tmp_path, "top.md", "y")
    docs = LocalDocReader().glob_docs(str(tmp_path))
    assert len(docs) == 2


def test_glob_docs_custom_pattern(tmp_path):
    write(tmp_path, "spec.yaml", "x")
    write(tmp_path, "other.md", "x")
    docs = LocalDocReader().glob_docs(str(tmp_path), pattern="*.yaml")
    assert len(docs) == 1
    assert docs[0].endswith("spec.yaml")
