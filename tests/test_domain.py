"""Tests for khonliang_researcher.domain."""

import pytest
import yaml

from khonliang_researcher.domain import DomainConfig


def test_generic_defaults():
    cfg = DomainConfig.generic()
    assert cfg.name == "generic"
    assert cfg.rules == []
    assert cfg.engines == []
    assert cfg.prompts == {}
    assert cfg.output_type == "concept_bundles"
    assert cfg.is_generic is True


def test_from_dict():
    data = {
        "name": "genealogy",
        "rules": ["Apply GPS", "Verify sources"],
        "engines": ["google", "familysearch"],
        "prompts": {"summarizer": "prompts/genealogy.md"},
        "output_type": "research_leads",
        "relevance_keywords": ["census", "vital records"],
    }
    cfg = DomainConfig.from_dict(data)
    assert cfg.name == "genealogy"
    assert len(cfg.rules) == 2
    assert "google" in cfg.engines
    assert cfg.output_type == "research_leads"
    assert cfg.is_generic is False
    assert cfg.has_rules is True


def test_from_dict_empty():
    cfg = DomainConfig.from_dict({})
    assert cfg.is_generic is True


def test_from_dict_none():
    cfg = DomainConfig.from_dict(None)
    assert cfg.is_generic is True


def test_from_yaml(tmp_path):
    config = {
        "domain": {
            "name": "finance",
            "rules": ["Verify against SEC filings"],
            "engines": ["google"],
        },
        "db_path": "data/test.db",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    cfg = DomainConfig.from_yaml(str(path))
    assert cfg.name == "finance"
    assert len(cfg.rules) == 1


def test_from_yaml_missing_file():
    cfg = DomainConfig.from_yaml("/nonexistent/config.yaml")
    assert cfg.is_generic is True


def test_from_yaml_no_domain_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"db_path": "data/test.db"}))
    cfg = DomainConfig.from_yaml(str(path))
    assert cfg.is_generic is True


def test_rules_prompt_fragment():
    cfg = DomainConfig(
        name="genealogy",
        rules=["Apply GPS", "Distinguish primary sources"],
    )
    fragment = cfg.rules_prompt_fragment()
    assert "Domain rules (genealogy)" in fragment
    assert "Apply GPS" in fragment
    assert "Distinguish primary sources" in fragment


def test_rules_prompt_fragment_empty():
    cfg = DomainConfig.generic()
    assert cfg.rules_prompt_fragment() == ""


def test_load_prompt_inline():
    cfg = DomainConfig(prompts={"summarizer": "Summarize concisely."})
    assert cfg.load_prompt("summarizer") == "Summarize concisely."
    assert cfg.load_prompt("assessor") is None


def test_load_prompt_from_file(tmp_path):
    prompt_file = tmp_path / "custom.md"
    prompt_file.write_text("Custom prompt content")
    cfg = DomainConfig(prompts={"summarizer": str(prompt_file)})
    assert cfg.load_prompt("summarizer") == "Custom prompt content"
