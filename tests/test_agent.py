"""Tests for khonliang_researcher.agent — BaseResearchAgent construction and skills."""

import pytest

from khonliang_researcher.agent import BaseResearchAgent
from khonliang_researcher.domain import DomainConfig
from khonliang_researcher.engines import EngineRegistry


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_base_research_agent_has_agent_type():
    assert BaseResearchAgent.agent_type == "researcher"


def test_base_research_agent_default_domain_is_generic():
    assert BaseResearchAgent.domain.is_generic


def test_subclass_with_domain():
    class TestAgent(BaseResearchAgent):
        agent_type = "test-researcher"
        domain = DomainConfig(
            name="test",
            rules=["rule 1"],
            engines=["google"],
        )

    assert TestAgent.domain.name == "test"
    assert TestAgent.domain.has_rules
    assert TestAgent.agent_type == "test-researcher"


def test_subclass_inherits_module_name():
    assert BaseResearchAgent.module_name == "khonliang_researcher.agent"


# ---------------------------------------------------------------------------
# Skill registration
# ---------------------------------------------------------------------------


def test_register_skills_returns_standard_set():
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.engine_registry = EngineRegistry()
    skills = agent.register_skills()

    skill_names = {s.name for s in skills}

    # Core search skills
    assert "find_relevant" in skill_names
    assert "knowledge_search" in skill_names
    assert "paper_context" in skill_names

    # Ingestion
    assert "fetch_paper" in skill_names
    assert "ingest_file" in skill_names
    assert "ingest_idea" in skill_names

    # Distillation
    assert "start_distillation" in skill_names

    # Ideas
    assert "research_idea" in skill_names
    assert "brief_idea" in skill_names

    # Synthesis
    assert "synthesize_topic" in skill_names
    assert "synthesize_project" in skill_names

    # Scoring
    assert "score_relevance" in skill_names
    assert "concepts_for_project" in skill_names

    # Bundling (NOT FRs)
    assert "synergize_concepts" in skill_names

    # Exploration
    assert "concept_tree" in skill_names
    assert "concept_path" in skill_names

    # Graph
    assert "triple_query" in skill_names

    # Ops
    assert "health_check" in skill_names


def test_no_fr_skills_in_base():
    """BaseResearchAgent must NOT register FR management skills — those belong to developer."""
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.engine_registry = EngineRegistry()
    skills = agent.register_skills()
    skill_names = {s.name for s in skills}

    # These are developer-specific, not generic research
    assert "feature_requests" not in skill_names
    assert "next_fr" not in skill_names
    assert "update_fr_status" not in skill_names
    assert "promote_fr" not in skill_names
    assert "fr_overlaps" not in skill_names
    assert "merge_frs" not in skill_names
    assert "review_feature_requests" not in skill_names
    assert "project_capabilities" not in skill_names
    assert "project_landscape" not in skill_names


def test_no_code_specific_skills_in_base():
    """Code-specific tools belong in developer-researcher, not the generic base."""
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.engine_registry = EngineRegistry()
    skills = agent.register_skills()
    skill_names = {s.name for s in skills}

    assert "ingest_github" not in skill_names
    assert "scan_codebase" not in skill_names
    assert "evaluate_capability" not in skill_names
    assert "register_repo" not in skill_names
    assert "list_repos" not in skill_names


def test_no_pipeline_management_skills_in_base():
    """Pipeline management tools belong in the full researcher, not the generic base."""
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.engine_registry = EngineRegistry()
    skills = agent.register_skills()
    skill_names = {s.name for s in skills}

    assert "browse_feeds" not in skill_names
    assert "reading_list" not in skill_names
    assert "worker_status" not in skill_names
    assert "paper_digest" not in skill_names


def test_skill_count():
    """Baseline skill count for the generic researcher."""
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.engine_registry = EngineRegistry()
    skills = agent.register_skills()
    # Should be around 18-22 generic skills
    assert 15 <= len(skills) <= 25, f"Expected 15-25 skills, got {len(skills)}"


# ---------------------------------------------------------------------------
# Domain config integration
# ---------------------------------------------------------------------------


def test_domain_config_from_yaml_loads_into_agent(tmp_path):
    import yaml

    config = {
        "domain": {
            "name": "test-domain",
            "rules": ["test rule"],
            "engines": ["web_search"],
        },
        "db_path": "data/test.db",
        "models": {},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    cfg = DomainConfig.from_yaml(str(path))
    assert cfg.name == "test-domain"
    assert cfg.rules == ["test rule"]


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_registry_initialized():
    agent = BaseResearchAgent.__new__(BaseResearchAgent)
    agent.__init__(agent_id="test", bus_url="http://test", config_path="")
    assert isinstance(agent.engine_registry, EngineRegistry)


# ---------------------------------------------------------------------------
# Domain rules → synthesizer SYSTEM_PROMPT injection
# ---------------------------------------------------------------------------


def _make_minimal_config(tmp_path, domain_config=None):
    """Build a minimal YAML config file for agent _setup() tests."""
    import yaml

    config = {
        "db_path": str(tmp_path / "test.db"),
        "models": {},
        "projects": {},
    }
    if domain_config:
        config["domain"] = domain_config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


@pytest.mark.asyncio
async def test_domain_rules_injected_into_synthesizer(tmp_path):
    """When DomainConfig has rules, they should be appended to the
    synthesizer's SYSTEM_PROMPT at setup time."""
    from khonliang_researcher.synthesizer import BaseSynthesizer

    config_path = _make_minimal_config(tmp_path, domain_config={
        "name": "genealogy",
        "rules": [
            "Apply the Genealogical Proof Standard",
            "Distinguish primary vs derivative sources",
        ],
    })

    agent = BaseResearchAgent(
        agent_id="test-genealogy",
        bus_url="http://test",
        config_path=config_path,
    )
    await agent._setup()

    base_prompt = BaseSynthesizer.SYSTEM_PROMPT
    assert agent.synthesizer.SYSTEM_PROMPT != base_prompt
    assert "Apply the Genealogical Proof Standard" in agent.synthesizer.SYSTEM_PROMPT
    assert "Distinguish primary vs derivative sources" in agent.synthesizer.SYSTEM_PROMPT
    assert "Domain rules (genealogy)" in agent.synthesizer.SYSTEM_PROMPT
    # The base prompt content should still be there (augmented, not replaced)
    assert base_prompt.rstrip() in agent.synthesizer.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_no_domain_rules_leaves_synthesizer_prompt_unchanged(tmp_path):
    """Generic researcher (no rules) gets the default SYSTEM_PROMPT unchanged."""
    from khonliang_researcher.synthesizer import BaseSynthesizer

    config_path = _make_minimal_config(tmp_path)  # no domain config

    agent = BaseResearchAgent(
        agent_id="test-generic",
        bus_url="http://test",
        config_path=config_path,
    )
    await agent._setup()

    # No rules means no augmentation — instance attr equals class attr
    assert agent.synthesizer.SYSTEM_PROMPT == BaseSynthesizer.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_empty_domain_rules_does_not_inject(tmp_path):
    """A domain with an empty rules list should not augment the prompt."""
    from khonliang_researcher.synthesizer import BaseSynthesizer

    config_path = _make_minimal_config(tmp_path, domain_config={
        "name": "custom",
        "rules": [],
    })

    agent = BaseResearchAgent(
        agent_id="test-empty",
        bus_url="http://test",
        config_path=config_path,
    )
    await agent._setup()

    assert agent.synthesizer.SYSTEM_PROMPT == BaseSynthesizer.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_domain_rules_does_not_mutate_class_attribute(tmp_path):
    """Instance-level override must not leak into BaseSynthesizer class."""
    from khonliang_researcher.synthesizer import BaseSynthesizer

    original_class_prompt = BaseSynthesizer.SYSTEM_PROMPT

    config_path = _make_minimal_config(tmp_path, domain_config={
        "name": "test",
        "rules": ["Rule that should not leak"],
    })

    agent = BaseResearchAgent(
        agent_id="test-isolation",
        bus_url="http://test",
        config_path=config_path,
    )
    await agent._setup()

    # After setup, class attribute must be unchanged
    assert BaseSynthesizer.SYSTEM_PROMPT == original_class_prompt
    # But the instance has an augmented version
    assert "Rule that should not leak" in agent.synthesizer.SYSTEM_PROMPT
