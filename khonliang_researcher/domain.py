"""Domain configuration for research agent specialization.

A ``DomainConfig`` declares how a domain differs from generic research:
evaluation rules, search engines, prompt overrides, output types, and
additional knowledge sources. ``BaseResearchAgent`` reads it at startup
to wire the pipeline accordingly.

Domain researchers provide a DomainConfig; the framework handles the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DomainConfig:
    """What makes this researcher instance domain-specific.

    Everything here shapes research behavior without changing code.
    A generic researcher uses ``DomainConfig.generic()`` (empty config).
    A domain researcher fills in the fields that matter for its domain.
    """

    name: str = "generic"

    # Evaluation behavior — injected into LLM prompts for summarization,
    # assessment, and extraction. These are the domain's evidence standards.
    rules: list[str] = field(default_factory=list)

    # Prompt overrides — paths to markdown files (or inline strings) that
    # replace the default prompts for each role. Keys: "summarizer",
    # "assessor", "extractor", "idea_parser". Missing keys use defaults.
    prompts: dict[str, str] = field(default_factory=dict)

    # Search engines to register at startup. Names must match registered
    # engine classes (e.g., "google", "arxiv", "familysearch").
    engines: list[str] = field(default_factory=list)

    # What synergize_concepts produces. The generic output is "concept_bundles".
    # Domains can define their own: "research_leads", "evidence_reports", etc.
    # This shapes the synthesis prompt, not the code.
    output_type: str = "concept_bundles"

    # Additional knowledge sources registered with KnowledgeRegistry at
    # startup. Each dict has fields matching KnowledgeSource.__init__.
    knowledge_sources: list[dict[str, Any]] = field(default_factory=list)

    # Relevance keywords — boost scoring for these terms in this domain.
    relevance_keywords: list[str] = field(default_factory=list)

    @classmethod
    def generic(cls) -> DomainConfig:
        """Default config — no domain specialization."""
        return cls(name="generic")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        """Build from a parsed YAML/dict ``domain:`` section."""
        if not data:
            return cls.generic()
        return cls(
            name=data.get("name", "generic"),
            rules=data.get("rules") or [],
            prompts=data.get("prompts") or {},
            engines=data.get("engines") or [],
            output_type=data.get("output_type", "concept_bundles"),
            knowledge_sources=data.get("knowledge_sources") or [],
            relevance_keywords=data.get("relevance_keywords") or [],
        )

    @classmethod
    def from_yaml(cls, path: str, section: str = "domain") -> DomainConfig:
        """Load the ``domain:`` section from a YAML config file."""
        p = Path(path)
        if not p.exists():
            return cls.generic()
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get(section) or {})

    def load_prompt(self, role: str) -> Optional[str]:
        """Load a prompt override for a role, if configured.

        If the value is a path to a file that exists, reads and returns
        the file content. Otherwise returns the value as an inline string.
        Returns None if no override is configured for this role.
        """
        raw = self.prompts.get(role)
        if raw is None:
            return None
        p = Path(raw)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
        return raw

    @property
    def has_rules(self) -> bool:
        return bool(self.rules)

    @property
    def is_generic(self) -> bool:
        return self.name == "generic" and not self.rules and not self.engines

    def rules_prompt_fragment(self) -> str:
        """Format domain rules for prompt injection."""
        if not self.rules:
            return ""
        lines = [f"Domain rules ({self.name}):"]
        for rule in self.rules:
            lines.append(f"- {rule}")
        return "\n".join(lines)
