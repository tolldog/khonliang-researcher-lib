"""Multi-document synthesis using LLM generation.

Subclass BaseSynthesizer and override prompt class attributes and
_get_summaries() for domain-specific synthesis.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.knowledge.store import KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore
from khonliang.pool import ModelPool

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    query: str
    synthesis_type: str  # topic, target, landscape
    content: str
    document_count: int
    document_ids: List[str] = field(default_factory=list)
    success: bool = False


class BaseSynthesizer:
    """Generate combined summaries across multiple documents.

    Override class-level prompts and _get_summaries() for domain-specific
    synthesis (research papers, news articles, etc.).
    """

    SYSTEM_PROMPT = """\
You are a synthesis assistant. Given summaries of multiple documents,
produce a combined analysis. Be specific — cite document titles and findings.
Do not invent information not present in the summaries.
"""

    TOPIC_PROMPT = """\
Synthesize these {count} document summaries into a coherent overview of: "{topic}"

For each major finding or theme, cite which document(s) it comes from.
Structure your response as:

## Key Themes
- Theme 1: description (Document A, Document B)

## Methods & Approaches
- Approach 1: description (Document A)

## Open Questions & Gaps
- Gap 1: description

## Connections Between Documents
- Document A extends Document B's work by...

Document summaries:
{summaries}
"""

    TARGET_PROMPT = """\
You are analyzing {count} documents for applicability to a specific target.

TARGET: {target_name}
DESCRIPTION: {target_description}

For EACH applicable document, explain SPECIFICALLY how its findings could be
applied. Be concrete — name specific components, methods, or actions.

Structure EXACTLY as:

## Apply Now
For each: Document title → what to do → expected benefit

## Worth Exploring
For each: Document title → idea → what needs validation first

## Background Only
Documents that inform thinking but don't have direct application

Keep each entry to 2-3 sentences max. Do not repeat documents.

DOCUMENTS:
{summaries}
"""

    LANDSCAPE_PROMPT = """\
Analyze these {count} document summaries to map the landscape.

Structure your response as:

## Major Directions
- Direction 1: description, key documents, maturity level

## Emerging Trends
- Trend 1: description, earliest/latest documents

## Consensus Views
- What most documents agree on

## Contested Areas
- Where documents disagree or take different approaches

## Gaps
- What's not being covered but should be

Document summaries:
{summaries}
"""

    def __init__(
        self,
        knowledge: KnowledgeStore,
        triples: TripleStore,
        pool: ModelPool,
        summary_tags: Optional[List[str]] = None,
        summary_scope: str = "",
    ):
        self.knowledge = knowledge
        self.triples = triples
        self.pool = pool
        self.summary_tags = summary_tags or ["summary"]
        self.summary_scope = summary_scope

    def _get_summaries(
        self, query: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get document summaries from the knowledge store.

        Override for domain-specific filtering or formatting.
        """
        if query:
            scope = self.summary_scope or None
            entries = self.knowledge.search(query, scope=scope, limit=limit)
            entries = [
                e for e in entries
                if e.tier == Tier.DERIVED
                and any(t in (e.tags or []) for t in self.summary_tags)
            ]
        else:
            entries = [
                e
                for e in self.knowledge.get_by_tier(Tier.DERIVED)
                if any(t in (e.tags or []) for t in self.summary_tags)
            ][:limit]

        summaries = []
        for entry in entries:
            try:
                data = json.loads(entry.content)
            except (json.JSONDecodeError, TypeError):
                data = {"raw": entry.content}

            summaries.append({
                "entry_id": entry.id,
                "parent_id": entry.metadata.get("parent_id", ""),
                "title": data.get("title", entry.title),
                "summary": data,
                "assessments": entry.metadata.get("assessments", {}),
            })

        return summaries

    def _format_summaries(self, summaries: List[Dict], max_chars: int = 10000) -> str:
        """Format summaries compactly for prompt injection.

        Keeps total under max_chars to avoid exceeding model context.
        """
        parts = []
        total = 0
        for i, s in enumerate(summaries, 1):
            data = s["summary"]
            title = data.get("title", s["title"])
            abstract = data.get("abstract", "")
            findings = data.get("key_findings", [])
            methods = data.get("methods", [])

            lines = [f"{i}. {title}"]
            if abstract:
                lines.append(f"   {abstract[:150]}")
            if findings:
                for f in findings[:3]:
                    lines.append(f"   - {f[:100]}")
            if methods:
                lines.append(f"   Methods: {', '.join(methods)}")

            block = "\n".join(lines)
            if total + len(block) > max_chars:
                parts.append(f"... and {len(summaries) - i + 1} more documents")
                break
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    async def _generate(self, prompt: str) -> str:
        """Run LLM generation via the summarizer model."""
        client = self.pool.get_client("summarizer")
        return await client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=6000,
        )

    async def topic_summary(
        self, topic: str, limit: int = 30
    ) -> SynthesisResult:
        """Synthesize documents around a topic/query."""
        summaries = self._get_summaries(query=topic, limit=limit)
        if not summaries:
            return SynthesisResult(
                query=topic, synthesis_type="topic",
                content="No summaries found.", document_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = self.TOPIC_PROMPT.format(
            count=len(summaries), topic=topic, summaries=formatted,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query=topic,
            synthesis_type="topic",
            content=content,
            document_count=len(summaries),
            document_ids=[s["entry_id"] for s in summaries],
            success=True,
        )

    async def target_brief(
        self, target_name: str, target_description: str, limit: int = 20
    ) -> SynthesisResult:
        """Generate applicability brief for a specific target."""
        summaries = self._get_summaries(limit=limit * 3)
        # Filter to summaries that scored for this target
        scored = [
            s for s in summaries
            if target_name in s.get("assessments", {})
            and isinstance(s["assessments"][target_name], dict)
            and float(s["assessments"][target_name].get("score", 0)) > 0.3
        ]
        if scored:
            scored.sort(
                key=lambda s: float(s.get("assessments", {}).get(target_name, {}).get("score", 0)),
                reverse=True,
            )
            summaries = scored[:limit]
        else:
            summaries = summaries[:limit]
        if not summaries:
            return SynthesisResult(
                query=target_name, synthesis_type="target",
                content="No summaries found.", document_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = self.TARGET_PROMPT.format(
            count=len(summaries),
            target_name=target_name,
            target_description=target_description,
            summaries=formatted,
        )

        content = await self._generate(prompt)
        return SynthesisResult(
            query=target_name,
            synthesis_type="target",
            content=content,
            document_count=len(summaries),
            document_ids=[s["entry_id"] for s in summaries],
            success=True,
        )

    async def landscape(self, limit: int = 50) -> SynthesisResult:
        """Generate a landscape overview across all documents."""
        summaries = self._get_summaries(limit=limit)
        if not summaries:
            return SynthesisResult(
                query="landscape", synthesis_type="landscape",
                content="No summaries found.", document_count=0, success=False,
            )

        formatted = self._format_summaries(summaries)
        prompt = self.LANDSCAPE_PROMPT.format(
            count=len(summaries), summaries=formatted,
        )

        content = await self._generate(prompt)

        triple_ctx = self.triples.build_context(max_triples=30, min_confidence=0.5)
        if triple_ctx:
            content += f"\n\n## Known Relationships\n{triple_ctx}"

        return SynthesisResult(
            query="landscape",
            synthesis_type="landscape",
            content=content,
            document_count=len(summaries),
            document_ids=[s["entry_id"] for s in summaries],
            success=True,
        )
