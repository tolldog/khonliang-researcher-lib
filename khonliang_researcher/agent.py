"""Base research agent for the khonliang platform.

Extends ``BaseAgent`` from bus-lib with standard research skills:
search, distill, synthesize, explore, idea pipeline. Domain researchers
subclass and add their own engines, rules, and prompts via ``DomainConfig``.

A generic researcher is just::

    class GenericResearcher(BaseResearchAgent):
        agent_type = "researcher"

    if __name__ == "__main__":
        agent = GenericResearcher.from_cli()
        asyncio.run(agent.start())

A domain researcher adds specialization::

    class GenealogyResearcher(BaseResearchAgent):
        agent_type = "genealogy-researcher"
        domain = DomainConfig(
            name="genealogy",
            rules=["Apply the Genealogical Proof Standard"],
            engines=["google", "familysearch"],
        )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from khonliang.knowledge.store import KnowledgeStore, Tier, EntryStatus
from khonliang.knowledge.triples import TripleStore
from khonliang.digest.store import DigestStore
from khonliang.pool import ModelPool
from khonliang_bus import BaseAgent, Skill, handler

from khonliang_researcher.domain import DomainConfig
from khonliang_researcher.engines import (
    EngineRegistry,
    WebFetchEngine,
    WebSearchEngine,
)
from khonliang_researcher.relevance import RelevanceScorer
from khonliang_researcher.synthesizer import BaseSynthesizer, SynthesisResult

logger = logging.getLogger(__name__)


class BaseResearchAgent(BaseAgent):
    """A research agent with standard skills.

    Subclass and set ``domain`` to specialize. Override nothing for a
    generic researcher. Add engines for domain-specific search.

    The agent wires its own pipeline (stores, models, embedder) from
    the config file passed via ``--config``. Domain rules, prompt
    overrides, and engine selection come from ``DomainConfig``.
    """

    agent_type = "researcher"
    module_name = "khonliang_researcher.agent"
    domain: DomainConfig = DomainConfig.generic()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.engine_registry = EngineRegistry()

        # Built during _setup()
        self.knowledge: Optional[KnowledgeStore] = None
        self.triples: Optional[TripleStore] = None
        self.digest: Optional[DigestStore] = None
        self.pool: Optional[ModelPool] = None
        self.relevance: Optional[RelevanceScorer] = None
        self.synthesizer: Optional[BaseSynthesizer] = None
        self._config_data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Setup (called before registration)
    # ------------------------------------------------------------------

    async def _setup(self) -> None:
        """Build the pipeline from config. Called during start().

        Subclasses should call ``await super()._setup()`` then register
        additional engines or knowledge sources.
        """
        self._config_data = self._load_config()
        config = self._config_data

        # Load domain config from the config file if not set on the class
        if self.domain.is_generic and "domain" in config:
            self.domain = DomainConfig.from_dict(config["domain"])

        # Resolve db_path against config file dir
        config_dir = Path(self.config_path).resolve().parent if self.config_path else Path.cwd()
        db_path = config.get("db_path", "data/researcher.db")
        if not Path(db_path).is_absolute():
            db_path = str((config_dir / db_path).resolve())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        config["db_path"] = db_path

        # Stores
        self.knowledge = KnowledgeStore(db_path)
        self.triples = TripleStore(
            db_path,
            predicate_aliases=config.get("predicate_aliases", {}),
        )
        self.digest = DigestStore(db_path)

        # Model pool
        models = config.get("models", {})
        role_model_map = {
            "summarizer": models.get("summarizer", "qwen2.5:7b"),
            "extractor": models.get("extractor", "llama3.2:3b"),
            "assessor": models.get("assessor", "qwen2.5:7b"),
            "idea_parser": models.get("idea_parser", "llama3.2:3b"),
        }
        self.pool = ModelPool(
            role_model_map,
            base_url=config.get("ollama_url", "http://localhost:11434"),
            model_timeouts=config.get("model_timeouts") or None,
        )

        # Relevance scorer
        self.relevance = RelevanceScorer(
            targets=config.get("projects", {}),
            ollama_url=config.get("ollama_url", "http://localhost:11434"),
            model=models.get("embedder", "nomic-embed-text"),
            threshold=config.get("relevance_threshold", 0.3),
        )

        # Synthesizer
        self.synthesizer = BaseSynthesizer(
            self.knowledge, self.triples, self.pool,
            summary_tags=["summary"],
        )

        # Default engines
        self.engine_registry.register(WebSearchEngine())
        self.engine_registry.register(WebFetchEngine())

        # Domain-specific engines from config
        for engine_name in self.domain.engines:
            if self.engine_registry.has(engine_name):
                continue
            logger.info(
                "Domain engine %r listed but not registered — "
                "subclass must register it in _setup()",
                engine_name,
            )

        logger.info(
            "BaseResearchAgent setup complete: domain=%s, db=%s, engines=%s",
            self.domain.name,
            db_path,
            self.engine_registry.list_engines(),
        )

    def _load_config(self) -> dict[str, Any]:
        """Load YAML config from self.config_path."""
        import yaml

        if not self.config_path:
            return {}
        p = Path(self.config_path)
        if not p.exists():
            logger.warning("Config not found: %s", p)
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Lifecycle override
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Setup pipeline, then connect to bus and handle requests."""
        await self._setup()
        await super().start()

    # ------------------------------------------------------------------
    # Standard research skills
    # ------------------------------------------------------------------

    def register_skills(self) -> list[Skill]:
        """Standard research skills — all generic, no domain assumptions."""
        return [
            # Search
            Skill("find_relevant", "Search the corpus by topic", {
                "query": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("knowledge_search", "Full-text search across stored content", {
                "query": {"type": "string", "required": True},
            }),
            Skill("paper_context", "Build evidence context for prompt injection", {
                "query": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),

            # Ingestion (single items, on-demand)
            Skill("fetch_paper", "Fetch and store a document by URL", {
                "url": {"type": "string", "required": True},
            }),
            Skill("ingest_file", "Ingest a local file", {
                "path": {"type": "string", "required": True},
            }),
            Skill("ingest_idea", "Decompose informal text into claims + queries", {
                "text": {"type": "string", "required": True},
                "source_label": {"type": "string", "default": ""},
            }),

            # Distillation
            Skill("start_distillation", "Process the pending distillation queue", {
                "batch_size": {"type": "integer", "default": 0},
            }),

            # Ideas pipeline
            Skill("research_idea", "Find papers backing an idea's claims", {
                "idea_id": {"type": "string", "required": True},
            }),
            Skill("brief_idea", "Synthesize claim-by-claim assessment", {
                "idea_id": {"type": "string", "required": True},
            }),

            # Synthesis
            Skill("synthesize_topic", "Cross-document analysis of a theme", {
                "topic": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("synthesize_project", "Applicability brief for a project", {
                "project": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),

            # Scoring
            Skill("score_relevance", "Score a document's relevance to projects", {
                "entry_id": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),
            Skill("concepts_for_project", "Concepts ranked by project relevance", {
                "project": {"type": "string", "required": True},
                "detail": {"type": "string", "default": "brief"},
            }),

            # Concept bundling (generic — no FRs)
            Skill("synergize_concepts", "Find conceptual connections, return bundles", {
                "min_score": {"type": "number", "default": 0.5},
                "max_concepts": {"type": "integer", "default": 10},
            }),

            # Exploration
            Skill("concept_tree", "Trace a concept's connections as a tree", {
                "concept": {"type": "string", "required": True},
            }),
            Skill("concept_path", "Find how two concepts connect", {
                "start": {"type": "string", "required": True},
                "end": {"type": "string", "required": True},
            }),

            # Graph
            Skill("triple_query", "Query the relationship graph", {
                "subject": {"type": "string", "required": True},
            }),

            # Ops
            Skill("health_check", "Verify stores, embedder, engines", {}),
        ]

    # ------------------------------------------------------------------
    # Handlers — generic implementations
    # ------------------------------------------------------------------

    @handler("find_relevant")
    async def handle_find_relevant(self, args: dict) -> dict:
        query = args["query"]
        results = self.knowledge.search(query, limit=10)
        if not results:
            return {"result": f"No results for: {query}"}
        entries = [
            {"id": e.id, "title": e.title, "tags": e.tags or []}
            for e in results
        ]
        return {"result": entries}

    @handler("knowledge_search")
    async def handle_knowledge_search(self, args: dict) -> dict:
        results = self.knowledge.search(args["query"], limit=10)
        entries = [
            {"id": e.id, "title": e.title, "tier": e.tier.name}
            for e in results
        ]
        return {"result": entries}

    @handler("paper_context")
    async def handle_paper_context(self, args: dict) -> dict:
        query = args["query"]
        # Build context from relevant papers + triples
        papers = self.knowledge.search(query, limit=5)
        triple_ctx = self.triples.build_context(max_triples=20, min_confidence=0.5)

        parts = []
        for p in papers:
            parts.append(f"[{p.id}] {p.title}\n{p.content[:300]}")
        if triple_ctx:
            parts.append(f"\n## Relationships\n{triple_ctx}")

        return {"result": "\n\n".join(parts) if parts else "No context found."}

    @handler("fetch_paper")
    async def handle_fetch_paper(self, args: dict) -> dict:
        url = args["url"]
        engine = self.engine_registry.get("web_fetch")
        if not engine:
            return {"error": "web_fetch engine not available"}
        results = await engine.search(url)
        if not results:
            return {"error": f"Could not fetch: {url}"}
        r = results[0]
        entry = self.knowledge.add_entry(
            title=r.title,
            content=r.snippet,
            source=url,
            tier=Tier.IMPORTED,
            tags=["paper"],
        )
        return {"result": {"id": entry.id, "title": r.title, "chars": len(r.snippet)}}

    @handler("ingest_file")
    async def handle_ingest_file(self, args: dict) -> dict:
        path = args["path"]
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        text = p.read_text(encoding="utf-8")
        title = p.stem.replace("_", " ").replace("-", " ").title()
        entry = self.knowledge.add_entry(
            title=f"{title} ({p.suffix.lstrip('.')})",
            content=text,
            source=f"file://{p.resolve()}",
            tier=Tier.IMPORTED,
            tags=["paper", f"format:{p.suffix.lstrip('.')}"],
        )
        return {"result": {"id": entry.id, "title": entry.title, "chars": len(text)}}

    @handler("ingest_idea")
    async def handle_ingest_idea(self, args: dict) -> dict:
        from khonliang_researcher.idea_parser import BaseIdeaParser

        parser = BaseIdeaParser(model_pool=self.pool)
        result = await parser.handle(args["text"])
        if not result.get("success"):
            return {"error": result.get("error", "parsing failed")}

        source_label = args.get("source_label", "")
        entry = self.knowledge.add_entry(
            title=result["title"],
            content=json.dumps(result),
            source=f"idea:{source_label}" if source_label else "idea",
            tier=Tier.IMPORTED,
            tags=["idea"],
            metadata={
                "claims": result.get("claims", []),
                "search_queries": result.get("search_queries", []),
                "keywords": result.get("keywords", []),
            },
        )
        return {
            "result": {
                "id": entry.id,
                "title": result["title"],
                "claims": result["claims"],
                "search_queries": result["search_queries"],
            }
        }

    @handler("start_distillation")
    async def handle_start_distillation(self, args: dict) -> dict:
        # Subclasses should override with their full distillation pipeline.
        # Base implementation is a no-op that reports queue status.
        pending = [
            e for e in self.knowledge.get_by_status(EntryStatus.INGESTED, tier=Tier.IMPORTED)
        ]
        return {"result": f"Pending: {len(pending)}. Override start_distillation in your agent for full pipeline."}

    @handler("research_idea")
    async def handle_research_idea(self, args: dict) -> dict:
        idea_id = args["idea_id"]
        entry = self.knowledge.get(idea_id)
        if not entry:
            return {"error": f"Idea {idea_id} not found"}

        queries = (entry.metadata or {}).get("search_queries", [])
        if not queries:
            return {"error": "No search queries in this idea"}

        # Search each query across registered engines
        all_results = []
        for q in queries:
            results = await self.engine_registry.search(q, max_results=5)
            all_results.extend(results)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique.append(r)

        return {
            "result": {
                "idea_id": idea_id,
                "queries_run": len(queries),
                "results_found": len(unique),
                "results": [
                    {"title": r.title, "url": r.url, "source": r.source}
                    for r in unique[:20]
                ],
            }
        }

    @handler("brief_idea")
    async def handle_brief_idea(self, args: dict) -> dict:
        idea_id = args["idea_id"]
        entry = self.knowledge.get(idea_id)
        if not entry:
            return {"error": f"Idea {idea_id} not found"}

        claims = (entry.metadata or {}).get("claims", [])
        if not claims:
            return {"error": "No claims in this idea"}

        # Simple claim listing — full evaluation requires domain-specific LLM prompts
        return {
            "result": {
                "idea_id": idea_id,
                "title": entry.title,
                "claims": claims,
                "note": "Full claim evaluation requires a domain-specific researcher with LLM prompts configured.",
            }
        }

    @handler("synthesize_topic")
    async def handle_synthesize_topic(self, args: dict) -> dict:
        topic = args["topic"]
        result = await self.synthesizer.topic_summary(topic)
        if not result.success:
            return {"error": result.content}
        return {
            "result": {
                "topic": topic,
                "document_count": result.document_count,
                "content": result.content,
            }
        }

    @handler("synthesize_project")
    async def handle_synthesize_project(self, args: dict) -> dict:
        project = args["project"]
        projects = self._config_data.get("projects", {})
        if project not in projects:
            return {"error": f"Project {project!r} not in config"}
        cfg = projects[project]
        result = await self.synthesizer.target_brief(
            project, cfg.get("description", "")
        )
        if not result.success:
            return {"error": result.content}
        return {
            "result": {
                "project": project,
                "document_count": result.document_count,
                "content": result.content,
            }
        }

    @handler("score_relevance")
    async def handle_score_relevance(self, args: dict) -> dict:
        entry_id = args["entry_id"]
        entry = self.knowledge.get(entry_id)
        if not entry:
            return {"error": f"Entry {entry_id} not found"}
        text = f"{entry.title}\n{entry.content[:2000]}"
        scores = {}
        for project, cfg in self._config_data.get("projects", {}).items():
            desc = cfg.get("description", "")
            if desc:
                score = await self.relevance.score(text, desc)
                scores[project] = round(score, 3)
        return {"result": {"entry_id": entry_id, "scores": scores}}

    @handler("concepts_for_project")
    async def handle_concepts_for_project(self, args: dict) -> dict:
        from khonliang_researcher.graph import build_target_scores

        project = args["project"]
        scores = build_target_scores(self.knowledge, self.triples)
        ranked = []
        for concept, proj_scores in scores.items():
            score = proj_scores.get(project, 0)
            if score >= args.get("min_score", 0.4):
                ranked.append({"concept": concept, "score": round(score, 3)})
        ranked.sort(key=lambda x: -x["score"])
        return {"result": ranked[: args.get("limit", 30)]}

    @handler("synergize_concepts")
    async def handle_synergize_concepts(self, args: dict) -> dict:
        """Concept bundling — find what goes together. No FRs."""
        from khonliang_researcher.graph import build_target_scores

        scores = build_target_scores(self.knowledge, self.triples)
        min_score = args.get("min_score", 0.5)
        max_concepts = args.get("max_concepts", 10)

        # Collect top concepts
        all_concepts = []
        for concept, proj_scores in scores.items():
            max_s = max(proj_scores.values()) if proj_scores else 0
            if max_s >= min_score:
                all_concepts.append({
                    "concept": concept,
                    "max_score": round(max_s, 3),
                    "projects": {
                        p: round(s, 3)
                        for p, s in sorted(proj_scores.items(), key=lambda x: -x[1])
                    },
                })
        all_concepts.sort(key=lambda x: -x["max_score"])
        top = all_concepts[:max_concepts]

        # Build bundles using the synthesizer
        if top and self.synthesizer:
            concepts_text = "\n".join(
                f"- {c['concept']} (score: {c['max_score']}, projects: {c['projects']})"
                for c in top
            )
            prompt = (
                f"Group these {len(top)} research concepts into bundles of related ideas.\n"
                f"For each bundle: what concepts belong together, why, and how strong is the connection.\n\n"
                f"Concepts:\n{concepts_text}"
            )
            client = self.pool.get_client("summarizer")
            bundle_text = await client.generate(
                prompt=prompt,
                system="You are a research analyst. Group related concepts into bundles. "
                       "Output structured bundles, not feature requests.",
                temperature=0.3,
                max_tokens=4000,
            )
            return {
                "result": {
                    "concept_count": len(top),
                    "concepts": top,
                    "bundles": bundle_text,
                }
            }

        return {"result": {"concept_count": len(top), "concepts": top, "bundles": ""}}

    @handler("concept_tree")
    async def handle_concept_tree(self, args: dict) -> dict:
        from khonliang_researcher.graph import build_entity_graph, trace_chain

        graph = build_entity_graph(self.knowledge, self.triples)
        chain = trace_chain(graph, args["concept"], depth=3)
        return {"result": chain}

    @handler("concept_path")
    async def handle_concept_path(self, args: dict) -> dict:
        from khonliang_researcher.graph import build_entity_graph, find_paths

        graph = build_entity_graph(self.knowledge, self.triples)
        paths = find_paths(graph, args["start"], args["end"], max_depth=5)
        return {"result": paths}

    @handler("triple_query")
    async def handle_triple_query(self, args: dict) -> dict:
        triples = self.triples.query(subject=args["subject"])
        return {
            "result": [
                {"subject": t.subject, "predicate": t.predicate,
                 "object": t.object, "confidence": t.confidence}
                for t in triples
            ]
        }

    @handler("health_check")
    async def handle_health_check(self, args: dict) -> dict:
        total = len(list(self.knowledge.get_by_tier(Tier.IMPORTED)))
        distilled = len(self.knowledge.get_by_status(EntryStatus.DISTILLED, tier=Tier.IMPORTED))
        engines = self.engine_registry.list_engines()
        return {
            "result": {
                "domain": self.domain.name,
                "db_path": self._config_data.get("db_path", "?"),
                "total_entries": total,
                "distilled": distilled,
                "engines": engines,
                "domain_rules": len(self.domain.rules),
            }
        }
