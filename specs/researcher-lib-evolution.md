# researcher-lib Evolution Plan

**Date:** 2026-04-11
**Status:** draft
**Purpose:** Define what researcher-lib becomes — from a set of extracted primitives to the SDK for building domain-scoped research agents.

---

## What researcher-lib is today (v0.1.0)

Seven modules extracted from the researcher app:

| Module | What it does |
|---|---|
| `RelevanceScorer` | Embedding-based relevance scoring via Ollama |
| `EntityGraph` builders | `build_entity_matrix`, `build_entity_graph`, `trace_chain`, `find_paths` |
| `BaseQueueWorker` | Background processing with retry tracking |
| `BaseSynthesizer` | Multi-document LLM synthesis (topic, target, landscape) |
| `BaseIdeaParser` | Decompose informal text into claims + search queries |
| `select_best_of_n` | Self-distillation: N parallel generations, model picks best |
| `LocalDocReader` | Structure-aware local file reads (frontmatter, sections, references) |

These are **cognitive primitives** — they work on any structured text. But they're disconnected. Each consumer (researcher app, developer app) wires them together independently.

## What researcher-lib becomes

The SDK for building domain-scoped research agents. Three layers:

```
Layer 1: Primitives (exists today)
  RelevanceScorer, BaseSynthesizer, BaseQueueWorker,
  BaseIdeaParser, select_best_of_n, LocalDocReader,
  EntityGraph builders

Layer 2: Research agent framework (new)
  BaseResearchAgent, DomainConfig, engine interface,
  pipeline wiring, standard skill registration

Layer 3: Integration hooks (new)
  Bus connectivity (via bus-lib dependency),
  LearningStore integration, KnowledgeRegistry sources,
  expert_knowledge function, prompt builder integration
```

## Layer 1: Primitives (existing, extend)

### Existing — no changes needed
- `RelevanceScorer` — embedding similarity
- `EntityGraph` — concept/entity graph builders
- `BaseQueueWorker` — background processing
- `BaseSynthesizer` — multi-doc synthesis
- `BaseIdeaParser` — text → claims + queries
- `select_best_of_n` — self-distillation
- `LocalDocReader` — structured file reads

### New primitives to add

#### VectorIndex
Persistent embedding storage + search. Wraps sqlite-vec when available, falls back to brute-force cosine.

```python
class VectorIndex:
    def __init__(self, db_path: str, embedder: RelevanceScorer): ...
    async def index(self, entry_id: str, text: str): ...
    async def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...
    async def hybrid_search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...
```

Hybrid search combines FTS5 keyword + embedding similarity via reciprocal rank fusion.

#### ExpertKnowledge
The single function agents call when they need to know something. Retrieves from multiple sources (KnowledgeRegistry), distills query-focused, fits within token budget.

```python
async def expert_knowledge(
    query: str,
    registry: KnowledgeRegistry,
    sources: list[str] = None,
    mode: str = "distilled",        # raw | extracted | distilled
    budget_tokens: int = 2000,
) -> str: ...
```

Replaces scattered per-agent retrieval code. Uses VectorIndex for knowledge/heuristic retrieval, TripleStore for graph queries, KnowledgeRegistry metadata for source selection.

## Layer 2: Research Agent Framework (new)

### DomainConfig

The contract between researcher-lib and domain specializations. Declares how a domain differs from generic:

```python
@dataclass
class DomainConfig:
    name: str                           # "platform-development", "genealogy"
    
    # Evaluation behavior
    rules: list[str] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)
    #   keys: "summarizer", "assessor", "extractor", "idea_parser"
    #   values: path to prompt override file (or inline string)
    
    # Search behavior
    engines: list[str] = field(default_factory=list)
    #   names of registered engines: "arxiv", "web_search", "familysearch"
    
    # Output behavior
    output_type: str = "concept_bundles"
    #   what synergize_concepts returns:
    #   "concept_bundles" (generic), "research_leads" (genealogy), etc.
    
    # Additional knowledge sources
    knowledge_sources: list[dict] = field(default_factory=list)
    #   registered with KnowledgeRegistry at startup
    
    @classmethod
    def from_yaml(cls, path: str) -> "DomainConfig":
        """Load domain section from a config YAML."""
        ...
    
    @classmethod
    def generic(cls) -> "DomainConfig":
        """Default config — no domain specialization."""
        return cls(name="generic")
```

### BaseSearchEngine

Interface for pluggable search engines. Domain researchers add their own.

```python
class BaseSearchEngine(ABC):
    name: str                           # "arxiv", "web_search", "familysearch"
    description: str
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str                         # engine name
    metadata: dict = field(default_factory=dict)
```

Standard engines shipped with researcher-lib:
- `WebSearchEngine` — DuckDuckGo instant answers (no API key needed)
- `WebFetchEngine` — fetch any URL, extract text

Domain-specific engines (NOT in researcher-lib):
- `ArxivEngine` — lives in developer-researcher
- `SemanticScholarEngine` — lives in developer-researcher
- `RSSEngine` — lives in developer-researcher
- `FamilySearchEngine` — lives in genealogy-researcher

### EngineRegistry

Config-driven engine selection. Each BaseResearchAgent instance registers its engines at startup.

```python
class EngineRegistry:
    def register(self, engine: BaseSearchEngine): ...
    def get(self, name: str) -> BaseSearchEngine: ...
    def list_engines(self) -> list[str]: ...
    async def search_all(self, query: str, engines: list[str] = None, max_results: int = 20) -> list[SearchResult]: ...
```

### BaseResearchAgent

The base class every research agent extends. Provides standard skills, pipeline wiring, and the framework for domain specialization.

```python
class BaseResearchAgent(BaseAgent):
    """A research agent with standard skills.
    
    Subclass and set `domain` to specialize. Override nothing
    for a generic researcher. Add engines for domain-specific search.
    """
    
    agent_type = "researcher"
    domain: DomainConfig = DomainConfig.generic()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = None        # built in _setup()
        self.engine_registry = EngineRegistry()
        self.knowledge_registry = KnowledgeRegistry()
    
    async def _setup(self):
        """Called during start(), after bus connection, before skill registration.
        
        Builds the pipeline, registers engines, registers knowledge sources,
        loads domain prompts. Subclasses call super()._setup() then add
        their own engines/sources.
        """
        # Build stores against config db_path
        self.knowledge = KnowledgeStore(self.config.db_path)
        self.triples = TripleStore(self.config.db_path)
        self.digest = DigestStore(self.config.db_path)
        
        # Build model pool from config
        self.pool = ModelPool(self.config.models, ...)
        
        # Build roles with domain prompt overrides
        self.summarizer = SummarizerRole(self.pool, prompt=self.domain.prompts.get("summarizer"))
        self.extractor = ExtractorRole(self.pool, prompt=self.domain.prompts.get("extractor"))
        self.assessor = AssessorRole(self.pool, prompt=self.domain.prompts.get("assessor"))
        self.idea_parser = IdeaParserRole(self.pool, prompt=self.domain.prompts.get("idea_parser"))
        
        # Build relevance scorer + vector index
        self.relevance = RelevanceScorer(...)
        self.vector_index = VectorIndex(self.config.db_path, self.relevance)
        
        # Build expert knowledge
        self.expert = ExpertKnowledge(
            registry=self.knowledge_registry,
            embedder=self.relevance,
            domain_rules=self.domain.rules,
        )
        
        # Register default engines
        self.engine_registry.register(WebSearchEngine())
        self.engine_registry.register(WebFetchEngine())
        
        # Register domain-specific engines from config
        for engine_name in self.domain.engines:
            engine = self.engine_registry.get(engine_name)
            if engine:
                self.engine_registry.register(engine)
        
        # Register knowledge sources
        self._register_default_knowledge_sources()
        for source_config in self.domain.knowledge_sources:
            self.knowledge_registry.register(KnowledgeSource(**source_config))
    
    def register_skills(self) -> list[Skill]:
        """Standard research skills — all generic, no domain assumptions."""
        return [
            # Search
            Skill("find_relevant", "Search the corpus by topic"),
            Skill("knowledge_search", "Full-text search across all stored content"),
            Skill("paper_context", "Build evidence context for prompt injection"),
            
            # Ingestion (single items, on-demand)
            Skill("fetch_paper", "Fetch and store a paper by URL"),
            Skill("ingest_file", "Ingest a local file"),
            Skill("ingest_idea", "Decompose informal text into claims + queries"),
            
            # Distillation
            Skill("start_distillation", "Process pending distillation queue"),
            
            # Ideas pipeline
            Skill("research_idea", "Find papers backing an idea's claims"),
            Skill("brief_idea", "Synthesize claim-by-claim assessment"),
            
            # Synthesis
            Skill("synthesize_topic", "Cross-paper analysis of a theme"),
            Skill("synthesize_project", "Applicability brief for a project"),
            
            # Scoring
            Skill("score_relevance", "Score a paper's relevance to projects"),
            Skill("concepts_for_project", "Concepts ranked by project relevance"),
            
            # Concept bundling (generic — no FRs)
            Skill("synergize_concepts", "Find conceptual connections, return bundles"),
            
            # Exploration
            Skill("concept_tree", "Trace a concept's connections as a tree"),
            Skill("concept_path", "Find how two concepts connect"),
            Skill("concept_matrix", "Entity x document coverage matrix"),
            
            # Graph
            Skill("triple_query", "Query the relationship graph"),
            Skill("triple_add", "Add a relationship triple"),
            Skill("triple_context", "Build context from triples"),
            
            # Ops
            Skill("health_check", "Verify stores, embedder, engines"),
            Skill("catalog", "List available skills"),
        ]
```

### Handler implementations

The skill handlers live in researcher-lib as methods on BaseResearchAgent. They're the generic implementations — domain researchers inherit them unchanged:

```python
class BaseResearchAgent(BaseAgent):
    
    @handler("find_relevant")
    async def handle_find_relevant(self, args):
        results = self.knowledge.search(args["query"], limit=10)
        return format_results(results, args.get("detail", "brief"))
    
    @handler("synergize_concepts")
    async def handle_synergize_concepts(self, args):
        # Generic bundling — returns concept clusters, NOT FRs
        bundles = await self.synthesizer.synergize_concepts(
            min_score=args.get("min_score", 0.5),
            max_concepts=args.get("max_concepts", 10),
        )
        return {"bundles": bundles}
    
    @handler("ingest_idea")
    async def handle_ingest_idea(self, args):
        result = await self.idea_parser.handle(args["text"])
        # ... store in knowledge, return id + claims
    
    # ... ~20 more handlers, all generic
```

## Layer 3: Integration Hooks (new)

### Bus integration

BaseResearchAgent extends BaseAgent (from bus-lib). Bus connection, WebSocket protocol, registration, heartbeat, learnings — all inherited. Domain researchers get bus connectivity for free.

```python
class BaseResearchAgent(BaseAgent):  # ← inherits from bus-lib
    
    async def start(self):
        await self._setup()           # build pipeline, engines, etc.
        await super().start()          # connect to bus, register, message loop
```

On registration, the bus sends back per-model learnings from the LearningStore. BaseResearchAgent loads them into the appropriate roles:

```python
    def _on_register_ack(self, ack):
        super()._on_register_ack(ack)
        if ack.get("learnings"):
            for role_name, data in ack["learnings"].items():
                if hasattr(self, role_name):
                    getattr(self, role_name).inject_learnings(data["rules"])
```

### KnowledgeRegistry integration

BaseResearchAgent registers standard knowledge sources at startup:

```python
    def _register_default_knowledge_sources(self):
        self.knowledge_registry.register(KnowledgeSource(
            name="corpus",
            source_type="vector",
            description="Distilled papers and documents",
            retrieve=self.vector_index.hybrid_search,
            ...
        ))
        self.knowledge_registry.register(KnowledgeSource(
            name="triples",
            source_type="graph",
            description="Concept relationships",
            retrieve=self.triples.query_related,
            ...
        ))
        self.knowledge_registry.register(KnowledgeSource(
            name="heuristics",
            source_type="vector",
            description="Learned rules from outcomes",
            retrieve=self.heuristic_pool.search_by_similarity,
            ...
        ))
```

Domain researchers add their own sources via `DomainConfig.knowledge_sources`.

### Prompt builder integration

BaseResearchAgent wires AgentPromptBuilder with all three layers:

```python
    def _build_prompt_builder(self):
        return AgentPromptBuilder(
            domain_rules=self.domain.rules,      # static (config)
            learnings=self.learnings,             # dynamic (bus)
            expert=self.expert,                   # runtime (per-request RAG)
        )
```

---

## Dependencies

```
khonliang (v0.6.4+)
├── BaseRole, ModelPool, AgentTeam, ConsensusEngine
├── KnowledgeStore, TripleStore, DigestStore, Blackboard
├── HeuristicPool, FeedbackStore
├── KnowledgeRegistry (FR fr_khonliang_ff389306)
├── AgentPromptBuilder (FR fr_khonliang_409d0386)
└── format_response, compact_summary, truncate

khonliang-bus-lib (v0.2.0+)
├── BaseAgent (WebSocket, registration, heartbeat, handlers)
├── Skill, Collaboration, handler decorator
└── BusClient

researcher-lib (this plan)
├── Layer 1: existing primitives + VectorIndex + ExpertKnowledge
├── Layer 2: BaseResearchAgent + DomainConfig + engine framework
└── Layer 3: bus + registry + prompt builder integration
```

## What domain researchers provide (NOT in researcher-lib)

| Concern | Provided by domain researcher, not the lib |
|---|---|
| Domain-specific engines | ArxivEngine, FamilySearchEngine, etc. |
| Domain evaluation rules | GPS for genealogy, architecture standards for dev |
| Domain prompt overrides | How to summarize genealogy sources vs code papers |
| Domain output types | Research leads vs development bundles |
| Domain knowledge sources | GEDCOM files, codebase scans |
| FR generation | developer.process_bundles() — not a researcher concern |

## Migration path

### Phase 1: Add BaseResearchAgent to researcher-lib
- New module: `khonliang_researcher/agent.py`
- Depends on bus-lib (add to pyproject.toml)
- Implements ~22 generic research skills as @handler methods
- DomainConfig dataclass with from_yaml + generic() factory
- BaseSearchEngine + EngineRegistry + WebSearchEngine + WebFetchEngine
- Tests: test_agent.py with AgentTestHarness from bus-lib

### Phase 2: Refactor current researcher to extend BaseResearchAgent
- `researcher/agent.py` changes from `BaseAgent.from_mcp()` wrapper to:
  ```python
  class DeveloperResearcher(BaseResearchAgent):
      domain = DomainConfig(name="developer", engines=["arxiv", ...], ...)
      # adds developer-specific skills on top of base
  ```
- Developer-specific tools (ingest_github, scan_codebase, FR tools) stay in researcher
- Generic tools now come from BaseResearchAgent (inherited)
- server.py (MCP entry point) stays as legacy/debug path

### Phase 3: Developer-researcher as a separate thin config
- Could be a separate entry point in the same repo
- Or its own repo with ~20 lines of agent class
- Filters to the evidence subset (~14 skills from base + 0 developer extensions)

### Phase 4: Generic researcher as CLI in researcher-lib
- `python -m khonliang_researcher.agent --id generic --bus ... --config ...`
- Zero-line agent class — just BaseResearchAgent with defaults
- Proves the SDK works stand-alone

### Phase 5: Genealogy-researcher (first external domain)
- New repo or module
- Extends BaseResearchAgent with genealogy engines + GPS rules
- Validates the DomainConfig framework against a real non-development domain

---

## FR mapping

| This plan section | Existing FR |
|---|---|
| BaseResearchAgent + DomainConfig | fr_researcher_1327f943 (developer-researcher subset) |
| DomainConfig framework | fr_researcher_3332344a (domain-scoped instances) |
| Generic/extension split | fr_researcher_becb6a55 (factor generic from extensions) |
| synergize_concepts (bundling only) | fr_researcher_9ac5f786 (synergize split) |
| VectorIndex | new — not yet FR'd |
| ExpertKnowledge | new — not yet FR'd |
| Engine interface | included in fr_researcher_becb6a55 |

---

## What ships in each version

### v0.2.0 — BaseResearchAgent + DomainConfig
- BaseResearchAgent with ~22 generic skills
- DomainConfig dataclass
- BaseSearchEngine + EngineRegistry + default engines
- synergize_concepts (bundling, no FRs)
- bus-lib dependency added
- Tests with AgentTestHarness

### v0.3.0 — Knowledge integration
- VectorIndex (sqlite-vec + fallback)
- ExpertKnowledge function
- KnowledgeRegistry source registration helpers
- AgentPromptBuilder integration hooks

### v0.4.0 — Domain specialization polish
- Prompt override loading from files
- Domain output type framework
- Domain knowledge source auto-registration
- CLI entry point for generic researcher
