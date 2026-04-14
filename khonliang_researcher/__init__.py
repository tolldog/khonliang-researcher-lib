"""khonliang_researcher — SDK for building domain-scoped research agents.

Layer 1 — Primitives:
- RelevanceScorer: embedding-based relevance scoring with adaptive learning
- Entity graph: build, traverse, and query relationship graphs
- BaseQueueWorker: background processing with retry tracking
- BaseSynthesizer: multi-document LLM synthesis
- BaseIdeaParser: decompose informal text into claims and search queries
- select_best_of_n: self-distillation for diverse candidate selection
- LocalDocReader: structure-aware reads of local docs (no LLM, no persistence)

Layer 2 — Research agent framework:
- BaseResearchAgent: bus agent with standard research skills (~20 skills built-in)
- DomainConfig: evaluation rules, engines, prompts, output types per domain
- EngineRegistry: pluggable search engines (web_search + web_fetch ship as defaults)

Usage (generic researcher — zero config)::

    from khonliang_researcher import BaseResearchAgent
    agent = BaseResearchAgent.from_cli()
    asyncio.run(agent.start())

Usage (domain researcher — add specialization)::

    from khonliang_researcher import BaseResearchAgent, DomainConfig

    class GenealogyResearcher(BaseResearchAgent):
        agent_type = "genealogy-researcher"
        domain = DomainConfig(
            name="genealogy",
            rules=["Apply the Genealogical Proof Standard"],
            engines=["web_search", "familysearch"],
        )
"""

from khonliang_researcher.relevance import RelevanceScorer, cosine_similarity
from khonliang_researcher.graph import (
    EntityNode,
    build_target_scores,
    build_entity_matrix,
    build_entity_graph,
    format_matrix,
    format_target_tags,
    trace_chain,
    find_paths,
)
from khonliang_researcher.worker import BaseQueueWorker
from khonliang_researcher.synthesizer import BaseSynthesizer, SynthesisResult
from khonliang_researcher.idea_parser import (
    BaseIdeaParser,
    DEFAULT_IDEA_PROMPT,
    clean_for_json,
)
from khonliang_researcher.best_of_n import (
    select_best_of_n,
    DEFAULT_SELECTION_PROMPT,
    serialize_candidates,
)
from khonliang_researcher.doc_reader import (
    LocalDocReader,
    DocContent,
    DEFAULT_REFERENCE_PATTERN,
)
from khonliang_researcher.domain import DomainConfig
from khonliang_researcher.engines import (
    BaseSearchEngine,
    EngineRegistry,
    SearchResult,
    WebFetchEngine,
    WebSearchEngine,
)
from khonliang_researcher.agent import BaseResearchAgent
from khonliang_researcher.vector_index import (
    VectorIndex,
    reciprocal_rank_fusion,
)

# Backward compatibility aliases
ConceptNode = EntityNode
build_project_scores = build_target_scores
build_concept_matrix = build_entity_matrix
build_concept_graph = build_entity_graph
format_project_tags = format_target_tags

__all__ = [
    # Relevance
    "RelevanceScorer",
    "cosine_similarity",
    # Graph
    "EntityNode",
    "build_target_scores",
    "build_entity_matrix",
    "build_entity_graph",
    "format_matrix",
    "format_target_tags",
    "trace_chain",
    "find_paths",
    # Worker
    "BaseQueueWorker",
    # Synthesizer
    "BaseSynthesizer",
    "SynthesisResult",
    # Idea parser
    "BaseIdeaParser",
    "DEFAULT_IDEA_PROMPT",
    "clean_for_json",
    # Best-of-N
    "select_best_of_n",
    "DEFAULT_SELECTION_PROMPT",
    "serialize_candidates",
    # Doc reader
    "LocalDocReader",
    "DocContent",
    "DEFAULT_REFERENCE_PATTERN",
    # Domain config
    "DomainConfig",
    # Engines
    "BaseSearchEngine",
    "EngineRegistry",
    "SearchResult",
    "WebFetchEngine",
    "WebSearchEngine",
    # Research agent
    "BaseResearchAgent",
    # Vector index + retrieval fusion
    "VectorIndex",
    "reciprocal_rank_fusion",
    # Backward compat
    "ConceptNode",
    "build_project_scores",
    "build_concept_matrix",
    "build_concept_graph",
    "format_project_tags",
]
