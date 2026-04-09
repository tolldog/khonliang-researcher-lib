"""khonliang_researcher — document relevance, entity graphs, and synthesis.

Generic research capabilities for khonliang-based projects:
- RelevanceScorer: embedding-based relevance scoring with adaptive learning
- Entity graph: build, traverse, and query relationship graphs
- BaseQueueWorker: background processing with retry tracking
- BaseSynthesizer: multi-document LLM synthesis
- BaseIdeaParser: decompose informal text into claims and search queries
- select_best_of_n: self-distillation for diverse candidate selection
- LocalDocReader: structure-aware reads of local docs (no LLM, no persistence)
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
    # Backward compat
    "ConceptNode",
    "build_project_scores",
    "build_concept_matrix",
    "build_concept_graph",
    "format_project_tags",
]
