"""khonliang_researcher — document relevance, entity graphs, and synthesis.

Generic research capabilities for khonliang-based projects:
- RelevanceScorer: embedding-based relevance scoring with adaptive learning
- Entity graph: build, traverse, and query relationship graphs
- BaseQueueWorker: background processing with retry tracking
- BaseSynthesizer: multi-document LLM synthesis
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
    # Backward compat
    "ConceptNode",
    "build_project_scores",
    "build_concept_matrix",
    "build_concept_graph",
    "format_project_tags",
]
