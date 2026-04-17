"""Entity graph and matrix views over a knowledge base.

Three views of the same data:

1. **Matrix**: Entities x Documents — which documents cover which entities,
   with relationship types and scores. Good for finding coverage gaps.

2. **Entity Graph**: Network of entities connected through documents.
   Traverse chains like "TSLA → supplies_from → PANASONIC" or
   "GRPO → improved_by → MAGRPO → extends → C3".

3. **Target Tags**: Entities annotated with target relevance scores
   derived from document assessments. Shows which entities matter for
   which downstream targets (stocks, projects, sectors).

All built from TripleStore + KnowledgeStore data.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from khonliang.knowledge.store import KnowledgeStore, Tier
from khonliang.knowledge.triples import TripleStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------

def build_target_scores(
    knowledge: KnowledgeStore,
    triples: TripleStore,
    min_score: float = 0.3,
    source_prefix: str = "paper:",
) -> Dict[str, Dict[str, float]]:
    """Build entity -> {target: score} mapping from document assessments.

    Walks: document assessment scores -> triples -> entities.
    An entity's score for a target = max score across documents it appears in.
    """
    # Step 1: doc_id -> {target: score} from summary metadata
    doc_scores: Dict[str, Dict[str, float]] = {}
    for entry in knowledge.get_by_tier(Tier.DERIVED):
        if "summary" not in (entry.tags or []):
            continue
        assessments = entry.metadata.get("assessments", {})
        parent_id = entry.metadata.get("parent_id", "")
        if not parent_id or not assessments:
            continue
        scores = {}
        for target, assessment in assessments.items():
            if isinstance(assessment, dict):
                score = float(assessment.get("score", 0))
                if score >= min_score:
                    scores[target] = score
        if scores:
            doc_scores[f"{source_prefix}{parent_id}"] = scores

    # Step 2: propagate to entities via triples
    entity_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_triples = triples.get(min_confidence=0.3, limit=5000)

    for t in all_triples:
        source = t.source or ""
        if source not in doc_scores:
            continue
        for target, score in doc_scores[source].items():
            for entity in (t.subject, t.object):
                current = entity_scores[entity].get(target, 0)
                entity_scores[entity][target] = max(current, score)

    return dict(entity_scores)


def format_target_tags(scores: Dict[str, float], threshold: float = 0.4) -> str:
    """Format target scores as compact tags: [TSLA:85% NVDA:72%]"""
    tags = []
    for target, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score >= threshold:
            tags.append(f"{target}:{score:.0%}")
    return f"[{' '.join(tags)}]" if tags else ""


# ---------------------------------------------------------------------------
# Matrix View
# ---------------------------------------------------------------------------

@dataclass
class MatrixCell:
    """A cell in the entity x document matrix."""
    predicates: List[str] = field(default_factory=list)
    confidence: float = 0.0


def build_entity_matrix(
    triples: TripleStore,
    min_confidence: float = 0.5,
    min_connections: int = 2,
    max_entities: int = 50,
    source_prefix: str = "paper:",
) -> Dict[str, Any]:
    """Build an entity x document matrix from the triple store.

    Returns:
        {
            "entities": ["GRPO", "TSLA", ...],
            "documents": ["paper:abc123", ...],
            "matrix": {"GRPO": {"paper:abc123": {"predicates": [...], "confidence": 0.9}}},
            "entity_counts": {"GRPO": 5, ...},
        }
    """
    all_triples = triples.get(min_confidence=min_confidence, limit=5000)

    entity_docs: Dict[str, Dict[str, MatrixCell]] = defaultdict(dict)
    doc_set: Set[str] = set()

    for t in all_triples:
        source = t.source or ""

        if source.startswith(source_prefix):
            if source not in entity_docs[t.subject]:
                entity_docs[t.subject][source] = MatrixCell()
            cell = entity_docs[t.subject][source]
            cell.predicates.append(t.predicate)
            cell.confidence = max(cell.confidence, t.confidence)
            doc_set.add(source)

            if source not in entity_docs[t.object]:
                entity_docs[t.object][source] = MatrixCell()
            cell2 = entity_docs[t.object][source]
            cell2.predicates.append(t.predicate)
            cell2.confidence = max(cell2.confidence, t.confidence)

    filtered = {
        entity: docs
        for entity, docs in entity_docs.items()
        if len(docs) >= min_connections
    }

    sorted_entities = sorted(
        filtered.keys(),
        key=lambda c: len(filtered[c]),
        reverse=True,
    )[:max_entities]

    matrix = {}
    for entity in sorted_entities:
        matrix[entity] = {
            doc: {"predicates": cell.predicates, "confidence": cell.confidence}
            for doc, cell in filtered[entity].items()
        }

    return {
        "entities": sorted_entities,
        "documents": sorted(doc_set),
        "matrix": matrix,
        "entity_counts": {c: len(filtered[c]) for c in sorted_entities},
    }


def format_matrix(
    matrix_data: Dict[str, Any],
    knowledge: KnowledgeStore,
    triples: Optional[TripleStore] = None,
) -> str:
    """Format the matrix as readable text with target tags."""
    entities = matrix_data["entities"]
    matrix = matrix_data["matrix"]
    counts = matrix_data["entity_counts"]

    target_scores: Dict[str, Dict[str, float]] = {}
    if triples:
        target_scores = build_target_scores(knowledge, triples)

    # Resolve document IDs to titles
    doc_titles = {}
    for doc_id in matrix_data["documents"]:
        # Strip any prefix to get the entry ID
        entry_id = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id
        entry = knowledge.get(entry_id)
        if entry:
            doc_titles[doc_id] = entry.title[:40]
        else:
            doc_titles[doc_id] = entry_id[:16]

    lines = [f"## Entity x Document Matrix ({len(entities)} entities, {len(matrix_data['documents'])} documents)\n"]

    for entity in entities[:30]:
        docs = matrix[entity]
        tags = format_target_tags(target_scores.get(entity, {}))
        header = f"### {entity} ({counts[entity]} documents)"
        if tags:
            header += f" {tags}"
        lines.append(header)
        for doc_id, cell in sorted(docs.items(), key=lambda x: -x[1]["confidence"]):
            title = doc_titles.get(doc_id, doc_id)
            preds = ", ".join(set(cell["predicates"]))
            lines.append(f"  [{cell['confidence']:.0%}] {title} — {preds}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity Graph
# ---------------------------------------------------------------------------

@dataclass
class EntityNode:
    """A node in the entity graph."""
    name: str
    document_count: int = 0
    connections: Dict[str, List[str]] = field(default_factory=dict)
    # connections = {"PANASONIC": ["supplies_from"], "RIVN": ["competes_with"]}
    targets: Dict[str, float] = field(default_factory=dict)
    # targets = {"TSLA": 0.85, "tech": 0.72}


@dataclass
class TaxonomyGroup:
    """A deterministic subject group for browsing concept neighborhoods."""
    code: str
    label: str
    audience: str
    entities: List[str] = field(default_factory=list)


@dataclass
class TaxonomyRelationship:
    """Typed relationship between two taxonomy groups."""
    source: str
    predicate: str
    target: str
    confidence: float = 1.0


def build_entity_graph(
    triples: TripleStore,
    min_confidence: float = 0.5,
    knowledge: Optional[KnowledgeStore] = None,
    source_prefix: str = "paper:",
) -> Dict[str, EntityNode]:
    """Build an entity graph from triples.

    Nodes are entities (subjects/objects). Edges are predicates.
    """
    all_triples = triples.get(min_confidence=min_confidence, limit=5000)

    nodes: Dict[str, EntityNode] = {}

    for t in all_triples:
        if t.subject not in nodes:
            nodes[t.subject] = EntityNode(name=t.subject)
        if t.object not in nodes:
            nodes[t.object] = EntityNode(name=t.object)

        subj_node = nodes[t.subject]
        if t.object not in subj_node.connections:
            subj_node.connections[t.object] = []
        if t.predicate not in subj_node.connections[t.object]:
            subj_node.connections[t.object].append(t.predicate)

        if t.source and t.source.startswith(source_prefix):
            subj_node.document_count = max(subj_node.document_count, 1)
            nodes[t.object].document_count = max(nodes[t.object].document_count, 1)

    if knowledge:
        scores = build_target_scores(knowledge, triples, source_prefix=source_prefix)
        for entity, target_scores in scores.items():
            if entity in nodes:
                nodes[entity].targets = target_scores

    return nodes


def build_concept_taxonomy(
    graph: Dict[str, EntityNode],
    *,
    entity_audiences: Optional[Dict[str, str]] = None,
    universal_concepts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a stable audience-scoped taxonomy over graph entities.

    The taxonomy is deliberately lightweight: it groups existing graph nodes,
    assigns deterministic library-style codes, and links audience-specific
    concepts back to universal parents when their token sets specialize a
    universal pattern. Candidate creation and LLM labeling can build on top of
    this primitive without changing the stable return shape.
    """
    entity_audiences = entity_audiences or {}
    universal_labels = [
        label.strip()
        for label in (universal_concepts or ())
        if label and label.strip()
    ]
    universal_tokens = {
        label: _entity_tokens(label)
        for label in universal_labels
        if _entity_tokens(label)
    }

    groups_by_key: Dict[Tuple[str, str], TaxonomyGroup] = {}
    entity_to_group: Dict[str, str] = {}

    for entity in sorted(graph, key=str.lower):
        node = graph[entity]
        audience = _audience_for_entity(entity, node, entity_audiences, universal_tokens)
        label = _taxonomy_label_for_entity(entity)
        key = (audience, label)
        if key not in groups_by_key:
            groups_by_key[key] = TaxonomyGroup(
                code="",
                label=label,
                audience=audience,
                entities=[],
            )
        groups_by_key[key].entities.append(entity)

    groups = sorted(groups_by_key.values(), key=lambda g: (g.audience, g.label))
    for audience, audience_groups in _groups_by_audience(groups).items():
        prefix = _taxonomy_code_prefix(audience)
        for idx, group in enumerate(audience_groups, start=1):
            group.code = f"{prefix}.{idx:03d}"
            for entity in group.entities:
                entity_to_group[entity] = group.code

    relationships = _taxonomy_relationships(groups, universal_tokens)

    return {
        "groups": [
            {
                "code": group.code,
                "label": group.label,
                "audience": group.audience,
                "entities": list(group.entities),
            }
            for group in groups
        ],
        "relationships": [
            {
                "source": rel.source,
                "predicate": rel.predicate,
                "target": rel.target,
                "confidence": rel.confidence,
            }
            for rel in relationships
        ],
        "entity_groups": entity_to_group,
    }


def _normalize_entity_name(name: str) -> str:
    return " ".join(
        name.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )


def _entity_tokens(name: str) -> Set[str]:
    return set(_normalize_entity_name(name).split())


def _audience_for_entity(
    entity: str,
    node: EntityNode,
    entity_audiences: Dict[str, str],
    universal_tokens: Dict[str, Set[str]],
) -> str:
    if entity in entity_audiences:
        return _normalize_audience(entity_audiences[entity])
    normalized = _normalize_entity_name(entity)
    if normalized in {_normalize_entity_name(label) for label in universal_tokens}:
        return "universal"
    if node.targets:
        return _normalize_audience(max(node.targets.items(), key=lambda item: (item[1], item[0]))[0])
    return "general"


def _taxonomy_label_for_entity(entity: str) -> str:
    return _normalize_entity_name(entity)


def _normalize_audience(value: str) -> str:
    normalized = _normalize_entity_name(value).replace(" ", "-")
    return normalized or "general"


def _taxonomy_code_prefix(audience: str) -> str:
    parts = [part for part in audience.replace("_", "-").split("-") if part]
    if not parts:
        return "GEN"
    if len(parts) == 1:
        return parts[0][:3].upper()
    return "".join(part[0].upper() for part in parts[:3])


def _groups_by_audience(groups: List[TaxonomyGroup]) -> Dict[str, List[TaxonomyGroup]]:
    grouped: Dict[str, List[TaxonomyGroup]] = defaultdict(list)
    for group in groups:
        grouped[group.audience].append(group)
    for audience_groups in grouped.values():
        audience_groups.sort(key=lambda group: group.label)
    return dict(grouped)


def _taxonomy_relationships(
    groups: List[TaxonomyGroup],
    universal_tokens: Dict[str, Set[str]],
) -> List[TaxonomyRelationship]:
    by_label_audience = {
        (group.label, group.audience): group
        for group in groups
    }
    relationships: List[TaxonomyRelationship] = []
    seen: Set[Tuple[str, str, str]] = set()
    for group in groups:
        if group.audience == "universal":
            continue
        group_tokens = _entity_tokens(group.label)
        for universal_label, tokens in universal_tokens.items():
            normalized_parent = _normalize_entity_name(universal_label)
            parent = by_label_audience.get((normalized_parent, "universal"))
            if not parent or not tokens or not tokens.issubset(group_tokens):
                continue
            key = (group.code, "specializes", parent.code)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(TaxonomyRelationship(
                source=group.code,
                predicate="specializes",
                target=parent.code,
            ))
    relationships.sort(key=lambda rel: (rel.source, rel.predicate, rel.target))
    return relationships


def suggest_entities(
    graph: Dict[str, EntityNode],
    query: str,
    limit: int = 5,
    min_score: float = 0.25,
    max_query_chars: int = 120,
) -> List[Tuple[str, float]]:
    """Rank existing graph nodes that look related to ``query``.

    Suggestions are intentionally limited to nodes already present in the
    graph. Candidate-node creation belongs in a higher-level neighborhood
    refresh step, while this helper keeps lookups deterministic and cheap.
    """
    normalized_query = _normalize_entity_name(query[:max_query_chars])
    query_tokens = set(normalized_query.split())
    if not normalized_query:
        return []

    scored: List[Tuple[str, float]] = []
    for name, node in graph.items():
        normalized_name = _normalize_entity_name(name)
        name_tokens = _entity_tokens(name)

        if normalized_name == normalized_query:
            score = 1.0
        elif normalized_query in normalized_name or normalized_name in normalized_query:
            score = 0.85
        else:
            overlap = len(query_tokens & name_tokens)
            union = len(query_tokens | name_tokens) or 1
            token_score = overlap / union
            if token_score > 0 or normalized_query[:3] in normalized_name:
                text_score = SequenceMatcher(None, normalized_query, normalized_name).ratio()
            else:
                text_score = 0.0
            score = max(token_score, text_score * 0.7)

        if score < min_score:
            continue

        # Prefer well-connected nodes when textual similarity ties.
        score = min(score + min(len(node.connections), 5) * 0.01, 1.0)
        scored.append((name, score))

    scored.sort(key=lambda item: (-item[1], item[0].lower()))
    return scored[:limit]


def resolve_entity(graph: Dict[str, EntityNode], query: str) -> Optional[str]:
    """Return the canonical graph node name for an exact/case-insensitive query."""
    if query in graph:
        return query
    matches = [name for name in graph if name.lower() == query.lower()]
    return matches[0] if matches else None


def format_entity_suggestions(suggestions: List[Tuple[str, float]]) -> str:
    """Format ranked suggestions for compact CLI/MCP output."""
    if not suggestions:
        return ""
    return "Suggestions: " + ", ".join(
        f"{name} ({score:.0%})" for name, score in suggestions
    )


def trace_chain(
    graph: Dict[str, EntityNode],
    start: str,
    max_depth: int = 4,
    max_branches: int = 3,
    depth: Optional[int] = None,
) -> str:
    """Trace an entity chain from a starting node.

    Returns a tree-like text representation:
        TSLA
        ├── supplies_from → PANASONIC
        │   └── competes_with → LG_ENERGY
        ├── competes_with → RIVN
        └── sector_member → EV
    """
    if depth is not None:
        max_depth = depth

    canonical_start = resolve_entity(graph, start)
    if canonical_start:
        start = canonical_start
    else:
        suggestion_text = format_entity_suggestions(suggest_entities(graph, start))
        if suggestion_text:
            return f"Entity '{start}' not found in graph. {suggestion_text}"
        return f"Entity '{start}' not found in graph."

    root_tags = format_target_tags(graph[start].targets)
    lines = [f"{start} {root_tags}" if root_tags else start]
    visited = {start}
    _trace_recursive(graph, start, lines, visited, "", max_depth, max_branches, 0)
    return "\n".join(lines)


def _trace_recursive(
    graph: Dict[str, EntityNode],
    node_name: str,
    lines: List[str],
    visited: Set[str],
    prefix: str,
    max_depth: int,
    max_branches: int,
    depth: int,
):
    if depth >= max_depth:
        return

    node = graph.get(node_name)
    if not node:
        return

    connections = [
        (target, preds)
        for target, preds in node.connections.items()
        if target not in visited
    ]

    connections.sort(key=lambda x: -len(x[1]))
    connections = connections[:max_branches]

    for i, (target, predicates) in enumerate(connections):
        is_last = i == len(connections) - 1
        branch = "└── " if is_last else "├── "
        continuation = "    " if is_last else "│   "

        pred_str = ", ".join(predicates[:2])
        target_node = graph.get(target)
        tags = format_target_tags(target_node.targets) if target_node else ""
        tag_suffix = f" {tags}" if tags else ""
        lines.append(f"{prefix}{branch}{pred_str} → {target}{tag_suffix}")

        visited.add(target)
        _trace_recursive(
            graph, target, lines, visited,
            prefix + continuation, max_depth, max_branches, depth + 1,
        )


def find_paths(
    graph: Dict[str, EntityNode],
    start: str,
    end: str,
    max_depth: int = 5,
) -> List[List[Tuple[str, str, str]]]:
    """Find paths between two entities.

    Returns list of paths, each path is [(node, predicate, next_node), ...].
    """
    canonical_start = resolve_entity(graph, start)
    canonical_end = resolve_entity(graph, end)
    if not canonical_start or not canonical_end:
        return []

    paths: List[List[Tuple[str, str, str]]] = []
    _find_paths_recursive(graph, canonical_start, canonical_end, [], set(), paths, max_depth)
    return paths


def _find_paths_recursive(
    graph: Dict[str, EntityNode],
    current: str,
    end: str,
    path: List[Tuple[str, str, str]],
    visited: Set[str],
    results: List,
    max_depth: int,
):
    if len(path) >= max_depth:
        return
    if current == end:
        results.append(list(path))
        return

    visited.add(current)
    node = graph.get(current)
    if not node:
        return

    for target, predicates in node.connections.items():
        if target not in visited:
            for pred in predicates[:1]:
                path.append((current, pred, target))
                _find_paths_recursive(
                    graph, target, end, path, visited, results, max_depth,
                )
                path.pop()

    visited.discard(current)
