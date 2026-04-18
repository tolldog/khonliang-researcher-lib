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


@dataclass
class InvestigationBranchSpec:
    """A labeled branch in a temporary investigation workspace."""
    label: str
    seeds: List[str] = field(default_factory=list)
    branch_id: str = ""


@dataclass
class InvestigationWorkspace:
    """A branchable, one-way evidence graph over the existing corpus."""
    workspace_id: str
    label: str
    status: str = "active"
    seeds: List[str] = field(default_factory=list)
    branches: List[Dict[str, Any]] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    corpus_refs: List[Dict[str, Any]] = field(default_factory=list)
    one_way_refs: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "label": self.label,
            "status": self.status,
            "seeds": list(self.seeds),
            "branches": list(self.branches),
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "corpus_refs": list(self.corpus_refs),
            "one_way_refs": self.one_way_refs,
            "metadata": dict(self.metadata),
        }


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


def build_investigation_workspace(
    triples: TripleStore,
    seeds: Iterable[str] | str,
    *,
    label: str = "",
    branch_specs: Optional[Iterable[InvestigationBranchSpec | Dict[str, Any] | str]] = None,
    knowledge: Optional[KnowledgeStore] = None,
    min_confidence: float = 0.5,
    max_depth: int = 2,
    max_branches: int = 4,
    source_prefix: str = "paper:",
    workspace_id: str = "",
) -> Dict[str, Any]:
    """Create a temporary branchable evidence graph from corpus triples.

    The returned workspace references corpus documents but does not create or
    mutate corpus triples. This lets an agent investigate a concept, compare
    several branches, and later archive the workspace without polluting the
    long-lived library graph.
    """
    seed_list = _normalize_seed_list(seeds)
    all_triples = [
        t for t in triples.get(min_confidence=min_confidence, limit=5000)
        if getattr(t, "subject", "") and getattr(t, "object", "")
    ]
    graph = build_entity_graph(
        _StaticTripleStore(all_triples),
        min_confidence=min_confidence,
        source_prefix=source_prefix,
    )

    branch_inputs = list(branch_specs or [])
    if not branch_inputs:
        branch_inputs = [InvestigationBranchSpec(label="main", seeds=seed_list)]

    specs = [
        _coerce_branch_spec(spec, default_seeds=seed_list, index=index)
        for index, spec in enumerate(branch_inputs, start=1)
    ]

    workspace_nodes: Dict[str, Dict[str, Any]] = {}
    workspace_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    corpus_refs: Dict[str, Dict[str, Any]] = {}
    branch_summaries: List[Dict[str, Any]] = []
    missing: Dict[str, List[str]] = {}

    adjacency = _triple_adjacency(all_triples)
    for spec in specs:
        branch_id = spec.branch_id or _slug(spec.label, fallback=f"branch-{len(branch_summaries) + 1}")
        branch_seeds = spec.seeds or seed_list
        resolved = []
        missing_for_branch = []
        for seed in branch_seeds:
            canonical = resolve_entity(graph, seed)
            if canonical:
                resolved.append(canonical)
            else:
                missing_for_branch.append(seed)

        branch_nodes, branch_edges = _walk_investigation_branch(
            adjacency,
            resolved,
            max_depth=max(0, int(max_depth)),
            max_branches=max(1, int(max_branches)),
        )

        for node_name in branch_nodes:
            node = graph.get(node_name)
            existing = workspace_nodes.setdefault(node_name, {
                "name": node_name,
                "branch_ids": [],
                "targets": dict(node.targets) if node else {},
                "document_count": node.document_count if node else 0,
            })
            if branch_id not in existing["branch_ids"]:
                existing["branch_ids"].append(branch_id)

        branch_ref_ids: Set[str] = set()
        for triple in branch_edges:
            source = getattr(triple, "source", "") or ""
            if source:
                branch_ref_ids.add(source)
                corpus_refs.setdefault(source, _corpus_ref(source, knowledge))

            key = (
                getattr(triple, "subject"),
                getattr(triple, "predicate", "related_to"),
                getattr(triple, "object"),
            )
            edge = workspace_edges.setdefault(key, {
                "source": key[0],
                "predicate": key[1],
                "target": key[2],
                "confidence": 0.0,
                "branch_ids": [],
                "corpus_refs": [],
            })
            edge["confidence"] = max(edge["confidence"], float(getattr(triple, "confidence", 0.0) or 0.0))
            if branch_id not in edge["branch_ids"]:
                edge["branch_ids"].append(branch_id)
            if source and source not in edge["corpus_refs"]:
                edge["corpus_refs"].append(source)

        if missing_for_branch:
            missing[branch_id] = missing_for_branch

        branch_summaries.append({
            "branch_id": branch_id,
            "label": spec.label,
            "seeds": branch_seeds,
            "resolved_seeds": resolved,
            "missing_seeds": missing_for_branch,
            "node_count": len(branch_nodes),
            "edge_count": len(branch_edges),
            "corpus_refs": sorted(branch_ref_ids),
        })

    resolved_seed_set = sorted({
        seed
        for branch in branch_summaries
        for seed in branch["resolved_seeds"]
    })
    workspace_label = label or ", ".join(seed_list) or "investigation"
    workspace = InvestigationWorkspace(
        workspace_id=workspace_id or f"investigation:{_slug(workspace_label)}",
        label=workspace_label,
        seeds=seed_list,
        branches=branch_summaries,
        nodes=sorted(workspace_nodes.values(), key=lambda node: node["name"].lower()),
        edges=sorted(
            workspace_edges.values(),
            key=lambda edge: (edge["source"].lower(), edge["predicate"], edge["target"].lower()),
        ),
        corpus_refs=sorted(corpus_refs.values(), key=lambda ref: ref["ref_id"]),
        metadata={
            "resolved_seeds": resolved_seed_set,
            "missing_seeds": missing,
            "min_confidence": min_confidence,
            "max_depth": max_depth,
            "max_branches": max_branches,
            "source_prefix": source_prefix,
        },
    )
    return workspace.to_dict()


def archive_investigation_workspace(
    workspace: Dict[str, Any],
    *,
    reason: str = "",
) -> Dict[str, Any]:
    """Return an archived copy of a workspace dict without mutating the input."""
    archived = dict(workspace)
    metadata = dict(archived.get("metadata", {}))
    if reason:
        metadata["archive_reason"] = reason
    archived["metadata"] = metadata
    archived["status"] = "archived"
    return archived


def format_investigation_workspace(workspace: Dict[str, Any], *, detail: str = "brief") -> str:
    """Format an investigation workspace for CLI/MCP output."""
    detail = (detail or "brief").lower()
    branches = workspace.get("branches", [])
    nodes = workspace.get("nodes", [])
    edges = workspace.get("edges", [])
    refs = workspace.get("corpus_refs", [])

    if detail == "compact":
        return (
            f"{workspace.get('workspace_id', '')}|status={workspace.get('status', '')}|"
            f"branches={len(branches)}|nodes={len(nodes)}|edges={len(edges)}|refs={len(refs)}"
        )

    lines = [
        f"Investigation workspace: {workspace.get('label', '')}",
        f"id: {workspace.get('workspace_id', '')}",
        f"status: {workspace.get('status', 'active')}",
        f"branches: {len(branches)} | nodes: {len(nodes)} | edges: {len(edges)} | corpus refs: {len(refs)}",
        "one-way corpus refs: yes" if workspace.get("one_way_refs", True) else "one-way corpus refs: no",
    ]

    missing = workspace.get("metadata", {}).get("missing_seeds", {})
    if missing:
        lines.append("missing seeds:")
        for branch_id, seeds in sorted(missing.items()):
            lines.append(f"  {branch_id}: {', '.join(seeds)}")

    lines.append("")
    lines.append("Branches:")
    for branch in branches:
        lines.append(
            f"- {branch['branch_id']}: {branch['label']} "
            f"({branch['node_count']} nodes, {branch['edge_count']} edges, "
            f"{len(branch['corpus_refs'])} refs)"
        )
        if branch.get("resolved_seeds"):
            lines.append(f"  seeds: {', '.join(branch['resolved_seeds'])}")

    if detail == "full":
        lines.append("")
        lines.append("Edges:")
        for edge in edges[:100]:
            refs_text = f" refs={len(edge.get('corpus_refs', []))}" if edge.get("corpus_refs") else ""
            lines.append(
                f"- {edge['source']} -[{edge['predicate']}]-> {edge['target']} "
                f"({edge['confidence']:.0%}){refs_text}"
            )
        if len(edges) > 100:
            lines.append(f"... {len(edges) - 100} more edges")

        lines.append("")
        lines.append("Corpus refs:")
        for ref in refs[:100]:
            title = ref.get("title") or ref["ref_id"]
            lines.append(f"- {ref['ref_id']}: {title}")
        if len(refs) > 100:
            lines.append(f"... {len(refs) - 100} more refs")

    return "\n".join(lines)


class _StaticTripleStore:
    def __init__(self, triples: List[Any]):
        self._triples = triples

    def get(self, *args: Any, **kwargs: Any) -> List[Any]:
        min_confidence = kwargs.get("min_confidence", 0.0)
        limit = kwargs.get("limit", len(self._triples))
        return [
            t for t in self._triples
            if float(getattr(t, "confidence", 0.0) or 0.0) >= min_confidence
        ][:limit]


def _normalize_seed_list(seeds: Iterable[str] | str) -> List[str]:
    if isinstance(seeds, str):
        raw = seeds.split(",")
    else:
        raw = list(seeds)
    return [str(seed).strip() for seed in raw if str(seed).strip()]


def _coerce_branch_spec(
    spec: InvestigationBranchSpec | Dict[str, Any] | str,
    *,
    default_seeds: List[str],
    index: int,
) -> InvestigationBranchSpec:
    if isinstance(spec, InvestigationBranchSpec):
        return spec
    if isinstance(spec, str):
        label, _, seed_text = spec.partition(":")
        seeds = _normalize_seed_list(seed_text) if seed_text else list(default_seeds)
        return InvestigationBranchSpec(
            label=label.strip() or f"branch {index}",
            seeds=seeds,
        )
    if isinstance(spec, dict):
        label = str(spec.get("label") or spec.get("branch_id") or f"branch {index}")
        seeds_value = spec.get("seeds") or default_seeds
        return InvestigationBranchSpec(
            label=label,
            seeds=_normalize_seed_list(seeds_value),
            branch_id=str(spec.get("branch_id") or ""),
        )
    raise TypeError(f"unsupported branch spec: {type(spec).__name__}")


def _triple_adjacency(triples: List[Any]) -> Dict[str, List[Any]]:
    adjacency: Dict[str, List[Any]] = defaultdict(list)
    for triple in triples:
        adjacency[getattr(triple, "subject")].append(triple)
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda t: (-float(getattr(t, "confidence", 0.0) or 0.0), getattr(t, "object", "")))
    return dict(adjacency)


def _walk_investigation_branch(
    adjacency: Dict[str, List[Any]],
    seeds: List[str],
    *,
    max_depth: int,
    max_branches: int,
) -> Tuple[Set[str], List[Any]]:
    nodes: Set[str] = set(seeds)
    edges: List[Any] = []
    seen_edges: Set[Tuple[str, str, str]] = set()
    queue: List[Tuple[str, int]] = [(seed, 0) for seed in seeds]
    visited: Set[Tuple[str, int]] = set()

    while queue:
        current, depth = queue.pop(0)
        if (current, depth) in visited:
            continue
        visited.add((current, depth))
        if depth >= max_depth:
            continue
        for triple in adjacency.get(current, [])[:max_branches]:
            target = getattr(triple, "object")
            already_seen = target in nodes
            key = (
                getattr(triple, "subject"),
                getattr(triple, "predicate", "related_to"),
                target,
            )
            if key not in seen_edges:
                edges.append(triple)
                seen_edges.add(key)
            nodes.add(target)
            if not already_seen:
                queue.append((target, depth + 1))

    return nodes, edges


def _corpus_ref(source: str, knowledge: Optional[KnowledgeStore]) -> Dict[str, Any]:
    ref = {"ref_id": source}
    if knowledge and ":" in source:
        entry_id = source.split(":", 1)[1]
        entry = knowledge.get(entry_id)
        if entry:
            ref["entry_id"] = entry_id
            ref["title"] = getattr(entry, "title", "")
    return ref


def _slug(value: str, *, fallback: str = "workspace") -> str:
    normalized = _normalize_entity_name(value)
    slug = "-".join(part for part in normalized.split() if part)
    return slug or fallback


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
    universal_tokens: Dict[str, Set[str]] = {}
    for label in universal_labels:
        tokens = _entity_tokens(label)
        if tokens:
            universal_tokens[label] = tokens
    normalized_universal_labels = {
        _normalize_entity_name(label)
        for label in universal_tokens
    }

    groups_by_key: Dict[Tuple[str, str], TaxonomyGroup] = {}
    entity_to_group: Dict[str, str] = {}

    for entity in sorted(graph, key=str.lower):
        node = graph[entity]
        audience = _audience_for_entity(
            entity,
            node,
            entity_audiences,
            normalized_universal_labels,
        )
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
    normalized_universal_labels: Set[str],
) -> str:
    if entity in entity_audiences:
        return _normalize_audience(entity_audiences[entity])
    normalized = _normalize_entity_name(entity)
    if normalized in normalized_universal_labels:
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
