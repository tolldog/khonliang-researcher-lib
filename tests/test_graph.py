"""Tests for khonliang_researcher.graph."""

from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest

from khonliang_researcher.graph import (
    EntityNode,
    archive_investigation_workspace,
    build_concept_taxonomy,
    build_investigation_workspace,
    build_target_scores,
    build_entity_matrix,
    build_entity_graph,
    format_investigation_workspace,
    format_target_tags,
    suggest_entities,
    trace_chain,
    find_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeTriple:
    subject: str
    predicate: str
    object: str
    source: str = ""
    confidence: float = 0.9


@dataclass
class FakeEntry:
    id: str
    tier: object = None
    title: str = ""
    content: str = ""
    tags: list = None
    metadata: dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


def make_triple_store(triples):
    store = MagicMock()
    store.get.return_value = triples
    return store


def make_knowledge_store(entries):
    store = MagicMock()
    store.get_by_tier.return_value = entries
    store.get.side_effect = lambda id: next((e for e in entries if e.id == id), None)
    return store


# ---------------------------------------------------------------------------
# format_target_tags
# ---------------------------------------------------------------------------

def test_format_target_tags_above_threshold():
    assert format_target_tags({"TSLA": 0.85, "NVDA": 0.72, "low": 0.2}, threshold=0.4) == "[TSLA:85% NVDA:72%]"


def test_format_target_tags_none_above():
    assert format_target_tags({"a": 0.1}, threshold=0.5) == ""


def test_format_target_tags_empty():
    assert format_target_tags({}) == ""


# ---------------------------------------------------------------------------
# build_entity_graph
# ---------------------------------------------------------------------------

def test_build_entity_graph_basic():
    triples = [
        FakeTriple("TSLA", "competes_with", "RIVN", source="article:1"),
        FakeTriple("TSLA", "supplies_from", "PANASONIC", source="article:2"),
    ]
    store = make_triple_store(triples)
    graph = build_entity_graph(store, source_prefix="article:")

    assert "TSLA" in graph
    assert "RIVN" in graph
    assert "PANASONIC" in graph
    assert "RIVN" in graph["TSLA"].connections
    assert "competes_with" in graph["TSLA"].connections["RIVN"]
    assert graph["TSLA"].document_count >= 1


def test_build_entity_graph_source_prefix_filters():
    triples = [
        FakeTriple("A", "rel", "B", source="article:1"),
        FakeTriple("C", "rel", "D", source="other:2"),
    ]
    store = make_triple_store(triples)
    graph = build_entity_graph(store, source_prefix="article:")

    assert graph["A"].document_count >= 1
    assert graph["C"].document_count == 0  # source didn't match prefix


def test_build_entity_graph_no_duplicate_predicates():
    triples = [
        FakeTriple("A", "rel", "B", source="paper:1"),
        FakeTriple("A", "rel", "B", source="paper:2"),
    ]
    store = make_triple_store(triples)
    graph = build_entity_graph(store)

    assert graph["A"].connections["B"] == ["rel"]


def test_build_entity_graph_with_project_scores():
    triples = [
        FakeTriple("concept_a", "uses", "concept_b", source="paper:parent1"),
    ]
    summary = FakeEntry(
        id="sum1",
        tier=MagicMock(value=3),
        tags=["summary"],
        metadata={"parent_id": "parent1", "assessments": {"proj": {"score": 0.8}}},
    )
    triple_store = make_triple_store(triples)
    knowledge = make_knowledge_store([summary])
    knowledge.get_by_tier.return_value = [summary]

    graph = build_entity_graph(triple_store, knowledge=knowledge)
    assert "proj" in graph["concept_a"].targets


def test_build_concept_taxonomy_assigns_stable_audience_codes():
    graph = {
        "Multi Agent Review": EntityNode(name="Multi Agent Review"),
        "Multi Agent Code Review": EntityNode(
            name="Multi Agent Code Review",
            targets={"developer-researcher": 0.9},
        ),
        "General Search": EntityNode(name="General Search"),
    }

    taxonomy = build_concept_taxonomy(
        graph,
        universal_concepts=["Multi Agent Review"],
    )

    groups = {(g["audience"], g["label"]): g for g in taxonomy["groups"]}
    assert groups[("universal", "multi agent review")]["code"] == "UNI.001"
    assert groups[("developer-researcher", "multi agent code review")]["code"] == "DR.001"
    assert groups[("general", "general search")]["code"] == "GEN.001"

    assert {
        "source": "DR.001",
        "predicate": "specializes",
        "target": "UNI.001",
        "confidence": 1.0,
    } in taxonomy["relationships"]
    assert taxonomy["entity_groups"]["Multi Agent Code Review"] == "DR.001"


def test_build_concept_taxonomy_accepts_explicit_entity_audiences():
    graph = {
        "Code Review": EntityNode(name="Code Review"),
        "Pull Request Review": EntityNode(name="Pull Request Review"),
    }

    taxonomy = build_concept_taxonomy(
        graph,
        entity_audiences={"Pull Request Review": "developer researcher"},
    )

    groups = {(g["audience"], g["label"]): g for g in taxonomy["groups"]}
    assert ("developer-researcher", "pull request review") in groups
    assert ("general", "code review") in groups


def test_build_investigation_workspace_creates_branchable_evidence_graph():
    triples = [
        FakeTriple("Claude Optimization", "uses", "Prompt Caching", source="paper:p1", confidence=0.95),
        FakeTriple("Prompt Caching", "reduces", "Token Cost", source="paper:p1", confidence=0.9),
        FakeTriple("Claude Optimization", "uses", "Session Hygiene", source="paper:p2", confidence=0.8),
        FakeTriple("Session Hygiene", "prevents", "Cache Cliff", source="paper:p2", confidence=0.75),
        FakeTriple("Parallel Investigation", "compares", "Prompt Caching", source="paper:p3", confidence=0.7),
    ]
    triple_store = make_triple_store(triples)
    knowledge = make_knowledge_store([
        FakeEntry(id="p1", title="Prompt Caching for Agents"),
        FakeEntry(id="p2", title="Session Hygiene"),
        FakeEntry(id="p3", title="Parallel Research"),
    ])

    workspace = build_investigation_workspace(
        triple_store,
        seeds=["Claude Optimization"],
        label="Claude optimization",
        branch_specs=[
            {"branch_id": "cache", "label": "Cache behavior", "seeds": ["Prompt Caching"]},
            {"branch_id": "sessions", "label": "Session management", "seeds": ["Session Hygiene"]},
        ],
        knowledge=knowledge,
        max_depth=1,
    )

    assert workspace["workspace_id"] == "investigation:claude-optimization"
    assert workspace["one_way_refs"] is True
    assert {branch["branch_id"] for branch in workspace["branches"]} == {"cache", "sessions"}
    assert {edge["source"] for edge in workspace["edges"]} == {"Prompt Caching", "Session Hygiene"}
    assert {ref["ref_id"] for ref in workspace["corpus_refs"]} == {"paper:p1", "paper:p2"}
    assert {ref.get("title") for ref in workspace["corpus_refs"]} == {
        "Prompt Caching for Agents",
        "Session Hygiene",
    }


def test_investigation_workspace_populates_target_scores():
    triples = [
        FakeTriple("Concept", "relates_to", "Neighbor", source="paper:p1"),
    ]
    summary = FakeEntry(
        id="sum1",
        tags=["summary"],
        metadata={"parent_id": "p1", "assessments": {"developer": {"score": 0.8}}},
    )

    workspace = build_investigation_workspace(
        make_triple_store(triples),
        seeds=["Concept"],
        knowledge=make_knowledge_store([summary]),
        max_depth=1,
    )

    concept = next(node for node in workspace["nodes"] if node["name"] == "Concept")
    assert concept["targets"] == {"developer": 0.8}


def test_investigation_workspace_reports_missing_seeds_and_formats():
    triple_store = make_triple_store([
        FakeTriple("Known", "relates_to", "Neighbor", source="paper:p1"),
    ])

    workspace = build_investigation_workspace(
        triple_store,
        seeds="Known",
        branch_specs=["unknown path:Missing Concept"],
    )

    assert workspace["branches"][0]["missing_seeds"] == ["Missing Concept"]
    assert workspace["metadata"]["missing_seeds"] == {"unknown-path": ["Missing Concept"]}
    compact = format_investigation_workspace(workspace, detail="compact")
    assert compact.startswith("investigation:known")
    assert "branches=1" in compact
    brief = format_investigation_workspace(workspace)
    assert "missing seeds:" in brief
    assert "unknown-path: Missing Concept" in brief


def test_archive_investigation_workspace_returns_archived_copy():
    workspace = {
        "workspace_id": "investigation:x",
        "status": "active",
        "branches": [{"branch_id": "main", "seeds": ["A"]}],
        "metadata": {"owner": "researcher", "nested": {"value": 1}},
    }

    archived = archive_investigation_workspace(workspace, reason="done")
    archived["branches"][0]["seeds"].append("B")
    archived["metadata"]["nested"]["value"] = 2

    assert archived["status"] == "archived"
    assert archived["metadata"]["owner"] == "researcher"
    assert archived["metadata"]["archive_reason"] == "done"
    assert workspace["status"] == "active"
    assert workspace["branches"][0]["seeds"] == ["A"]
    assert workspace["metadata"]["nested"]["value"] == 1


# ---------------------------------------------------------------------------
# build_entity_matrix
# ---------------------------------------------------------------------------

def test_build_entity_matrix_basic():
    triples = [
        FakeTriple("concept_a", "uses", "concept_b", source="paper:1"),
        FakeTriple("concept_a", "extends", "concept_c", source="paper:2"),
        FakeTriple("concept_b", "applies_to", "concept_a", source="paper:1"),
    ]
    store = make_triple_store(triples)
    matrix = build_entity_matrix(store, min_connections=2)

    assert "concept_a" in matrix["entities"]
    assert len(matrix["documents"]) > 0


def test_build_entity_matrix_filters_by_min_connections():
    triples = [
        FakeTriple("popular", "rel", "x", source="paper:1"),
        FakeTriple("popular", "rel", "y", source="paper:2"),
        FakeTriple("rare", "rel", "z", source="paper:1"),
    ]
    store = make_triple_store(triples)
    matrix = build_entity_matrix(store, min_connections=2)

    assert "popular" in matrix["entities"]
    assert "rare" not in matrix["entities"]


def test_build_entity_matrix_respects_source_prefix():
    triples = [
        FakeTriple("A", "rel", "B", source="article:1"),
        FakeTriple("A", "rel", "C", source="article:2"),
    ]
    store = make_triple_store(triples)
    matrix = build_entity_matrix(store, min_connections=2, source_prefix="article:")

    assert "A" in matrix["entities"]
    assert all(d.startswith("article:") for d in matrix["documents"])


def test_build_entity_matrix_max_entities():
    triples = []
    for i in range(10):
        for j in range(5):
            triples.append(FakeTriple(f"concept_{i}", "rel", f"other_{j}", source=f"paper:{j}"))
    store = make_triple_store(triples)
    matrix = build_entity_matrix(store, min_connections=1, max_entities=3)

    assert len(matrix["entities"]) == 3


# ---------------------------------------------------------------------------
# trace_chain
# ---------------------------------------------------------------------------

def test_trace_chain_basic():
    graph = {
        "TSLA": EntityNode(name="TSLA", connections={"RIVN": ["competes_with"]}),
        "RIVN": EntityNode(name="RIVN", connections={}),
    }
    result = trace_chain(graph, "TSLA")
    assert "TSLA" in result
    assert "competes_with" in result
    assert "RIVN" in result


def test_trace_chain_not_found():
    graph = {"A": EntityNode(name="A")}
    result = trace_chain(graph, "Z")
    assert "not found" in result


def test_trace_chain_not_found_suggests_existing_nodes():
    graph = {
        "AI Code Review": EntityNode(
            name="AI Code Review",
            connections={"Pull Request Review": ["applies_to"]},
        ),
        "Multi Agent Review": EntityNode(name="Multi Agent Review"),
        "Vector Index": EntityNode(name="Vector Index"),
    }
    result = trace_chain(graph, "code review")
    assert "not found" in result
    assert "Suggestions:" in result
    assert "AI Code Review" in result
    assert "Vector Index" not in result


def test_suggest_entities_ranks_phrase_matches():
    graph = {
        "AI Code Review": EntityNode(name="AI Code Review"),
        "Automated Code Review": EntityNode(name="Automated Code Review"),
        "Vector Index": EntityNode(name="Vector Index"),
    }
    suggestions = suggest_entities(graph, "code review", limit=2)
    assert [name for name, _score in suggestions] == [
        "AI Code Review",
        "Automated Code Review",
    ]


def test_trace_chain_case_insensitive():
    graph = {"TSLA": EntityNode(name="TSLA", connections={})}
    result = trace_chain(graph, "tsla")
    assert "TSLA" in result


def test_trace_chain_depth_limit():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["rel"]}),
        "B": EntityNode(name="B", connections={"C": ["rel"]}),
        "C": EntityNode(name="C", connections={"D": ["rel"]}),
        "D": EntityNode(name="D", connections={}),
    }
    result = trace_chain(graph, "A", max_depth=2)
    assert "C" in result
    assert "D" not in result


def test_trace_chain_branch_limit():
    graph = {
        "A": EntityNode(name="A", connections={
            "B": ["rel"], "C": ["rel"], "D": ["rel"], "E": ["rel"]
        }),
        "B": EntityNode(name="B"), "C": EntityNode(name="C"),
        "D": EntityNode(name="D"), "E": EntityNode(name="E"),
    }
    result = trace_chain(graph, "A", max_branches=2)
    lines = result.strip().split("\n")
    # Root + 2 branches
    assert len(lines) == 3


def test_trace_chain_no_cycles():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["rel"]}),
        "B": EntityNode(name="B", connections={"A": ["rel"]}),
    }
    result = trace_chain(graph, "A", max_depth=10)
    # Should not loop forever — A appears once at root, B once as child
    assert result.count("A") == 1


# ---------------------------------------------------------------------------
# find_paths
# ---------------------------------------------------------------------------

def test_find_paths_direct():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["rel"]}),
        "B": EntityNode(name="B", connections={}),
    }
    paths = find_paths(graph, "A", "B")
    assert len(paths) == 1
    assert paths[0] == [("A", "rel", "B")]


def test_find_paths_case_insensitive():
    graph = {
        "AI Code Review": EntityNode(
            name="AI Code Review",
            connections={"Pull Request Review": ["applies_to"]},
        ),
        "Pull Request Review": EntityNode(name="Pull Request Review", connections={}),
    }
    paths = find_paths(graph, "ai code review", "pull request review")
    assert paths == [[("AI Code Review", "applies_to", "Pull Request Review")]]


def test_find_paths_indirect():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["rel1"]}),
        "B": EntityNode(name="B", connections={"C": ["rel2"]}),
        "C": EntityNode(name="C", connections={}),
    }
    paths = find_paths(graph, "A", "C")
    assert len(paths) == 1
    assert paths[0] == [("A", "rel1", "B"), ("B", "rel2", "C")]


def test_find_paths_no_path():
    graph = {
        "A": EntityNode(name="A", connections={}),
        "B": EntityNode(name="B", connections={}),
    }
    paths = find_paths(graph, "A", "B")
    assert paths == []


def test_find_paths_not_in_graph():
    graph = {"A": EntityNode(name="A")}
    assert find_paths(graph, "A", "Z") == []
    assert find_paths(graph, "Z", "A") == []


def test_find_paths_multiple():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["r1"], "C": ["r2"]}),
        "B": EntityNode(name="B", connections={"D": ["r3"]}),
        "C": EntityNode(name="C", connections={"D": ["r4"]}),
        "D": EntityNode(name="D", connections={}),
    }
    paths = find_paths(graph, "A", "D")
    assert len(paths) == 2


def test_find_paths_depth_limit():
    graph = {
        "A": EntityNode(name="A", connections={"B": ["r"]}),
        "B": EntityNode(name="B", connections={"C": ["r"]}),
        "C": EntityNode(name="C", connections={}),
    }
    paths = find_paths(graph, "A", "C", max_depth=1)
    assert paths == []  # path length 2 > max_depth 1


# ---------------------------------------------------------------------------
# build_target_scores
# ---------------------------------------------------------------------------

def test_build_target_scores_propagates():
    summary = FakeEntry(
        id="sum1",
        tags=["summary"],
        metadata={
            "parent_id": "paper1",
            "assessments": {"proj_a": {"score": 0.7}},
        },
    )
    triples = [
        FakeTriple("concept_x", "uses", "concept_y", source="paper:paper1"),
    ]

    knowledge = make_knowledge_store([summary])
    triple_store = make_triple_store(triples)

    scores = build_target_scores(knowledge, triple_store)
    assert "concept_x" in scores
    assert scores["concept_x"]["proj_a"] == pytest.approx(0.7)
    assert "concept_y" in scores
    assert scores["concept_y"]["proj_a"] == pytest.approx(0.7)


def test_build_target_scores_takes_max():
    summaries = [
        FakeEntry(id="s1", tags=["summary"], metadata={"parent_id": "p1", "assessments": {"proj": {"score": 0.5}}}),
        FakeEntry(id="s2", tags=["summary"], metadata={"parent_id": "p2", "assessments": {"proj": {"score": 0.9}}}),
    ]
    triples = [
        FakeTriple("concept", "rel", "other", source="paper:p1"),
        FakeTriple("concept", "rel", "other2", source="paper:p2"),
    ]

    knowledge = make_knowledge_store(summaries)
    triple_store = make_triple_store(triples)
    scores = build_target_scores(knowledge, triple_store)

    assert scores["concept"]["proj"] == pytest.approx(0.9)


def test_build_target_scores_custom_prefix():
    summary = FakeEntry(
        id="s1", tags=["summary"],
        metadata={"parent_id": "art1", "assessments": {"stock": {"score": 0.6}}},
    )
    triples = [FakeTriple("TSLA", "mentioned_in", "news", source="article:art1")]

    knowledge = make_knowledge_store([summary])
    triple_store = make_triple_store(triples)
    scores = build_target_scores(knowledge, triple_store, source_prefix="article:")

    assert "TSLA" in scores
    assert scores["TSLA"]["stock"] == pytest.approx(0.6)
