"""Tests for khonliang_researcher.graph."""

from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest

from khonliang_researcher.graph import (
    EntityNode,
    build_target_scores,
    build_entity_matrix,
    build_entity_graph,
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
