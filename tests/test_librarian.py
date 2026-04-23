from __future__ import annotations

from dataclasses import dataclass

import pytest

from khonliang_researcher.librarian import (
    AmbiguityRecord,
    GapReport,
    LibrarianStore,
    NeighborhoodSnapshot,
    PaperClassification,
    classify_paper_from_triples,
    identify_gap_candidates,
)


@dataclass
class FakeTriple:
    subject: str
    predicate: str
    object: str
    source: str = ""


def test_librarian_store_round_trips_classification_and_preserves_created_at(tmp_path):
    store = LibrarianStore(str(tmp_path / "librarian.db"))

    first = store.upsert_classification(
        PaperClassification(
            paper_id="paper1",
            classification_code="DR.001",
            audience_tags=["developer-researcher"],
            confidence=0.9,
            rationale="initial",
            source_snapshot_id="snap1",
        )
    )
    second = store.upsert_classification(
        PaperClassification(
            paper_id="paper1",
            classification_code="DR.001",
            audience_tags=["developer-researcher"],
            confidence=0.95,
            rationale="updated",
            source_snapshot_id="snap2",
        )
    )

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    loaded = store.get_classification("paper1")
    assert loaded is not None
    assert loaded.confidence == 0.95
    assert loaded.source_snapshot_id == "snap2"


def test_librarian_store_round_trips_ambiguity_gap_and_snapshot(tmp_path):
    store = LibrarianStore(str(tmp_path / "librarian.db"))

    store.log_ambiguity(
        AmbiguityRecord(
            paper_id="paper1",
            candidates=[{"code": "DR.001", "score": 0.5}, {"code": "UNI.001", "score": 0.45}],
            reason="close scores",
        )
    )
    gap = store.upsert_gap_report(
        GapReport(
            request_id="gap_dr_001",
            topic="code review",
            audience="developer-researcher",
            branch="DR.001",
            rationale="coverage gap",
        )
    )
    snap = store.store_snapshot(
        NeighborhoodSnapshot(
            snapshot_id="snap1",
            audience="developer-researcher",
            artifact_id="art_123",
            reason="rebuild",
            content={"groups": 3},
        )
    )

    assert store.list_ambiguities(status="open")[0].paper_id == "paper1"
    assert store.get_gap_report("gap_dr_001") is not None
    assert store.latest_snapshot("developer-researcher").artifact_id == "art_123"
    health = store.health_summary(total_papers=10)
    assert health["open_gap_count"] == 1
    assert health["snapshot_count"] == 1
    assert gap.request_id == "gap_dr_001"
    assert snap.snapshot_id == "snap1"


def test_classify_paper_from_triples_returns_classification():
    taxonomy = {
        "groups": [
            {"code": "UNI.001", "label": "multi agent review", "audience": "universal"},
            {"code": "DR.001", "label": "multi agent code review", "audience": "developer-researcher"},
        ],
        "entity_groups": {
            "Multi Agent Review": "UNI.001",
            "Multi Agent Code Review": "DR.001",
        },
    }
    triples = [
        FakeTriple("Multi Agent Code Review", "specializes", "Multi Agent Review", source="paper:p1"),
        FakeTriple("Multi Agent Code Review", "applies_to", "Pull Request", source="paper:p1"),
    ]

    result = classify_paper_from_triples("p1", triples, taxonomy, audience="developer-researcher")

    assert result["status"] == "classified"
    assert result["classification_code"] == "DR.001"
    assert "developer-researcher" in result["audience_tags"]


def test_classify_paper_from_triples_returns_ambiguity():
    taxonomy = {
        "groups": [
            {"code": "GEN.001", "label": "code review", "audience": "general"},
            {"code": "GEN.002", "label": "prompt caching", "audience": "general"},
        ],
        "entity_groups": {
            "Code Review": "GEN.001",
            "Prompt Caching": "GEN.002",
        },
    }
    triples = [
        FakeTriple("Code Review", "relates_to", "Prompt Caching", source="paper:p2"),
        FakeTriple("Prompt Caching", "supports", "Code Review", source="paper:p2"),
    ]

    result = classify_paper_from_triples("p2", triples, taxonomy, ambiguity_margin=0.2)

    assert result["status"] == "ambiguous"
    assert len(result["candidates"]) == 2


def test_identify_gap_candidates_returns_undercovered_groups():
    taxonomy = {
        "groups": [
            {"code": "DR.001", "label": "multi agent code review", "audience": "developer-researcher"},
            {"code": "UNI.001", "label": "multi agent review", "audience": "universal"},
        ]
    }
    classifications = [
        PaperClassification(
            paper_id="paper1",
            classification_code="UNI.001",
            audience_tags=["universal"],
        )
    ]

    gaps = identify_gap_candidates(taxonomy, classifications, audience="developer-researcher")

    assert len(gaps) == 1
    assert gaps[0].branch == "DR.001"


def test_identify_gap_candidates_respects_min_papers_threshold():
    taxonomy = {
        "groups": [
            {"code": "DR.001", "label": "multi agent code review", "audience": "developer-researcher"},
            {"code": "DR.002", "label": "tool schema loading", "audience": "developer-researcher"},
        ]
    }
    classifications = [
        PaperClassification(
            paper_id="paper1",
            classification_code="DR.001",
            audience_tags=["developer-researcher"],
        ),
        PaperClassification(
            paper_id="paper2",
            classification_code="DR.001",
            audience_tags=["developer-researcher"],
        ),
        PaperClassification(
            paper_id="paper3",
            classification_code="DR.002",
            audience_tags=["developer-researcher"],
        ),
    ]

    gaps = identify_gap_candidates(
        taxonomy,
        classifications,
        audience="developer-researcher",
        min_papers=2,
    )

    assert len(gaps) == 1
    assert gaps[0].branch == "DR.002"


def test_librarian_store_list_classifications_includes_universal_for_audience(tmp_path):
    store = LibrarianStore(str(tmp_path / "librarian.db"))

    store.upsert_classification(
        PaperClassification(
            paper_id="paper1",
            classification_code="UNI.001",
            audience_tags=["universal"],
            confidence=0.8,
            rationale="universal concept",
            source_snapshot_id="snap1",
        )
    )

    items = store.list_classifications(audience="developer-researcher")

    assert len(items) == 1
    assert items[0].classification_code == "UNI.001"


def test_librarian_store_count_rows_rejects_unknown_table_name(tmp_path):
    store = LibrarianStore(str(tmp_path / "librarian.db"))

    with pytest.raises(ValueError, match="Unsupported table"):
        store._count_rows("librarian_paper_catalog; DROP TABLE --")

    with pytest.raises(ValueError, match="Unsupported table"):
        store._count_rows("librarian_paper_catalog")


def test_classify_paper_from_triples_returns_unclassified_when_no_matches():
    taxonomy = {
        "groups": [
            {"code": "UNI.001", "label": "multi agent review", "audience": "universal"},
        ],
        "entity_groups": {
            "Multi Agent Review": "UNI.001",
        },
    }
    triples = [
        FakeTriple("Unrelated Subject", "relates_to", "Unrelated Object", source="paper:p3"),
    ]

    result = classify_paper_from_triples("p3", triples, taxonomy)

    assert result == {
        "paper_id": "p3",
        "status": "unclassified",
        "reason": "no_taxonomy_entities_found",
        "candidates": [],
    }
