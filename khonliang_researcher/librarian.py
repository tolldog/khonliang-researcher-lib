"""Durable library helpers for researcher-librarian workflows."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS librarian_paper_catalog (
    paper_id TEXT PRIMARY KEY,
    classification_code TEXT NOT NULL,
    audience_tags TEXT NOT NULL,
    classification_version TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    source_snapshot_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS librarian_ambiguity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    logged_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS librarian_gap_reports (
    request_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    audience TEXT NOT NULL,
    branch TEXT NOT NULL,
    priority TEXT NOT NULL,
    rationale TEXT NOT NULL,
    suggested_sources_json TEXT NOT NULL,
    detail TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS librarian_neighborhood_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    audience TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    content_json TEXT NOT NULL,
    rebuilt_at REAL NOT NULL
);
"""


@dataclass
class PaperClassification:
    paper_id: str
    classification_code: str
    audience_tags: list[str] = field(default_factory=list)
    classification_version: str = "v1"
    confidence: float = 0.0
    rationale: str = ""
    source_snapshot_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class AmbiguityRecord:
    paper_id: str
    candidates: list[dict[str, Any]]
    reason: str
    status: str = "open"
    logged_at: float = 0.0


@dataclass
class GapReport:
    request_id: str
    topic: str
    audience: str = ""
    branch: str = ""
    priority: str = "medium"
    rationale: str = ""
    suggested_sources: list[str] = field(default_factory=list)
    detail: str = "brief"
    status: str = "open"
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class NeighborhoodSnapshot:
    snapshot_id: str
    audience: str = ""
    artifact_id: str = ""
    reason: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    rebuilt_at: float = 0.0


class LibrarianStore:
    """SQLite-backed durable storage for librarian workflows."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def upsert_classification(self, record: PaperClassification) -> PaperClassification:
        now = time.time()
        created_at = record.created_at or now
        existing = self.get_classification(record.paper_id)
        if existing:
            created_at = existing.created_at
        updated = PaperClassification(
            **{
                **asdict(record),
                "created_at": created_at,
                "updated_at": now,
            }
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO librarian_paper_catalog (
                    paper_id, classification_code, audience_tags,
                    classification_version, confidence, rationale,
                    source_snapshot_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    updated.paper_id,
                    updated.classification_code,
                    json.dumps(updated.audience_tags),
                    updated.classification_version,
                    float(updated.confidence),
                    updated.rationale,
                    updated.source_snapshot_id,
                    float(updated.created_at),
                    float(updated.updated_at),
                ),
            )
        return updated

    def get_classification(self, paper_id: str) -> Optional[PaperClassification]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM librarian_paper_catalog WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        if row is None:
            return None
        return PaperClassification(
            paper_id=row["paper_id"],
            classification_code=row["classification_code"],
            audience_tags=json.loads(row["audience_tags"]),
            classification_version=row["classification_version"],
            confidence=float(row["confidence"]),
            rationale=row["rationale"],
            source_snapshot_id=row["source_snapshot_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def list_classifications(self, audience: str = "") -> list[PaperClassification]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM librarian_paper_catalog ORDER BY paper_id ASC"
            ).fetchall()
        items = []
        for row in rows:
            tags = json.loads(row["audience_tags"])
            if audience and audience not in tags and "universal" not in tags:
                continue
            items.append(
                PaperClassification(
                    paper_id=row["paper_id"],
                    classification_code=row["classification_code"],
                    audience_tags=tags,
                    classification_version=row["classification_version"],
                    confidence=float(row["confidence"]),
                    rationale=row["rationale"],
                    source_snapshot_id=row["source_snapshot_id"],
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                )
            )
        return items

    def log_ambiguity(self, record: AmbiguityRecord) -> AmbiguityRecord:
        logged_at = record.logged_at or time.time()
        updated = AmbiguityRecord(
            paper_id=record.paper_id,
            candidates=list(record.candidates),
            reason=record.reason,
            status=record.status,
            logged_at=logged_at,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO librarian_ambiguity_log (
                    paper_id, candidates_json, reason, status, logged_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    updated.paper_id,
                    json.dumps(updated.candidates),
                    updated.reason,
                    updated.status,
                    float(updated.logged_at),
                ),
            )
        return updated

    def list_ambiguities(self, status: str = "") -> list[AmbiguityRecord]:
        query = "SELECT * FROM librarian_ambiguity_log"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY logged_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            AmbiguityRecord(
                paper_id=row["paper_id"],
                candidates=json.loads(row["candidates_json"]),
                reason=row["reason"],
                status=row["status"],
                logged_at=float(row["logged_at"]),
            )
            for row in rows
        ]

    def upsert_gap_report(self, report: GapReport) -> GapReport:
        now = time.time()
        existing = self.get_gap_report(report.request_id)
        created_at = existing.created_at if existing else (report.created_at or now)
        updated = GapReport(
            **{
                **asdict(report),
                "created_at": created_at,
                "updated_at": now,
            }
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO librarian_gap_reports (
                    request_id, topic, audience, branch, priority,
                    rationale, suggested_sources_json, detail,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    updated.request_id,
                    updated.topic,
                    updated.audience,
                    updated.branch,
                    updated.priority,
                    updated.rationale,
                    json.dumps(updated.suggested_sources),
                    updated.detail,
                    updated.status,
                    float(updated.created_at),
                    float(updated.updated_at),
                ),
            )
        return updated

    def get_gap_report(self, request_id: str) -> Optional[GapReport]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM librarian_gap_reports WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return GapReport(
            request_id=row["request_id"],
            topic=row["topic"],
            audience=row["audience"],
            branch=row["branch"],
            priority=row["priority"],
            rationale=row["rationale"],
            suggested_sources=json.loads(row["suggested_sources_json"]),
            detail=row["detail"],
            status=row["status"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def list_gap_reports(self, audience: str = "", status: str = "") -> list[GapReport]:
        clauses = []
        params: list[Any] = []
        if audience:
            clauses.append("audience = ?")
            params.append(audience)
        if status:
            clauses.append("status = ?")
            params.append(status)
        query = "SELECT * FROM librarian_gap_reports"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            GapReport(
                request_id=row["request_id"],
                topic=row["topic"],
                audience=row["audience"],
                branch=row["branch"],
                priority=row["priority"],
                rationale=row["rationale"],
                suggested_sources=json.loads(row["suggested_sources_json"]),
                detail=row["detail"],
                status=row["status"],
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
            )
            for row in rows
        ]

    def store_snapshot(self, snapshot: NeighborhoodSnapshot) -> NeighborhoodSnapshot:
        rebuilt_at = snapshot.rebuilt_at or time.time()
        updated = NeighborhoodSnapshot(
            snapshot_id=snapshot.snapshot_id,
            audience=snapshot.audience,
            artifact_id=snapshot.artifact_id,
            reason=snapshot.reason,
            content=dict(snapshot.content),
            rebuilt_at=rebuilt_at,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO librarian_neighborhood_snapshots (
                    snapshot_id, audience, artifact_id, reason, content_json, rebuilt_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    updated.snapshot_id,
                    updated.audience,
                    updated.artifact_id,
                    updated.reason,
                    json.dumps(updated.content),
                    float(updated.rebuilt_at),
                ),
            )
        return updated

    def latest_snapshot(self, audience: str = "") -> Optional[NeighborhoodSnapshot]:
        query = "SELECT * FROM librarian_neighborhood_snapshots"
        params: tuple[Any, ...] = ()
        if audience:
            query += " WHERE audience = ?"
            params = (audience,)
        query += " ORDER BY rebuilt_at DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return NeighborhoodSnapshot(
            snapshot_id=row["snapshot_id"],
            audience=row["audience"],
            artifact_id=row["artifact_id"],
            reason=row["reason"],
            content=json.loads(row["content_json"]),
            rebuilt_at=float(row["rebuilt_at"]),
        )

    def health_summary(self, *, total_papers: int = 0) -> dict[str, Any]:
        classifications = self.list_classifications()
        ambiguities = self.list_ambiguities(status="open")
        gaps = self.list_gap_reports(status="open")
        latest = self.latest_snapshot()
        classified_count = len(classifications)
        coverage_pct = (
            round((classified_count / total_papers) * 100, 2)
            if total_papers > 0
            else 0.0
        )
        return {
            "total_papers": total_papers,
            "classified_count": classified_count,
            "coverage_pct": coverage_pct,
            "ambiguous_count": len(ambiguities),
            "open_gap_count": len(gaps),
            "last_rebuilt_at": latest.rebuilt_at if latest else 0.0,
            "snapshot_count": self._count_rows("librarian_neighborhood_snapshots"),
        }

    def _count_rows(self, table: str) -> int:
        queries = {
            "librarian_neighborhood_snapshots": (
                "SELECT COUNT(*) AS c FROM librarian_neighborhood_snapshots"
            ),
        }
        query = queries.get(table)
        if query is None:
            raise ValueError(f"Unsupported table for row count: {table}")
        with self._conn() as conn:
            row = conn.execute(query).fetchone()
        return int(row["c"]) if row else 0


def classify_paper_from_triples(
    paper_id: str,
    triples: Iterable[Any],
    taxonomy: dict[str, Any],
    *,
    audience: str = "",
    ambiguity_margin: float = 0.15,
) -> dict[str, Any]:
    """Classify a paper by counting taxonomy entities in its source triples."""
    entity_groups = dict(taxonomy.get("entity_groups", {}))
    groups_by_code = {group["code"]: group for group in taxonomy.get("groups", [])}
    counts: dict[str, int] = {}
    # Accept both bare ("abc") and already-prefixed ("paper:abc") paper_id values.
    # Document IDs elsewhere in the codebase are typically stored already-prefixed,
    # so callers may pass either form. Double-prepending would produce a phantom
    # "paper:paper:abc" source that silently matches nothing.
    source_id = paper_id if paper_id.startswith("paper:") else f"paper:{paper_id}"
    total_hits = 0

    for triple in triples:
        if getattr(triple, "source", "") != source_id:
            continue
        for entity in (getattr(triple, "subject", ""), getattr(triple, "object", "")):
            code = entity_groups.get(entity)
            if not code:
                continue
            group = groups_by_code.get(code, {})
            group_audience = group.get("audience", "")
            if audience and group_audience not in {audience, "universal"}:
                continue
            counts[code] = counts.get(code, 0) + 1
            total_hits += 1

    if not counts:
        return {
            "paper_id": paper_id,
            "status": "unclassified",
            "reason": "no_taxonomy_entities_found",
            "candidates": [],
        }

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    candidates = []
    for code, count in ranked:
        group = groups_by_code.get(code, {})
        score = count / total_hits if total_hits else 0.0
        # Preserve the unrounded score as the canonical value used for all
        # downstream comparisons (ambiguity margin, confidence, audience tag
        # inclusion). Rounding happens only at display/projection time so
        # near-boundary cases are not reclassified by display precision.
        candidates.append(
            {
                "code": code,
                "label": group.get("label", code),
                "audience": group.get("audience", ""),
                "score": score,
                "count": count,
            }
        )

    top = candidates[0]
    second = candidates[1]["score"] if len(candidates) > 1 else 0.0
    projected_candidates = [
        {**candidate, "score": round(candidate["score"], 4)} for candidate in candidates
    ]
    if len(candidates) > 1 and (top["score"] - second) < ambiguity_margin:
        return {
            "paper_id": paper_id,
            "status": "ambiguous",
            "reason": "close_classification_scores",
            "candidates": projected_candidates,
        }

    return {
        "paper_id": paper_id,
        "status": "classified",
        "classification_code": top["code"],
        "audience_tags": sorted(
            {
                top["audience"],
                *(
                    c["audience"]
                    for c in candidates
                    if c["score"] >= max(top["score"] - ambiguity_margin, 0.0)
                ),
            }
            - {""}
        ),
        "confidence": round(top["score"], 4),
        "rationale": f"classified from {total_hits} taxonomy-linked triple hits",
        "candidates": projected_candidates,
    }


def identify_gap_candidates(
    taxonomy: dict[str, Any],
    classifications: Iterable[PaperClassification],
    *,
    audience: str = "",
    min_papers: int = 1,
) -> list[GapReport]:
    """Identify lightly covered taxonomy groups from existing classifications."""
    group_counts: dict[str, int] = {}
    groups = taxonomy.get("groups", [])

    for item in classifications:
        group_counts[item.classification_code] = group_counts.get(item.classification_code, 0) + 1

    gaps: list[GapReport] = []
    for group in groups:
        if audience and group.get("audience") not in {audience, "universal"}:
            continue
        count = group_counts.get(group["code"], 0)
        if count >= min_papers:
            continue
        request_id = f"gap_{group['code'].lower().replace('.', '_')}"
        gaps.append(
            GapReport(
                request_id=request_id,
                topic=group["label"],
                audience=group.get("audience", ""),
                branch=group["code"],
                priority="medium",
                rationale=f"taxonomy group {group['code']} has {count} classified papers",
                suggested_sources=[],
                detail="brief",
            )
        )
    return gaps
