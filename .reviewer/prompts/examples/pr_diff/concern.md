# khonliang-researcher-lib — concern-level invariants

Repo-specific invariants distilled from real cross-vendor review findings
(see PR #16, store-helper primitives). A local hot-tier model tends to catch
type/docstring/dead-code issues but miss these cross-cutting correctness
invariants. Flag a diff at **concern** severity when it violates one of the
patterns below. Each entry is a *canonical* bad/good pair — match by shape,
not by exact identifier.

## universal_audience_wildcard

Audience-filtered list/query methods: a record tagged `universal` matches any
non-empty audience filter. The exclusion test must be
`audience not in tags AND "universal" not in tags`, not `audience not in tags`
alone — otherwise `universal` records are wrongly dropped from every filtered
query.

- Bad: `if audience not in entry.tags: continue`
- Good: `if audience not in entry.tags and "universal" not in entry.tags: continue`

## allowlist_sql_tables

SQL where a table name is derived from a parameter must use an allowlist
mapping (name → fixed SQL), **not** f-string interpolation. Even if every
current caller passes a constant, the API is a latent injection vector.

- Bad: `cur.execute(f"SELECT * FROM {table}")`
- Good: `sql = TABLE_SQL[table]  # KeyError on unknown table` then `cur.execute(sql)`

## prefix_idempotency

When prepending a namespace prefix like `paper:` to an id, check for the
existing prefix first. `f"paper:{paper_id}"` produces `paper:paper:abc` when
the caller already passes `paper:abc` — a silent match failure.

- Bad: `key = f"paper:{paper_id}"`
- Good: `key = paper_id if paper_id.startswith("paper:") else f"paper:{paper_id}"`

## unrounded_score_for_math

Scores used in threshold comparisons (ambiguity margin, confidence buckets)
must stay **unrounded** through the decision path. Round only at display /
serialization time — rounding-then-comparing produces boundary
misclassifications.

- Bad: `if round(score, 2) >= THRESHOLD:`
- Good: `if score >= THRESHOLD:` then round only when formatting for output.
