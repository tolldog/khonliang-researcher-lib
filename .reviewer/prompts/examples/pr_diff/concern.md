# khonliang-researcher-lib — concern-level invariants

Repo-specific invariants distilled from real cross-vendor review history
(store-helper primitives, PR #16). A local hot-tier model tends to catch
type/docstring/dead-code issues but misses these cross-cutting correctness
invariants. Flag a diff at **concern** severity when it violates one of the
patterns below. Each entry is a *canonical* bad/good pair — match by shape,
not by exact identifier.

## universal_audience_wildcard

Audience-filtered list/query methods must treat a record tagged `universal` as
matching any non-empty audience filter. The exclusion test must be
`audience not in tags AND "universal" not in tags`, not `audience not in tags`
alone — otherwise `universal` records vanish from every audience-scoped view.

- Bad: `if audience and audience not in tags: continue`
- Good: `if audience and audience not in tags and "universal" not in tags: continue`. (PR #16)

## no_unchecked_sql_identifiers

SQL where a table/column name is derived from a parameter must never
interpolate an **unchecked** identifier (a latent injection vector the moment a
caller threads user input through). Identifiers can't use `?` placeholders, so
either mitigation is valid: (a) an allowlist mapping (name → fixed SQL), or
(b) strict identifier validation against a fixed pattern *before* interpolation
(this repo's `VectorIndex` validates `table` against
`_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")`). Only **raw,
unvalidated** interpolation is a concern.

- Bad: `cur.execute(f"SELECT COUNT(*) FROM {table}")`  *(table unchecked)*
- Good (allowlist): `sql = _TABLE_SQL.get(table)` → `ValueError` on unknown → `cur.execute(sql)`
- Good (validate): `if not _SAFE_IDENTIFIER.match(table): raise ValueError(...)` then interpolate the validated identifier. (PR #16)

## prefix_idempotency

Prepending a namespace prefix like `paper:` must be idempotent — check for the
existing prefix first. A blind `f"paper:{paper_id}"` produces `paper:paper:abc`
when the caller already passed `paper:abc`, a silent match failure downstream.

- Bad: `return f"paper:{paper_id}"`
- Good: `return paper_id if paper_id.startswith(_PREFIX) else f"{_PREFIX}{paper_id}"`. (PR #16)

## unrounded_score_for_math

Scores used in threshold comparisons (ambiguity margin, confidence bucketing,
ranking) must stay **unrounded** through the decision path; round only at
display/serialization time. Rounding-then-comparing produces boundary
misclassifications.

- Bad: `score = round(raw_score, 2); if score >= 0.5: ...`  *(0.499 → 0.50 crosses the boundary)*
- Good: `if raw_score >= 0.5: ...`; round only inside `serialize()`. (PR #16)

## missing_key_vs_explicit_false

Optional-metadata booleans have three meaningful states: present-True,
present-False, absent. `bool(meta.get("owned_locally", False))` collapses absent
into False, destroying "user said no" vs "user didn't say". When the inference
path differs from an explicit value, the response must preserve which occurred.

- Bad: `owned = bool(meta.get("owned_locally", False))`  *(absent == False)*
- Good: `raw = meta.get("owned_locally")` → if `None`, infer (`path_exists`) and tag `source="inferred"`; else `bool(raw)` with `source="explicit"`. (PR #16)
