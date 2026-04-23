---
kind: code_review
severity: concern
---

# Parameter-derived SQL table names must go through an allowlist map

**Invariant**: SQL queries where a table/column name is derived from a parameter must use an allowlist mapping (name -> fixed SQL string), not f-string interpolation. Even when current callers pass constants, the API is a latent SQL-injection vector the moment a new caller threads user input through.

**Bad pattern**:
```python
def count(self, table):
    return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
```

**Good pattern**:
```python
_TABLE_SQL = {
    "papers": "SELECT COUNT(*) FROM papers",
    "ideas":  "SELECT COUNT(*) FROM ideas",
}

def count(self, table):
    sql = self._TABLE_SQL.get(table)
    if sql is None:
        raise ValueError(f"unknown table: {table!r}")
    return self.conn.execute(sql).fetchone()[0]
```

**Rationale**: identifier names cannot be parameterized via `?` placeholders; the only safe construction is a fixed-set allowlist. Sourced from PR #16.
