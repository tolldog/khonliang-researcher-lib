---
kind: code_review
severity: concern
---

# Audience filters must treat `universal` as matching any audience

**Invariant**: audience-filtered list/query methods must treat records tagged `universal` as matching any non-empty audience filter. Otherwise "universal" records vanish the moment a caller scopes by audience.

**Bad pattern**:
```python
def list(self, audience=None):
    out = []
    for r in self._all():
        tags = r.audience_tags or []
        if audience and audience not in tags:
            continue  # excludes records tagged "universal"
        out.append(r)
    return out
```

**Good pattern**:
```python
def list(self, audience=None):
    out = []
    for r in self._all():
        tags = r.audience_tags or []
        if audience and audience not in tags and "universal" not in tags:
            continue
        out.append(r)
    return out
```

**Rationale**: `universal` is a wildcard tag by design; filter code that ignores that semantic silently drops shared records from every audience-scoped view. Sourced from PR #16 (librarian-store).
