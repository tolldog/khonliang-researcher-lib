---
kind: code_review
severity: concern
---

# Optional-metadata booleans: preserve absent / explicit-false / explicit-true

**Invariant**: boolean fields in optional metadata have three meaningful states: key-present-and-True, key-present-and-False, key-absent. `bool(meta.get("owned_locally", False))` collapses key-absent to False, destroying the distinction between "user said no" and "user didn't say". When the inference path differs from the explicit value, the response shape must preserve which occurred.

**Bad pattern**:
```python
def resolve_owned(meta, path):
    owned = bool(meta.get("owned_locally", False))  # absent == False
    return {"owned_locally": owned}
```

**Good pattern**:
```python
def resolve_owned(meta, path):
    raw = meta.get("owned_locally")
    if raw is None:
        return {
            "owned_locally": path_exists(path),
            "owned_locally_source": "inferred",
        }
    return {
        "owned_locally": bool(raw),
        "owned_locally_source": "explicit",
    }
```

**Rationale**: downstream logic may need to prefer explicit user intent over inferred state; collapsing the three states hides that signal at the boundary. Sourced from PR #16.
