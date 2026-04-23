---
kind: code_review
severity: concern
---

# Namespace prefixing must be idempotent

**Invariant**: when prepending a namespace prefix like `paper:` to an id, check for the existing prefix first. A blind `f"paper:{paper_id}"` produces `paper:paper:abc` if the caller already passed `paper:abc`, causing silent match failures downstream.

**Bad pattern**:
```python
def key_for(paper_id):
    return f"paper:{paper_id}"  # "paper:paper:abc" if already prefixed
```

**Good pattern**:
```python
_PREFIX = "paper:"

def key_for(paper_id):
    return paper_id if paper_id.startswith(_PREFIX) else f"{_PREFIX}{paper_id}"
```

**Rationale**: callers mix bare ids and pre-namespaced ids across layers; idempotent normalization is cheap insurance. Sourced from PR #16.
