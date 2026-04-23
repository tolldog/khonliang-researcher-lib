---
kind: code_review
severity: concern
---

# Scores used in threshold comparisons must stay unrounded

**Invariant**: scores used in threshold comparisons (ambiguity margin, confidence bucketing, ranking) MUST stay unrounded through the decision path. Round only at display / serialization time. Rounding-then-comparing produces boundary misclassifications.

**Bad pattern**:
```python
def bucket(raw_score):
    score = round(raw_score, 2)   # 0.499 -> 0.50
    if score >= 0.5:              # boundary crosses wrong direction
        return "high"
    return "low"
```

**Good pattern**:
```python
def bucket(raw_score):
    if raw_score >= 0.5:
        return "high"
    return "low"

def serialize(raw_score):
    return {"score": round(raw_score, 2), "bucket": bucket(raw_score)}
```

**Rationale**: rounding is a lossy transform; the lossless value is what the math is entitled to. Sourced from PR #16.
