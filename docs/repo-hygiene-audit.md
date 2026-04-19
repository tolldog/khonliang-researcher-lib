# Repo Hygiene Audit

Generated: 1776579548.271988
Repo: `/mnt/dev/ttoll/dev/khonliang-researcher-lib`

## Summary

- 2 docs drift findings, 1 stale/deprecated findings, 3 proposed actions, 0 applied changes
- Python files: 24
- Test files: 12
- Docs files: 2

## Cleanup Plan

- **docs-refresh** [low] Refresh README/CLAUDE/config documentation (`CLAUDE.md`, `README.md`)
  - Docs drift findings indicate setup or architecture guidance is stale or incomplete.
- **review-stale-references** [low] Review stale wording in docs and source comments (`specs/researcher-lib-evolution.md`)
  - Stale terms may be historical, but current guidance should not point at retired milestones or runtimes.
- **write-hygiene-artifact** [low] Write compact repo hygiene artifact (`docs/repo-hygiene-audit.md`)
  - Persist the audit so future sessions can resume without rereading raw files.

## Docs Drift

- [high] `README.md`: README.md is missing. Action: add current setup, workflow, and test guidance
- [medium] `CLAUDE.md`: CLAUDE.md is missing. Action: add repo-specific agent guidance

## Deprecated Or Stale Paths

- [low] `specs/researcher-lib-evolution.md`: Found stale marker 'from_mcp'. Action: review whether this is historical context or current guidance

## Test Plan

- `.venv/bin/python -m pytest -q`
- `.venv/bin/python -m compileall .`
