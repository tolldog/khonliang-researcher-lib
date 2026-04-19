# khonliang-researcher-lib Agent Notes

This repository is the shared importable SDK for research primitives. Keep it small, reusable, and free of application-specific workflow ownership.

When working here:

- Put generic reusable primitives here.
- Keep live FR lifecycle, milestones, specs, git/GitHub workflow, and repo hygiene in `khonliang-developer`.
- Keep ingestion app behavior, paper queues, and researcher-specific skills in `khonliang-researcher`.
- Keep bus transport and skill registration contracts in `khonliang-bus-lib`.
- Do not add local config files, MCP app config, or application databases.
- Preserve backward-compatible aliases unless the consuming apps have already migrated.

Validation:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall khonliang_researcher
```

For docs-only changes, still check that examples reference exported names from `khonliang_researcher.__all__`.
