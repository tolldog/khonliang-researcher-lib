# khonliang-researcher-lib

Reusable research SDK for khonliang agents.

This library contains the shared primitives used by researcher-style agents and by developer workflows that need evidence, synthesis, local document reads, or concept graph operations without depending on the full researcher application.

## What Belongs Here

Use this library for reusable, domain-neutral research mechanics:

- relevance scoring and embedding similarity
- entity and concept graph construction
- concept taxonomy and investigation workspace helpers
- queue-worker scaffolding
- multi-document synthesis base classes
- idea parsing primitives
- best-of-N candidate selection
- local document reading
- vector indexing and reciprocal-rank fusion
- `BaseResearchAgent`, `DomainConfig`, and search engine contracts

Application policy, domain-specific engines, persistent app workflows, and project-specific commands should stay in the application repositories.

## Repository Boundaries

- `khonliang-researcher-lib`: importable research primitives and base contracts
- `khonliang-researcher`: ingestion/distillation app and researcher agent skills
- `khonliang-developer`: FR bundles, milestones, specs, git/GitHub workflow, and repo hygiene
- `khonliang-bus-lib`: bus client, agent base, skill contracts, and service registration
- `khonliang-bus`: running bus service and MCP adapter

When code is useful to both researcher and developer, it belongs here. When code owns live workflow state, keep it in the app repo that owns that state.

## Install

For local development:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

For downstream apps, depend on the GitHub package until release packaging is formalized:

```toml
khonliang-researcher-lib @ git+https://github.com/tolldog/khonliang-researcher-lib.git@main
```

Pin to a commit for reproducible app builds.

## Usage

Import primitives directly:

```python
from khonliang_researcher import EntityNode, LocalDocReader, trace_chain

doc = LocalDocReader().read("docs/repo-hygiene-audit.md")
graph = {
    "researcher": EntityNode(
        name="researcher",
        connections={"developer": ["provides_evidence_to"]},
    ),
    "developer": EntityNode(name="developer"),
}
paths = trace_chain(graph, "researcher", max_depth=2)
```

Build a domain-scoped research agent:

```python
import asyncio

from khonliang_researcher import BaseResearchAgent, DomainConfig


class DeveloperResearchAgent(BaseResearchAgent):
    agent_type = "developer-researcher"
    domain = DomainConfig(
        name="developer",
        rules=["Prefer reusable engineering mechanisms over benchmark-only claims."],
        engines=["web_search"],
    )


asyncio.run(DeveloperResearchAgent.from_cli().start())
```

The agent registers skills through `khonliang-bus-lib`; callers should discover and invoke those skills through the bus.

## Tests

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall khonliang_researcher
```

Optional vector acceleration:

```sh
.venv/bin/python -m pip install -e '.[fast-knn]'
```

Without `sqlite-vec`, vector search falls back to brute-force cosine similarity.

## Hygiene Trace

The current repo hygiene baseline is tracked in `docs/repo-hygiene-audit.md`.
