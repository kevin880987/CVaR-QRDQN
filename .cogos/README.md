# .cogos

Cognitive OS state for this project. Two tiers, and which one a thing belongs to
decides whether losing it costs a rebuild or costs the record.

| Tier | What it holds | In version control |
| --- | --- | --- |
| Derived | Views rebuilt from this directory: `index/`, `reports/` | Excluded by `.cogos/.gitignore` |
| Authored | This project's own record of itself | Committed, so every clone carries it |

Deleting the whole directory is safe: the derived half rebuilds and the authored
half is in this repository's history. Deleting the authored half alone loses the
record, because nothing else holds it.

`README.md` and `registry.json` are rendered by scanning this directory, so an
edit to either is overwritten on the next render. Nothing here is hand-maintained.

Contract: `mind.policy.entries.execution.project_local_state`.
Commands: `mind/skill/entries/ops/project-state/handler.py`.
