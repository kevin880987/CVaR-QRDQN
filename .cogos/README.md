# .cogos

Cognitive OS state for this project. Two tiers, and which one a thing belongs to
decides whether losing it costs a rebuild or costs the record.

| Tier | What it holds | In version control |
| --- | --- | --- |
| Derived | Views rebuilt from this directory: `index/`, `reports/` | Excluded by `.cogos/.gitignore` |
| Authored | This project's own record of itself | Intended for version control; verify the actual commit and backup |

Do not delete this directory merely because some views are derived. Authored
records may be uncommitted, unsynced, or the only copy. Preserve their exact bytes
and verify recovery before any separately authorized cleanup; Git history alone
does not prove that the current record is recoverable.

`README.md` and `registry.json` are rendered by scanning this directory, so an
edit to either is overwritten on the next render. This applies only to those
generated views, not to the authored records they describe.

Contract: `mind.policy.entries.execution.project_local_state`.
Commands: `mind/skill/entries/ops/project-management/project-state/handler.py`.
