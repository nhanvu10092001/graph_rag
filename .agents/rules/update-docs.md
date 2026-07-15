---
trigger: model_decision
description: Update CLAUDE.md after completing any significant task so it remains an accurate, up-to-date source of project knowledge for future work.
---

# CLAUDE.md Maintenance Rule

## Description

Keep `CLAUDE.md` up to date after completing any significant task that changes the project's architecture, structure, conventions, workflows, or important implementation details.

## Rule

After completing a task, review whether the work introduces knowledge that would help future development.

If the task changes any of the following, update `CLAUDE.md` before considering the task complete:

* Project architecture
* Directory structure
* Development workflow
* Build, test, or deployment commands
* Coding conventions
* Design decisions
* Important dependencies
* Key modules or responsibilities
* Common troubleshooting notes
* Any project knowledge that future contributors should know

Treat `CLAUDE.md` as the project's living documentation and single source of truth. Every significant change should be reflected there to keep future work efficient and consistent.
