---
trigger: model_decision
description: Read this rule before starting any major task or whenever you need a better understanding of the project's source code, architecture, conventions, and development workflow.
---

# CLAUDE.md Bootstrap Instruction

## Mandatory First Step

**Before planning, reasoning, or executing ANY task, ALWAYS look for a `CLAUDE.md` file in the current workspace.**

### If `CLAUDE.md` exists

1. Read the entire file.
2. Treat it as the project's primary source of instructions, conventions, architecture, and context.
3. Follow its guidance unless it directly conflicts with higher-priority system or user instructions.
4. Only after fully understanding `CLAUDE.md` may you begin planning or executing the requested task.

### If `CLAUDE.md` does NOT exist

Before starting the task:

1. Explore the workspace to understand:

   * Project purpose
   * Folder structure
   * Tech stack
   * Build/test commands
   * Coding conventions
   * Important dependencies
   * Existing documentation (`README`, `docs/`, wiki, etc.)
   * Configuration files
   * Development workflow

2. Create a comprehensive `CLAUDE.md` at the workspace root containing at least:

   * Project overview
   * Architecture summary
   * Directory structure
   * Technology stack
   * Setup instructions
   * Build & run commands
   * Test commands
   * Lint/format commands
   * Coding conventions
   * Important design decisions
   * Common workflows
   * Key files and their responsibilities
   * Known assumptions and constraints

3. After creating the file:

   * Use it as the persistent project memory.
   * Keep it updated whenever the project structure, architecture, workflow, or conventions change significantly.

## Ongoing Rule

Whenever a task begins:

1. Check for `CLAUDE.md`.
2. Read it completely.
3. Refresh your understanding of the project.
4. Then plan.
5. Then execute.

**Never skip this process, even if you have previously worked on the same workspace during the conversation.**