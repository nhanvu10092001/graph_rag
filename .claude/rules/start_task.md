# Agent Task Workflow Rules

This rule defines mandatory protocols for any AI agent working within this repository (`graph_rag`).

---

## 1. Pre-Task Phase (Preparation Before Starting Work)

Never attempt to modify or create files without first understanding the project structure. Before taking action on any task, the Agent **MUST** complete the following steps:

1. **Read Root `CLAUDE.md` Completely**:
   - Read the entire `CLAUDE.md` file located at the repository root to understand:
     - **Project Overview**: Scope of HLG LLM Utils framework.
     - **Architecture Summary**: Decoupled package dependencies within the `uv` workspace.
     - **Directory Structure**: Layout across Frontend (`FE/`), Backend (`BE/`), and Monorepo packages (`RAG_package/packages/`).
     - **Setup & Running Demos**: Prerequisites, database initializations (PostgreSQL, Neo4j, Qdrant), `make` targets, and Playwright verification scripts.
     - **Development & Test Commands**: Testing (`pytest`), formatting (`black`), linting (`ruff`), and static type checking (`mypy`).
     - **Coding Conventions**: Extensibility via `TaskPlugin`, stateless subagents, service-oriented core, and explicit type hinting.

2. **Scope Assessment**:
   - Identify the exact packages or modules to be touched (e.g., `llm-utils-graph-rag`, `BE/app/services`, `FE/src`).
   - Review relevant configuration files (e.g., `BE/graph_rag_config.yaml`, `RAG_package/pyproject.toml`).
   - If database schemas (PostgreSQL / Neo4j / Qdrant) are affected, check Alembic migration scripts or existing models.

---

## 2. Execution Phase (Implementation Standards)

1. **Adhere to Codebase Conventions**:
   - Maintain the Service-Oriented pattern (delegate business logic to Service classes rather than raw routers/plugins).
   - Ensure comprehensive type hints for all new Python classes, methods, and functions.
   - Prefer modifying existing files over creating duplicate new files.

2. **Verification & Quality Assurance**:
   - Run formatting & linting tools: `ruff check .`, `black .`
   - Perform static type checks: `mypy .`
   - Run relevant test suites (`pytest packages/<package_name>/tests` or specific integration tests).

---

## 3. Post-Task Phase (Updating `CLAUDE.md` Upon Completion)

After completing the task and prior to reporting results to the user, the Agent **MUST** review and update `CLAUDE.md` if the session introduced any of the following changes:

1. **Architecture & New Packages**:
   - Added a package to `RAG_package/packages/` or altered cross-package dependencies -> Update **Architecture Summary**.
2. **Directory Structure**:
   - Added, removed, or renamed major directories/files in `BE/`, `FE/`, or `RAG_package/` -> Update **Directory Structure**.
3. **Dependencies & Setup Commands**:
   - Added dependencies, new `.env` parameters, new demo commands (`make <target>`), or test commands -> Update **Setup & Running Demos**.
4. **Configuration & Conventions**:
   - Modified options in `BE/graph_rag_config.yaml` or introduced new coding guidelines -> Update **Configuration** & **Coding Conventions**.

---

## 4. Agent Self-Verification Checklist

- [ ] Read `CLAUDE.md` prior to starting the task.
- [ ] Identified target modules, files, and relevant config settings.
- [ ] Verified implementation with linters, static type checkers, or test suites.
- [ ] Reviewed and updated `CLAUDE.md` to reflect any architectural, configuration, dependency, or command updates.
