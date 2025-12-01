# Databricks integrations

Helpers for executing code on Databricks clusters, configuring jobs, and working
with SQL and workspace resources.

## Remote compute (`compute/remote.py`)
- `databricks_remote_compute` decorator executes wrapped callables on a target
  cluster with optional package upload, debugger attach parameters, and context
  reuse.
- `remote_invoke` underpins the decorator, handling serialization, upload of the
  caller's package or specified paths, and orchestrating command execution.
- `clear_persisted_context` and `clear_all_persisted_contexts` manage cached
  command contexts when persistence is enabled.

## Jobs configuration (`jobs/config.py`)
- Utilities for composing job dictionaries aligned with Databricks Jobs REST
  semantics, including cluster definitions and task parameters.

## SQL helpers (`sql/`)
- `engine.py` wraps Databricks SQL connections and query execution helpers.
- `exceptions.py` normalizes common SQL error shapes for easier handling.

## Workspace utilities (`workspaces/workspace.py`)
- `DBXWorkspace` simplifies SDK client creation, host configuration, and token
  management for Databricks workspaces.
